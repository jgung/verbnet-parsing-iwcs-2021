import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tfnlp.common.config import append_label
from tfnlp.common.constants import ARC_PROBS, DEPREL_KEY, HEAD_KEY, PREDICT_KEY, REL_PROBS, LABEL_SCORES
from tfnlp.common.constants import LABEL_KEY, LENGTH_KEY, MARKER_KEY, SENTENCE_INDEX
from tfnlp.common.utils import binary_np_array_to_unicode


class EvalHook(session_run_hook.SessionRunHook):
    def __init__(self, tensors, evaluator, label_key=LABEL_KEY, predict_key=PREDICT_KEY, output_dir=None,
                 eval_update=None, eval_placeholder=None):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param tensors: dictionary of batch-sized tensors necessary for computing evaluation
        :param evaluator: evaluator to accumulate results and calculate metrics
        :param label_key: targets key in `tensors`
        :param predict_key: predictions key in `tensors`
        :param output_dir: output directory for any predictions or extra logs
        :param eval_update: operation to update current best score
        :param eval_placeholder: placeholder for update operation
        """
        self._fetches = tensors
        self._evaluator = evaluator
        self._target = evaluator.target
        self._label_key = label_key
        self._predict_key = predict_key
        self._output_dir = output_dir
        self._eval_update = eval_update
        self._eval_placeholder = eval_placeholder

    def begin(self):
        self._evaluator.start()

    def before_run(self, run_context):
        return SessionRunArgs(fetches=self._fetches)

    def _create_instances(self, run_context, run_values):
        lengths = run_values.results[LENGTH_KEY]
        indices = run_values.results[SENTENCE_INDEX]

        instances = [{SENTENCE_INDEX: index, LENGTH_KEY: length} for index, length in zip(indices, lengths)]
        results = [{} for _ in indices]

        return instances, results

    def after_run(self, run_context, run_values):
        for instance, result in zip(*self._create_instances(run_context, run_values)):
            self._evaluator.accumulate(instance, result)

    def end(self, session):
        step = session.run(tf.train.get_global_step(session.graph))
        self._evaluator.evaluate(str(step))


class ClassifierEvalHook(EvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._score_name = append_label(LABEL_SCORES, self._target.name)

    def _create_instances(self, run_context, run_values):
        instances, results = super()._create_instances(run_context, run_values)

        constraint_keys = run_values.results[self._target.constraint_key] if self._target.constraints else [None] * len(instances)
        for instance, result, gold, prediction, scores, constraint_key in zip(instances,
                                                                              results,
                                                                              run_values.results[self._label_key],
                                                                              run_values.results[self._predict_key],
                                                                              run_values.results[LABEL_SCORES],
                                                                              constraint_keys):
            instance[self._target.key] = self._target.index_to_feat(gold)
            result[self._target.name] = self._target.index_to_feat(prediction)
            result[self._score_name] = scores
            if self._target.constraints:
                instance[self._target.constraint_key] = constraint_key.decode('utf-8')

        return instances, results

    def end(self, session):
        super().end(session)
        session.run(self._eval_update, feed_dict={self._eval_placeholder: self._evaluator.metric})


class SequenceEvalHook(EvalHook):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_instances(self, run_context, run_values):
        instances, results = super()._create_instances(run_context, run_values)
        for instance, result, gold, prediction in zip(instances,
                                                      results,
                                                      run_values.results[self._label_key],
                                                      run_values.results[self._predict_key]):
            seq_len = instance[LENGTH_KEY]
            instance[self._target.key] = [self._target.index_to_feat(val) for val in gold][:seq_len]
            result[self._target.name] = [self._target.index_to_feat(val) for val in prediction][:seq_len]
        return instances, results

    def end(self, session):
        super().end(session)
        session.run(self._eval_update, feed_dict={self._eval_placeholder: self._evaluator.metric})


class SrlEvalHook(SequenceEvalHook):
    def __init__(self, *args, output_confusions=False, **kwargs):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param output_confusions: display a confusion matrix along with normal outputs
        """
        super().__init__(*args, **kwargs)
        self._output_confusions = output_confusions

    def _create_instances(self, run_context, run_values):
        instances, results = super()._create_instances(run_context, run_values)
        for instance, markers in zip(instances, run_values.results[MARKER_KEY]):
            seq_len = instance[LENGTH_KEY]
            instance[MARKER_KEY] = binary_np_array_to_unicode(markers[:seq_len])
        return instances, results

    def end(self, session):
        super().end(session)


class ParserEvalHook(EvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_instances(self, run_context, run_values):
        instances, results = super()._create_instances(run_context, run_values)
        for instance, result, arc_probs, rel_probs, rels, heads in zip(instances,
                                                                       results,
                                                                       run_values.results[ARC_PROBS],
                                                                       run_values.results[REL_PROBS],
                                                                       run_values.results[DEPREL_KEY],
                                                                       run_values.results[HEAD_KEY]):
            seq_len = instance[LENGTH_KEY]
            result[ARC_PROBS] = arc_probs[:seq_len, :seq_len]
            result[REL_PROBS] = rel_probs[:seq_len, :, :seq_len]
            # we add the head in the evaluator, so don't include twice
            instance[DEPREL_KEY] = binary_np_array_to_unicode(rels[1:seq_len])
            instance[HEAD_KEY] = heads[1:seq_len].tolist()
        return instances, results

    def end(self, session):
        super().end(session)
        session.run(self._eval_update, feed_dict={self._eval_placeholder: self._evaluator.metric})


def metric_compare_fn(metric_key):
    def _metric_compare_fn(best_eval_result, current_eval_result):
        if not best_eval_result or metric_key not in best_eval_result:
            raise ValueError(
                'best_eval_result cannot be empty or no loss %s is found in it.' % metric_key)

        if not current_eval_result or metric_key not in current_eval_result:
            raise ValueError(
                'current_eval_result cannot be empty or no loss %s is found in it.', metric_key)

        new = current_eval_result[metric_key]
        prev = best_eval_result[metric_key]
        tf.logging.info("Comparing new score with previous best (%f vs. %f)", new, prev)
        return prev <= new

    return _metric_compare_fn
