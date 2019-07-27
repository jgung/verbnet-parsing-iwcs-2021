import os

import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from tfnlp.common.constants import ARC_PROBS, DEPREL_KEY, HEAD_KEY, PREDICT_KEY, REL_PROBS, LABEL_SCORES
from tfnlp.common.constants import LABEL_KEY, LENGTH_KEY, MARKER_KEY, SENTENCE_INDEX
from tfnlp.common.eval import PREDICTIONS_FILE, append_srl_prediction_output, GOLD_FILE
from tfnlp.common.eval import accuracy_eval, conll_eval, conll_srl_eval, write_props_to_file, parser_write_and_eval
from tfnlp.common.utils import binary_np_array_to_unicode


class EvalHook(session_run_hook.SessionRunHook):
    def __init__(self, tensors, vocab, label_key=LABEL_KEY, predict_key=PREDICT_KEY, output_dir=None):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param tensors: dictionary of batch-sized tensors necessary for computing evaluation
        :param vocab: label feature vocab
        :param label_key: targets key in `tensors`
        :param predict_key: predictions key in `tensors`
        :param output_dir: output directory for any predictions or extra logs
        """
        self._fetches = tensors
        self._vocab = vocab
        self._label_key = label_key
        self._predict_key = predict_key
        self._output_dir = output_dir

        self._predictions = None
        self._gold = None
        self._indices = None

    def begin(self):
        self._predictions = []
        self._gold = []
        self._indices = []

    def before_run(self, run_context):
        return SessionRunArgs(fetches=self._fetches)

    def after_run(self, run_context, run_values):
        self._indices.extend(run_values.results[SENTENCE_INDEX])


class ClassifierEvalHook(EvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scores = None
        self._constraint_keys = None

    def begin(self):
        super().begin()
        if self._vocab.constraints:
            self._scores = []
            self._constraint_keys = []

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        self._gold.extend([self._vocab.index_to_feat(gold) for gold in run_values.results[self._label_key]])
        self._predictions.extend([self._vocab.index_to_feat(prediction) for prediction in run_values.results[self._predict_key]])

        if self._vocab.constraints:
            for scoring in run_values.results[LABEL_SCORES]:
                self._scores.append({self._vocab.index_to_feat(i): score for i, score in enumerate(scoring)})
            self._constraint_keys.extend(binary_np_array_to_unicode(run_values.results[self._vocab.constraint_key]))

    def end(self, session):
        if self._vocab.constraints:
            self._predictions = []
            for scoring, ck in zip(self._scores, self._constraint_keys):
                valid_scores = {label: score for label, score in scoring.items() if label in self._vocab.constraints.get(ck, [])}
                label = max(valid_scores.items(), key=lambda x: x[1], default=('<UNK>', 0))[0]
                self._predictions.append(label)

        accuracy_eval(self._gold,
                      self._predictions,
                      self._indices,
                      output_file=os.path.join(self._output_dir, PREDICTIONS_FILE))


class SequenceEvalHook(EvalHook):

    def __init__(self, *args, eval_update=None, eval_placeholder=None, **kwargs):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param eval_update: operation to update current best score
        :param eval_placeholder: placeholder for update operation
        """
        super().__init__(*args, **kwargs)
        self._eval_update = eval_update
        self._eval_placeholder = eval_placeholder

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        for gold, predictions, seq_len in zip(run_values.results[self._label_key],
                                              run_values.results[self._predict_key],
                                              run_values.results[LENGTH_KEY]):
            self._gold.append([self._vocab.index_to_feat(val) for val in gold][:seq_len])
            self._predictions.append([self._vocab.index_to_feat(val) for val in predictions][:seq_len])

    def end(self, session):
        score, result = conll_eval(self._gold, self._predictions, self._indices, os.path.join(self._output_dir, PREDICTIONS_FILE))
        tf.logging.info(result)
        session.run(self._eval_update, feed_dict={self._eval_placeholder: score})


class SrlEvalHook(SequenceEvalHook):
    def __init__(self, *args, output_confusions=False, **kwargs):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param output_confusions: display a confusion matrix along with normal outputs
        """
        super().__init__(*args, **kwargs)
        self._output_confusions = output_confusions

        self._markers = None

    def begin(self):
        super().begin()
        self._markers = []

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        for markers, seq_len in zip(run_values.results[MARKER_KEY], run_values.results[LENGTH_KEY]):
            self._markers.append(binary_np_array_to_unicode(markers[:seq_len]))

    def end(self, session):
        result = conll_srl_eval(self._gold, self._predictions, self._markers, self._indices)
        tf.logging.info(str(result))

        # update model's best score for early stopping
        session.run(self._eval_update, feed_dict={self._eval_placeholder: result.evaluation.prec_rec_f1()[2]})

        if self._output_dir:
            predictions_path = os.path.join(self._output_dir, PREDICTIONS_FILE)
            write_props_to_file(predictions_path + '.gold', self._gold, self._markers, self._indices)
            write_props_to_file(predictions_path, self._predictions, self._markers, self._indices)

            step = session.run(tf.train.get_global_step(session.graph))
            append_srl_prediction_output(str(step), result, self._output_dir, output_confusions=self._output_confusions)


class ParserEvalHook(session_run_hook.SessionRunHook):
    def __init__(self, tensors, features, script_path, output_dir=None):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param tensors: dictionary of batch-sized tensors necessary for computing evaluation
        :param features: feature vocabularies
        """
        self._tensors = tensors
        self._features = features
        self._script_path = script_path
        self._output_dir = output_dir
        self._arc_probs = None
        self._arcs = None
        self._rel_probs = None
        self._rels = None

    def begin(self):
        self._arc_probs = []
        self._rel_probs = []
        self._rels = []
        self._arcs = []

    def before_run(self, run_context):
        return SessionRunArgs(fetches=self._tensors)

    def after_run(self, run_context, run_values):
        self._rel_probs.extend(run_values.results[REL_PROBS])
        for arc_probs, rels, heads, seq_len in zip(run_values.results[ARC_PROBS],
                                                   run_values.results[DEPREL_KEY],
                                                   run_values.results[HEAD_KEY],
                                                   run_values.results[LENGTH_KEY]):
            self._arc_probs.append(arc_probs[:seq_len, :seq_len])
            self._rels.append(binary_np_array_to_unicode(rels[:seq_len]))
            self._arcs.append(heads[:seq_len])

    def end(self, session):
        output_file = os.path.join(self._output_dir, PREDICTIONS_FILE)
        gold_file = os.path.join(self._output_dir, GOLD_FILE)
        result = parser_write_and_eval(arc_probs=self._arc_probs,
                                       rel_probs=self._rel_probs,
                                       heads=self._arcs,
                                       rels=self._rels,
                                       script_path=self._script_path,
                                       features=self._features,
                                       out_path=output_file,
                                       gold_path=gold_file)
        tf.logging.info('\n%s', result)


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
