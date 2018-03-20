import os
import re
import subprocess
import tempfile

import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.canned.metric_keys import MetricKeys
from tensorflow.python.estimator.exporter import LatestExporter
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import saver, session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from tfnlp.common.constants import LABEL_KEY, LENGTH_KEY, PREDICT_KEY


def conll_eval(gold_batches, predicted_batches, script_path):
    """
    Run the CoNLL-2003 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param script_path: path to CoNLL-2003 eval script
    :return: tuple of (overall F-score, script_output)
    """
    with tempfile.NamedTemporaryFile(mode='wt') as temp:
        for gold_seq, predicted_seq in zip(gold_batches, predicted_batches):
            for label, prediction in zip(gold_seq, predicted_seq):
                temp.write("_ {} {}\n".format(label, prediction))
            temp.write("\n")  # sentence break
        temp.flush()
        temp.seek(0)
        result = subprocess.check_output(["perl", script_path], stdin=temp, universal_newlines=True)
        return float(re.split('\s+', re.split('\n', result)[1].strip())[7]), result


def accuracy_eval(gold_batches, predicted_batches):
    gold = []
    test = []
    for gold_seq, predicted_seq in zip(gold_batches, predicted_batches):
        gold.extend(gold_seq)
        test.extend(predicted_seq)
    cm = nltk.ConfusionMatrix(gold, test)
    print(cm.pretty_format(sort_by_count=True, show_percents=True))
    accuracy = nltk.metrics.scores.accuracy(gold, test)
    return accuracy


class SequenceEvalHook(session_run_hook.SessionRunHook):
    def __init__(self, script_path, predict_tensor, gold_tensor, length_tensor, vocab):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param script_path: path to eval script
        :param predict_tensor: iterable over a batch of predictions
        :param gold_tensor: iterable over the corresponding batch of labels
        :param length_tensor: batch-sized tensor of sequence lengths
        :param vocab: label feature vocab
        """
        self._script_path = script_path
        self._predict_tensor = predict_tensor
        self._gold_tensor = gold_tensor
        self._length_tensor = length_tensor
        self._vocab = vocab

        self._predictions = None
        self._gold = None
        self._best = -1

    def begin(self):
        self._predictions = []
        self._gold = []

    def before_run(self, run_context):
        fetches = {LABEL_KEY: self._gold_tensor,
                   PREDICT_KEY: self._predict_tensor,
                   LENGTH_KEY: self._length_tensor}
        return SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        for gold, predictions, seq_len in zip(run_values.results[LABEL_KEY],
                                              run_values.results[PREDICT_KEY],
                                              run_values.results[LENGTH_KEY]):
            self._gold.append([self._vocab.index_to_feat(val) for val in gold][:seq_len])
            self._predictions.append([self._vocab.index_to_feat(val) for val in predictions][:seq_len])

    def end(self, session):
        if self._best >= 0:
            tf.logging.info("Current best score: %f", self._best)
        if self._script_path is not None:
            score, result = conll_eval(self._gold, self._predictions, self._script_path)
            tf.logging.info(result)
        else:
            score = accuracy_eval(self._gold, self._predictions)
        if score > self._best:
            self._best = score


class BestExporter(LatestExporter):

    def __init__(self, serving_input_receiver_fn, assets_extra=None, as_text=False, exports_to_keep=5,
                 compare_fn=None, event_file_pattern=None, compare_key=MetricKeys.LOSS):
        super().__init__('best_model', serving_input_receiver_fn, assets_extra, as_text, exports_to_keep)
        self._compare_fn = compare_fn or self._default_compare_fn
        self._best_eval_result = self._get_best_eval_result(event_file_pattern)
        self._default_compare_key = compare_key

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        if not checkpoint_path:
            checkpoint_path = saver.latest_checkpoint(estimator.model_dir)
        export_checkpoint_path, export_eval_result = self._update(checkpoint_path, eval_result)

        if export_checkpoint_path and export_eval_result is not None:
            checkpoint_base = os.path.basename(export_checkpoint_path)
            export_dir = os.path.join(tf.compat.as_str_any(export_path), tf.compat.as_str_any(checkpoint_base))
            return super().export(estimator, export_dir, export_checkpoint_path, export_eval_result, is_the_final_export)
        else:
            return ''

    def _update(self, checkpoint_path, eval_result):
        """
        Records a given checkpoint and exports if this is the best model.
        :param checkpoint_path: the checkpoint path to export
        :param eval_result: a dictionary which is usually generated in evaluation runs
        :return: path to export checkpoint and dictionary and eval_result, or None
        """
        if not checkpoint_path:
            raise ValueError('Checkpoint path is empty.')
        if eval_result is None:
            raise ValueError('%s has empty evaluation results.', checkpoint_path)

        if (self._best_eval_result is None or
                self._compare_fn(self._best_eval_result, eval_result)):
            tf.logging.info("Updating best result from %s to %s", self._best_eval_result, eval_result)
            self._best_eval_result = eval_result
            return checkpoint_path, eval_result
        else:
            return '', None

    def _get_best_eval_result(self, event_files):
        if not event_files:
            return None

        best_eval_result = None
        for event_file in tf.gfile.Glob(os.path.join(event_files)):
            for event in summary_iterator.summary_iterator(event_file):
                if event.HasField('summary'):
                    event_eval_result = {}
                    for value in event.summary.value:
                        if value.HasField('simple_value'):
                            event_eval_result[value.tag] = value.simple_value
                    if best_eval_result is None or self._compare_fn(best_eval_result, event_eval_result):
                        best_eval_result = event_eval_result
        return best_eval_result

    def _default_compare_fn(self, curr_best_eval_result, cand_eval_result):
        if not curr_best_eval_result or self._default_compare_key not in curr_best_eval_result:
            raise ValueError('curr_best_eval_result cannot be empty or no loss is found in it.')
        if not cand_eval_result or self._default_compare_key not in cand_eval_result:
            raise ValueError('cand_eval_result cannot be empty or no loss is found in it.')

        return cand_eval_result[self._default_compare_key] > curr_best_eval_result[self._default_compare_key]


def log_trainable_variables():
    """
    Log every trainable variable name and shape and return the total number of trainable variables.
    :return: total number of trainable variables
    """
    all_weights = {variable.name: variable for variable in tf.trainable_variables()}
    total_size = 0
    for variable_name in sorted(list(all_weights)):
        variable = all_weights[variable_name]
        tf.logging.info("%s\tshape    %s", variable.name[:-2].ljust(80),
                        str(variable.shape).ljust(20))
        variable_size = int(np.prod(np.array(variable.shape.as_list())))
        total_size += variable_size

    tf.logging.info("Total trainable variables size: %d", total_size)
    return total_size
