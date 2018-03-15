import os
import re
import subprocess
import tempfile

import tensorflow as tf
from tensorflow.contrib.learn import ExportStrategy, make_export_strategy
from tensorflow.contrib.learn.python.learn.utils.saved_model_export_utils import BestModelSelector
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
        score, result = conll_eval(self._gold, self._predictions, self._script_path)
        tf.logging.info(result)
        if score > self._best:
            self._best = score


def make_best_model_export_strategy(
        serving_input_fn,
        exports_to_keep=1,
        model_dir=None,
        event_file_pattern=None,
        compare_fn=None,
        default_output_alternative_key=None,
        strip_default_attrs=None):
    best_model_export_strategy = make_export_strategy(
        serving_input_fn,
        exports_to_keep=exports_to_keep,
        default_output_alternative_key=default_output_alternative_key,
        strip_default_attrs=strip_default_attrs)

    full_event_file_pattern = os.path.join(
        model_dir,
        event_file_pattern) if model_dir and event_file_pattern else None
    best_model_selector = BestModelSelector(full_event_file_pattern, compare_fn)

    def export_fn(estimator, export_dir_base, checkpoint_path, eval_result=None):
        if not checkpoint_path:
            checkpoint_path = saver.latest_checkpoint(estimator.model_dir)
        export_checkpoint_path, export_eval_result = best_model_selector.update(
            checkpoint_path, eval_result)

        if export_checkpoint_path and export_eval_result is not None:
            checkpoint_base = os.path.basename(export_checkpoint_path)
            export_dir = os.path.join(tf.compat.as_str_any(export_dir_base), tf.compat.as_str_any(checkpoint_base))
            return best_model_export_strategy.export(
                estimator, export_dir, export_checkpoint_path, export_eval_result)
        else:
            return ''

    return ExportStrategy('best_model', export_fn)
