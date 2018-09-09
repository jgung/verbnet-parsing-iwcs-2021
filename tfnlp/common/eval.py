import re
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import tensorflow as tf
from nltk import ConfusionMatrix
from tensorflow.python.lib.io import file_io
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from tfnlp.common.chunk import chunk
from tfnlp.common.constants import ARC_PROBS, DEPREL_KEY, HEAD_KEY, LABEL_KEY, LENGTH_KEY, MARKER_KEY, PREDICT_KEY, REL_PROBS, \
    SENTENCE_INDEX, WORD_KEY
from tfnlp.common.parsing import nonprojective
from tfnlp.common.srleval import evaluate


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


def conll_srl_eval(gold_batches, predicted_batches, words, markers, ids):
    """
    Run the CoNLL-2005 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param words: list of word sequences
    :param markers: list of predicate marker sequences
    :param ids: list of sentence indices
    :return: tuple of (overall F-score, script_output)
    """
    gold_props = _convert_to_sentences(xs=words, ys=gold_batches, indices=markers, ids=ids)
    pred_props = _convert_to_sentences(xs=words, ys=predicted_batches, indices=markers, ids=ids)
    result = evaluate(gold_props, pred_props)
    return result.evaluation.prec_rec_f1()[2], str(result)


def _convert_to_sentences(xs, ys, indices, ids):
    sentences = []
    current_sentence = defaultdict(list)
    prev_sentence = -1

    predicates = []
    args = []
    for words, labels, markers, sentence in zip(xs, ys, indices, ids):
        if prev_sentence != sentence:
            prev_sentence = sentence
            if predicates:
                for index, predicate in enumerate(predicates):
                    current_sentence[0].append(predicate)
                    for i, prop in enumerate(args):
                        current_sentence[i + 1].append(prop[index])
                sentences.append(current_sentence)
                current_sentence = defaultdict(list)
                predicates = []
                args = []
        if not predicates:
            predicates = ["-"] * markers.size
        index = markers.tolist().index(1)
        predicates[index] = words[index]
        args.append(chunk(labels, conll=True))

    if predicates:
        for index, predicate in enumerate(predicates):
            current_sentence[0].append(predicate)
            for i, prop in enumerate(args):
                current_sentence[i + 1].append(prop[index])
        sentences.append(current_sentence)
    return sentences


def _write_to_file(output_file, xs, ys, indices, ids):
    prev_sentence = -1

    predicates = []
    args = []
    for words, labels, markers, sentence in zip(xs, ys, indices, ids):
        if prev_sentence != sentence:
            prev_sentence = sentence
            if predicates:
                line = ''
                for index, predicate in enumerate(predicates):
                    line += '{} {}\n'.format(predicate, " ".join([prop[index] for prop in args]))
                output_file.write(line + '\n')
                predicates = []
                args = []
        if not predicates:
            predicates = ["-"] * markers.size
        index = markers.tolist().index(1)
        predicates[index] = words[index]
        args.append(chunk(labels, conll=True))

    if predicates:
        line = ''
        for index, predicate in enumerate(predicates):
            line += '{} {}\n'.format(predicate, " ".join([prop[index] for prop in args]))
        output_file.write(line + '\n')

    output_file.flush()
    output_file.seek(0)


def accuracy_eval(gold_batches, predicted_batches):
    gold = []
    test = []
    for gold_seq, predicted_seq in zip(gold_batches, predicted_batches):
        gold.extend(gold_seq)
        test.extend(predicted_seq)
    cm = ConfusionMatrix(gold, test)
    tf.logging.info(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

    if len(gold) != len(test):
        raise ValueError("Predictions and gold labels must have the same length.")
    correct = sum(x == y for x, y in zip(gold, test))
    total = len(test)
    accuracy = correct / total
    tf.logging.info("Accuracy: %f (%d/%d)" % (accuracy, correct, total))
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


class SrlEvalHook(session_run_hook.SessionRunHook):
    def __init__(self, tensors, vocab, word_vocab):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param tensors
        :param vocab: label feature vocab
        """
        self._predict_tensor = tensors[PREDICT_KEY]
        self._gold_tensor = tensors[LABEL_KEY]
        self._length_tensor = tensors[LENGTH_KEY]
        self._marker_tensor = tensors[MARKER_KEY]
        self._word_tensor = tensors[WORD_KEY]
        self._index_tensor = tensors[SENTENCE_INDEX]
        self._vocab = vocab
        self._word_vocab = word_vocab

        # initialized in self.begin
        self._predictions = None
        self._gold = None
        self._markers = None
        self._words = None
        self._ids = None
        self._best = -1

    def begin(self):
        self._predictions = []
        self._gold = []
        self._markers = []
        self._words = []
        self._ids = []

    def before_run(self, run_context):
        fetches = {LABEL_KEY: self._gold_tensor,
                   PREDICT_KEY: self._predict_tensor,
                   LENGTH_KEY: self._length_tensor,
                   MARKER_KEY: self._marker_tensor,
                   WORD_KEY: self._word_tensor,
                   SENTENCE_INDEX: self._index_tensor}
        return SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        for gold, predictions, words, markers, seq_len, idx in zip(run_values.results[LABEL_KEY],
                                                                   run_values.results[PREDICT_KEY],
                                                                   run_values.results[WORD_KEY],
                                                                   run_values.results[MARKER_KEY],
                                                                   run_values.results[LENGTH_KEY],
                                                                   run_values.results[SENTENCE_INDEX]):
            self._gold.append([self._vocab.index_to_feat(val) for val in gold][:seq_len])
            self._predictions.append([self._vocab.index_to_feat(val) for val in predictions][:seq_len])
            self._words.append([self._word_vocab.index_to_feat(val) for val in words][:seq_len])
            self._markers.append(markers[:seq_len])
            self._ids.append(idx)

    def end(self, session):
        if self._best >= 0:
            tf.logging.info("Current best score: %f", self._best)
        score, result = conll_srl_eval(self._gold, self._predictions, self._words, self._markers, self._ids)
        tf.logging.info(result)
        if score > self._best:
            self._best = score


class ParserEvalHook(session_run_hook.SessionRunHook):
    def __init__(self, tensors, features, script_path, out_path=None, gold_path=None):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param tensors: dictionary of batch-sized tensors necessary for computing evaluation
        :param features: feature vocabularies
        """
        self._tensors = tensors
        self._features = features
        self._script_path = script_path
        self._output_path = out_path
        self._gold_path = gold_path
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
        fetches = {REL_PROBS: self._tensors[REL_PROBS],
                   ARC_PROBS: self._tensors[ARC_PROBS],
                   LENGTH_KEY: self._tensors[LENGTH_KEY],
                   HEAD_KEY: self._tensors[HEAD_KEY],
                   DEPREL_KEY: self._tensors[DEPREL_KEY]}
        return SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        for rel_probs, arc_probs, rels, heads, seq_len in zip(run_values.results[REL_PROBS],
                                                              run_values.results[ARC_PROBS],
                                                              run_values.results[DEPREL_KEY],
                                                              run_values.results[HEAD_KEY],
                                                              run_values.results[LENGTH_KEY]):
            self._rel_probs.append(rel_probs)  # rel_probs[:seq_len, :, :seq_len]
            self._arc_probs.append(arc_probs[:seq_len, :seq_len])
            self._rels.append(rels[:seq_len])
            self._arcs.append(heads[:seq_len])

    def end(self, session):
        parser_write_and_eval(arc_probs=self._arc_probs,
                              rel_probs=self._rel_probs,
                              heads=self._arcs,
                              rels=self._rels,
                              features=self._features,
                              out_path=self._output_path,
                              gold_path=self._gold_path,
                              script_path=self._script_path)


def parser_write_and_eval(arc_probs, rel_probs, heads, rels, features, script_path, out_path=None, gold_path=None):
    _gold_file = file_io.FileIO(gold_path, 'w') if gold_path else tempfile.NamedTemporaryFile(mode='w', encoding='utf-8')
    _out_file = file_io.FileIO(out_path, 'w') if out_path else tempfile.NamedTemporaryFile(mode='w', encoding='utf-8')
    sys_heads, sys_rels = get_parse_predictions(arc_probs, rel_probs)
    with _out_file as system_file, _gold_file as gold_file:
        write_parse_results_to_file(sys_heads, sys_rels, features, system_file)
        write_parse_results_to_file(heads, rels, features, gold_file)
        result = subprocess.check_output(['perl', script_path, '-g', gold_file.name, '-s', system_file.name, '-q'],
                                         universal_newlines=True)
        tf.logging.info('\n%s', result)


def get_parse_predictions(arc_probs, rel_probs):
    heads = []
    rels = []
    for arc_prob_matrix, rel_prob_tensor in zip(arc_probs, rel_probs):
        arc_preds = nonprojective(arc_prob_matrix)
        arc_preds_one_hot = np.zeros([rel_prob_tensor.shape[0], rel_prob_tensor.shape[2]])
        arc_preds_one_hot[np.arange(len(arc_preds)), arc_preds] = 1.
        rel_preds = np.argmax(np.einsum('nrb,nb->nr', rel_prob_tensor, arc_preds_one_hot), axis=1)
        rels.append(rel_preds)
        heads.append(arc_preds)
    return heads, rels


def write_parse_results_to_file(heads, rels, features, file):
    for sentence_heads, sentence_rels in zip(heads, rels):
        for index, (arc_pred, rel_pred) in enumerate(zip(sentence_heads[1:], sentence_rels[1:])):
            # ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs
            token = ['_'] * 15
            token[0] = str(index + 1)
            token[1] = '_'
            token[8] = str(arc_pred)
            token[9] = str(arc_pred)
            token[10] = features.target(DEPREL_KEY).index_to_feat(rel_pred)
            token[11] = features.target(DEPREL_KEY).index_to_feat(rel_pred)
            file.write('\t'.join(token) + '\n')
        file.write('\n')
    file.flush()
    file.seek(0)


def metric_compare_fn(metric_key):
    def _metric_compare_fn(best_eval_result, current_eval_result):
        if not best_eval_result or metric_key not in best_eval_result:
            raise ValueError(
                'best_eval_result cannot be empty or no loss is found in it.')

        if not current_eval_result or metric_key not in current_eval_result:
            raise ValueError(
                'current_eval_result cannot be empty or no loss is found in it.')

        new = current_eval_result[metric_key]
        prev = best_eval_result[metric_key]
        tf.logging.info("Comparing new score with previous best (%f vs. %f)", new, prev)
        return prev < new

    return _metric_compare_fn


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
