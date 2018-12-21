import glob
import os
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
    SENTENCE_INDEX
from tfnlp.common.parsing import nonprojective
from tfnlp.common.srleval import evaluate


def conll_eval(gold_batches, predicted_batches, indices, script_path, output_file=None):
    """
    Run the CoNLL-2003 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param indices: order of sequences
    :param script_path: path to CoNLL-2003 eval script
    :param output_file: optional output file name for predictions
    :return: tuple of (overall F-score, script_output)
    """
    _output_file = file_io.FileIO(output_file, 'w') if output_file else tempfile.NamedTemporaryFile(mode='wt')
    with _output_file as temp:
        # sort by sentence index to maintain original order of instances
        for gold_seq, predicted_seq, index in sorted(zip(gold_batches, predicted_batches, indices), key=lambda k: k[2]):
            for label, prediction in zip(gold_seq, predicted_seq):
                temp.write("_ {} {}\n".format(label, prediction))
            temp.write("\n")  # sentence break
        temp.flush()
        temp.seek(0)
        result = subprocess.check_output(["perl", script_path], stdin=temp, universal_newlines=True)
        return float(re.split('\s+', re.split('\n', result)[1].strip())[7]), result


def conll_srl_eval(gold_batches, predicted_batches, markers, ids):
    """
    Run the CoNLL-2005 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param markers: list of predicate marker sequences
    :param ids: list of sentence indices
    :return: tuple of (overall F-score, script_output, confusion_matrix)
    """
    gold_props = _convert_to_sentences(ys=gold_batches, indices=markers, ids=ids)
    pred_props = _convert_to_sentences(ys=predicted_batches, indices=markers, ids=ids)
    return evaluate(gold_props, pred_props)


def _convert_to_sentences(ys, indices, ids):
    sentences = []
    current_sentence = defaultdict(list)
    prev_sentence = -1

    predicates = []
    args = []
    for labels, markers, sentence in zip(ys, indices, ids):
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
        index = markers.tolist().index(b'1')
        predicates[index] = 'x'
        args.append(chunk(labels, conll=True))

    if predicates:
        for index, predicate in enumerate(predicates):
            current_sentence[0].append(predicate)
            for i, prop in enumerate(args):
                current_sentence[i + 1].append(prop[index])
        sentences.append(current_sentence)
    return sentences


def _write_to_file(output_file, ys, indices, ids):
    prev_sentence = -1

    predicates = []
    args = []
    for labels, markers, sentence in zip(ys, indices, ids):
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
        predicates[index] = 'x'
        args.append(chunk(labels, conll=True))

    if predicates:
        line = ''
        for index, predicate in enumerate(predicates):
            line += '{} {}\n'.format(predicate, " ".join([prop[index] for prop in args]))
        output_file.write(line + '\n')

    output_file.flush()
    output_file.seek(0)


def accuracy_eval(gold_batches, predicted_batches, indices, output_file=None):
    gold = []
    test = []

    if output_file:
        with file_io.FileIO(output_file, 'w') as _out_file:
            # sort by sentence index to maintain original order of instances
            for predicted_seq, index in sorted(zip(predicted_batches, indices), key=lambda k: k[1]):
                for prediction in predicted_seq:
                    _out_file.write("{}\n".format(prediction))
                _out_file.write("\n")  # sentence break

    for gold_seq, predicted_seq in zip(gold_batches, predicted_batches):
        gold.extend(gold_seq)
        test.extend(predicted_seq)
    cm = ConfusionMatrix(gold, test)
    tf.logging.info('\n%s' % cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

    if len(gold) != len(test):
        raise ValueError("Predictions and gold labels must have the same length.")
    correct = sum(x == y for x, y in zip(gold, test))
    total = len(test)
    accuracy = correct / total
    tf.logging.info("Accuracy: %f (%d/%d)" % (accuracy, correct, total))
    return accuracy


class EvalHook(session_run_hook.SessionRunHook):
    def __init__(self, tensors, vocab, label_key=LABEL_KEY, predict_key=PREDICT_KEY, output_file=None):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param tensors: dictionary of batch-sized tensors necessary for computing evaluation
        :param vocab: label feature vocab
        :param label_key: targets key in `tensors`
        :param predict_key: predictions key in `tensors`
        :param output_file: optional output file name for predictions
        """
        self._fetches = tensors
        self._vocab = vocab
        self._label_key = label_key
        self._predict_key = predict_key
        self._output_file = output_file

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

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        self._gold.append((self._vocab.index_to_feat(gold) for gold in run_values.results[self._label_key]))
        self._predictions.append((self._vocab.index_to_feat(prediction) for prediction in run_values.results[self._predict_key]))

    def end(self, session):
        accuracy_eval(self._gold, self._predictions, self._indices, output_file=self._output_file)


class SequenceEvalHook(EvalHook):

    def __init__(self, *args, script_path=None, eval_update=None, eval_placeholder=None, **kwargs):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param script_path: path to eval script
        :param eval_update: operation to update current best score
        :param eval_placeholder: placeholder for update operation
        """
        super().__init__(*args, **kwargs)
        self._script_path = script_path
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
        if self._script_path is not None:
            score, result = conll_eval(self._gold, self._predictions, self._indices, self._script_path, self._output_file)
            tf.logging.info(result)
        else:
            score = accuracy_eval(self._gold, self._predictions, self._indices, output_file=self._output_file)
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
            self._markers.append(markers[:seq_len])

    def end(self, session):
        result = conll_srl_eval(self._gold, self._predictions, self._markers, self._indices)
        tf.logging.info(str(result))
        if self._output_confusions:
            tf.logging.info('\n%s' % result.confusion_matrix())
        session.run(self._eval_update, feed_dict={self._eval_placeholder: result.evaluation.prec_rec_f1()[2]})


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
        return SessionRunArgs(fetches=self._tensors)

    def after_run(self, run_context, run_values):
        self._rel_probs.extend(run_values.results[REL_PROBS])
        for arc_probs, rels, heads, seq_len in zip(run_values.results[ARC_PROBS],
                                                   run_values.results[DEPREL_KEY],
                                                   run_values.results[HEAD_KEY],
                                                   run_values.results[LENGTH_KEY]):
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

    write_func = write_parse_results_to_conllx_file if 'conllx' in script_path else write_parse_results_to_file

    with _out_file as system_file, _gold_file as gold_file:
        write_func(sys_heads, sys_rels, system_file, features)
        write_func(heads, rels, gold_file)
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


def write_parse_results_to_file(heads, rels, file, features=None):
    for sentence_heads, sentence_rels in zip(heads, rels):
        for index, (arc_pred, rel_pred) in enumerate(zip(sentence_heads[1:], sentence_rels[1:])):
            # ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs
            token = ['_'] * 15
            token[0] = str(index + 1)
            token[1] = '_'
            token[8] = str(arc_pred)
            token[9] = str(arc_pred)
            if features:
                rel = features.index_to_feat(rel_pred)
            else:
                rel = rel_pred.decode('utf-8')
            token[10] = rel
            token[11] = rel
            file.write('\t'.join(token) + '\n')
        file.write('\n')
    file.flush()
    file.seek(0)


def write_parse_results_to_conllx_file(heads, rels, file, features=None):
    for sentence_heads, sentence_rels in zip(heads, rels):
        for index, (arc_pred, rel_pred) in enumerate(zip(sentence_heads[1:], sentence_rels[1:])):
            # ID FORM LEMMA CPOS POS FEAT HEAD DEPREL PHEAD PDEPREL
            token = ['_'] * 10
            token[0] = str(index + 1)
            token[1] = 'x'
            token[6] = str(arc_pred)
            if features:
                rel = features.index_to_feat(rel_pred)
            else:
                rel = rel_pred.decode('utf-8')
            token[7] = rel
            file.write('\t'.join(token) + '\n')
        file.write('\n')
    file.flush()
    file.seek(0)


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


def log_trainable_variables():
    """
    Log every trainable variable name and shape and return the total number of trainable variables.
    :return: total number of trainable variables
    """
    all_weights = {variable.name: variable for variable in tf.trainable_variables()}
    total_size = 0
    weights = []
    for variable_name in sorted(list(all_weights)):
        variable = all_weights[variable_name]
        weights.append("%s\tshape    %s" % (variable.name[:-2].ljust(80), str(variable.shape).ljust(20)))
        variable_size = int(np.prod(np.array(variable.shape.as_list())))
        total_size += variable_size

    weights.append("Total trainable variables size: %d" % total_size)
    tf.logging.log_first_n(tf.logging.INFO, "Trainable variables:\n%s\n", 1, '\n'.join(weights))
    return total_size


CKPT_PATTERN = re.compile('(\S+\.ckpt-(\d+))\.index')


def get_earliest_checkpoint(model_dir):
    """
    Returns the path to the earliest checkpoint in a particular model directory.
    :param model_dir: base model directory containing checkpoints
    :return: path to earliest checkpoint
    """
    ckpts = glob.glob(os.path.join(model_dir, '*.index'))
    path_step_ckpts = []
    for ckpt in ckpts:
        match = CKPT_PATTERN.search(ckpt)
        if match:
            path_step_ckpts.append((match.group(1), int(match.group(2))))
    # noinspection PyTypeChecker
    return min(path_step_ckpts, key=lambda x: x[1], default=(None, None))[0]
