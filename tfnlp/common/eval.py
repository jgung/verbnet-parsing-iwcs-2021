import os
import re
from collections import defaultdict
from typing import Iterable, Tuple, Dict, List

import numpy as np
import tensorflow as tf
from nltk import ConfusionMatrix
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io.file_io import get_matching_files

from tfnlp.common.bert import BERT_SUBLABEL, BERT_CLS, BERT_SEP
from tfnlp.common.chunk import chunk
from tfnlp.common.conlleval import conll_eval_lines
from tfnlp.common.parsing import nonprojective
from tfnlp.common.srleval import evaluate

from sklearn.metrics import classification_report

SUMMARY_FILE = 'eval-summary.tsv'
EVAL_LOG = 'eval.log'
PREDICTIONS_FILE = 'predictions.txt'
GOLD_FILE = 'gold.txt'


def conll_eval(gold_batches, predicted_batches, indices, output_file=None):
    """
    Run the CoNLL-2003 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param indices: order of sequences
    :param output_file: optional output file name for predictions
    :return: tuple of (overall F-score, script_output)
    """

    def get_lines():
        for gold_seq, predicted_seq, index in sorted(zip(gold_batches, predicted_batches, indices), key=lambda k: k[2]):
            for i, (label, prediction) in enumerate(zip(gold_seq, predicted_seq)):
                if label in [BERT_SUBLABEL, BERT_CLS, BERT_SEP]:
                    continue
                if prediction == BERT_SUBLABEL:
                    prediction = 'O'
                res = "_ {} {}".format(label, prediction)
                yield res
            yield ""  # sentence break

    if output_file:
        with file_io.FileIO(output_file, 'w') as output:
            for line in get_lines():
                output.write(line + '\n')

    result = conll_eval_lines(get_lines(), raw=True).to_conll_output()
    return float(re.split('\\s+', re.split('\n', result)[1].strip())[7]), result


def conll_srl_eval(gold_batches, predicted_batches, markers, ids):
    """
    Run the CoNLL-2005 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param markers: list of predicate marker sequences
    :param ids: list of sentence indices
    :return: tuple of (overall F-score, script_output, confusion_matrix)
    """
    gold_props = _convert_to_sentences(labels=gold_batches, markers=markers, sentence_ids=ids)
    pred_props = _convert_to_sentences(labels=predicted_batches, markers=markers, sentence_ids=ids)
    return evaluate(gold_props, pred_props)


def _convert_to_sentences(labels: Iterable[Iterable[str]],
                          markers: Iterable[Iterable[str]],
                          sentence_ids: Iterable[int]) -> List[Dict[int, List[str]]]:
    sentences = []

    for predicates, props_by_predicate in _get_predicates_and_props(labels, markers, sentence_ids):
        current_sentence = defaultdict(list)
        props = list(props_by_predicate.values())
        for tok, predicate in enumerate(predicates):
            current_sentence[0].append(predicate)
            for i, prop in enumerate(props):
                current_sentence[i + 1].append(prop[tok])
        sentences.append(current_sentence)

    return sentences


def write_props_to_file(output_file,
                        labels: Iterable[Iterable[str]],
                        markers: Iterable[Iterable[str]],
                        sentence_ids: Iterable[int]):
    """
    Write PropBank predictions to a file.
    :param output_file: output file
    :param labels: lists of labels
    :param markers: predicate markers
    :param sentence_ids: sentence indices
    """
    with file_io.FileIO(output_file, 'w') as output_file:
        for predicates, props_by_predicate in _get_predicates_and_props(labels, markers, sentence_ids):
            # sorting to ensure proposition columns are in correct order (by appearance of predicate in sentence)
            prop_list = [arg for _, arg in sorted(props_by_predicate.items(), key=lambda item: item[0])]
            line = ''
            for tok, predicate in enumerate(predicates):
                line += '%s %s\n' % (predicate, ' '.join([prop[tok] for prop in prop_list]))
            output_file.write(line + '\n')


def _get_predicates_and_props(labels: Iterable[Iterable[str]],
                              markers: Iterable[Iterable[str]],
                              sentence_ids: Iterable[int]) -> Iterable[Tuple[Iterable[str], Dict[int, List[str]]]]:
    prev_sent_idx = -1  # previous sentence's index
    predicates = []  # list of '-' or 'x', with one per token ('x' indicates the token is a predicate)
    props_by_predicate = {}  # dict from predicate indices to list of predicted or gold argument labels (1 per token)
    for labels, markers, curr_sent_idx in sorted(zip(labels, markers, sentence_ids), key=lambda x: x[2]):

        filtered_labels = []
        filtered_markers = []
        for label, marker in zip(labels, markers):
            if label == BERT_SUBLABEL:
                continue
            filtered_labels.append(label)
            filtered_markers.append(marker)

        if prev_sent_idx != curr_sent_idx:  # either first sentence, or a new sentence
            prev_sent_idx = curr_sent_idx

            if predicates:
                yield predicates, props_by_predicate

            predicates = ["-"] * len(filtered_markers)
            props_by_predicate = {}

        predicate_idx = filtered_markers.index('1')  # index of predicate in tokens
        predicates[predicate_idx] = 'x'  # official eval script requires predicate to be a character other than '-'
        props_by_predicate[predicate_idx] = chunk(filtered_labels, conll=True)  # assign SRL labels for this predicate

    if predicates:
        yield predicates, props_by_predicate


def append_srl_prediction_output(identifier, result, output_dir, output_confusions=False):
    summary_file = os.path.join(output_dir, SUMMARY_FILE)
    eval_log = os.path.join(output_dir, EVAL_LOG)

    exists = tf.gfile.Exists(summary_file) and tf.gfile.Exists(eval_log)

    if not exists:
        with file_io.FileIO(summary_file, 'w') as summary:
            summary.write('ID\t# Props\t% Perfect\tPrecision\tRecall\tF1\n')
        with file_io.FileIO(eval_log, 'w') as log:
            log.write('%s\n\n' % output_dir)

    with file_io.FileIO(summary_file, 'a') as summary:
        p, r, f1 = result.evaluation.prec_rec_f1()
        summary.write('%s\t%d\t%f\t%f\t%f\t%f\n' % (identifier,
                                                    result.ntargets,
                                                    result.perfect_props(),
                                                    p, r, f1))

    with file_io.FileIO(eval_log, 'a') as eval_log:
        eval_log.write('\nID: %s\n' % identifier)
        eval_log.write(str(result) + '\n')
        if output_confusions:
            eval_log.write('\n%s\n\n' % result.confusion_matrix())


def accuracy_eval(gold_labels, predicted_labels, indices, output_file=None):
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Predictions and gold labels must have the same length.")

    if output_file:
        with file_io.FileIO(output_file, 'w') as _out_file:
            # sort by sentence index to maintain original order of instances
            for predicted, index, gold in sorted(zip(predicted_labels, indices, gold_labels), key=lambda k: k[1]):
                _out_file.write("{}\t{}\t{}\t{}\n".format(index, gold, predicted, '-' if gold != predicted else ''))

    cm = ConfusionMatrix(gold_labels, predicted_labels)

    tf.logging.info('\n%s' % cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    report = classification_report(y_true=gold_labels, y_pred=predicted_labels, digits=4)
    tf.logging.info('\n%s' % report)

    correct = sum(x == y for x, y in zip(gold_labels, predicted_labels))
    total = len(predicted_labels)
    accuracy = correct / total
    tf.logging.info("Accuracy: %f (%d/%d)" % (accuracy, correct, total))
    return accuracy


def get_parse_prediction(arc_prob_matrix, rel_prob_tensor, rel_feat=None):
    arc_preds = nonprojective(arc_prob_matrix)
    arc_preds_one_hot = np.zeros([rel_prob_tensor.shape[0], rel_prob_tensor.shape[2]])
    arc_preds_one_hot[np.arange(len(arc_preds)), arc_preds] = 1.
    rel_preds = np.argmax(np.einsum('nrb,nb->nr', rel_prob_tensor, arc_preds_one_hot), axis=1)
    if rel_feat:
        rel_preds = [rel_feat.index_to_feat(rel) for rel in rel_preds]
    return arc_preds, rel_preds


def to_conllx_line(index, arc_pred, rel_pred):
    # ID FORM LEMMA CPOS POS FEAT HEAD DEPREL PHEAD PDEPREL
    fields = ['_'] * 10
    fields[0] = str(index + 1)
    fields[1] = 'x'
    fields[6] = str(arc_pred)
    fields[7] = rel_pred
    return fields


def to_conll09_line(index, arc_pred, rel_pred):
    # ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs
    fields = ['_'] * 15
    fields[0] = str(index + 1)
    fields[1] = '_'
    fields[8] = str(arc_pred)
    fields[9] = str(arc_pred)
    fields[10] = rel_pred
    fields[11] = rel_pred
    return fields


def write_parse_result_to_file(sentence_heads, sentence_rels, file, line_func=to_conllx_line):
    for index, (arc_pred, rel_pred) in enumerate(zip(sentence_heads[1:], sentence_rels[1:])):
        fields = line_func(index, arc_pred, rel_pred)
        file.write('\t'.join(fields) + '\n')
    file.write('\n')


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


CKPT_PATTERN = re.compile('(\\S+\\.ckpt-(\\d+))\\.index')


def get_earliest_checkpoint(model_dir):
    """
    Returns the path to the earliest checkpoint in a particular model directory.
    :param model_dir: base model directory containing checkpoints
    :return: path to earliest checkpoint
    """
    ckpts = get_matching_files(os.path.join(model_dir, '*.index'))
    path_step_ckpts = []
    for ckpt in ckpts:
        match = CKPT_PATTERN.search(ckpt)
        if match:
            path_step_ckpts.append((match.group(1), int(match.group(2))))
    # noinspection PyTypeChecker
    return min(path_step_ckpts, key=lambda x: x[1], default=(None, None))[0]
