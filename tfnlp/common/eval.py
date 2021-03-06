import os
import re
from collections import defaultdict
from typing import Iterable, Tuple, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io.file_io import get_matching_files

from tfnlp.common.bert import BERT_SUBLABEL, BERT_CLS, BERT_SEP
from tfnlp.common.chunk import chunk
from tfnlp.common.conlleval import conll_eval_lines
from tfnlp.common.parsing import nonprojective
from tfnlp.common.srleval import evaluate
from tfnlp.common.utils import read_json

SUMMARY_FILE = 'eval-summary'
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


def conll_srl_eval(gold_batches, predicted_batches, markers, ids, senses=None, mappings_file=None):
    """
    Run the CoNLL-2005 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param markers: list of predicate marker sequences
    :param ids: list of sentence indices
    :param senses: list of per-instance gold senses for mappings
    :param mappings_file: optional predicate sense label mappings file
    :return: tuple of (overall F-score, script_output, confusion_matrix)
    """
    mappings = {}
    if mappings_file and file_io.file_exists(mappings_file):
        mappings = read_json(mappings_file, as_params=False)
    gold_props = _convert_to_sentences(labels=gold_batches, markers=markers, sentence_ids=ids, senses=senses,
                                       mappings=mappings, gold=True)
    pred_props = _convert_to_sentences(labels=predicted_batches, markers=markers, sentence_ids=ids, senses=senses,
                                       mappings=mappings)
    return evaluate(gold_props, pred_props)


def _convert_to_sentences(labels: List[Iterable[str]],
                          markers: List[Iterable[str]],
                          sentence_ids: List[int],
                          senses: List[Tuple], mappings=None, gold=False) -> List[Dict[int, List[str]]]:
    sentences = []

    for predicates, props_by_predicate in _get_predicates_and_props(labels, markers, sentence_ids, senses):
        current_sentence = defaultdict(list)
        props = [v for k, v in sorted(props_by_predicate.items(), key=lambda x: x[0])]
        for tok, predicate in enumerate(predicates):
            current_sentence[0].append(predicate)
            for i, prop in enumerate(props):
                current_sentence[i + 1].append(prop[tok])
        if mappings:
            label_matcher = re.compile("^\\((?:[CR]-)?([^*(_]+)(?:_[^*(]+)?")
            senses = [x for x in predicates if x is not '-']
            for sense, sent in zip(senses, [current_sentence[i] for i in range(1, len(current_sentence))]):
                if sense == 'x':
                    continue
                roleset_mappings = mappings.get(sense, {})
                for index, label in enumerate(sent):
                    match = label_matcher.match(label)
                    if match:
                        val = match.group(1)
                        mapped = roleset_mappings.get(val)
                        if not mapped:
                            if gold and not (val.startswith("ARGM") or val == "V"):
                                tf.logging.info('Missing mapping for %s / %s' % (sense, label))
                        if not mapped:
                            if val == "PAG":
                                mapped = "0"
                            elif val == "PPT":
                                mapped = "1"
                            else:
                                mapped = val
                        sent[index] = label.replace(val, mapped)

        sentences.append(current_sentence)

    return sentences


def write_props_to_file(output_file,
                        labels: List[Iterable[str]],
                        markers: List[Iterable[str]],
                        sentence_ids: List[int]):
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


def _get_predicates_and_props(labels: List[Iterable[str]],
                              markers: List[Iterable[str]],
                              sentence_ids: List[int],
                              senses: List[Tuple] = None) -> Iterable[Tuple[Iterable[str], Dict[int, List[str]]]]:
    prev_sent_idx = -1  # previous sentence's index
    predicates = []  # list of '-' or 'x', with one per token ('x' indicates the token is a predicate)
    props_by_predicate = {}  # dict from predicate indices to list of predicted or gold argument labels (1 per token)
    if not senses or len(senses) == 0:
        senses = [''] * len(labels)
    sense_dict = {}
    for labels, markers, curr_sent_idx, sense in sorted(zip(labels, markers, sentence_ids, senses), key=lambda x: x[2]):

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
                for key, val in sense_dict.items():
                    predicates[key] = val
                yield predicates, props_by_predicate

            predicates = ["-"] * len(filtered_markers)
            sense_dict = {}
            props_by_predicate = {}

        if sense and sense is not '':
            sense_dict[sense[0]] = sense[1]
        predicate_idx = filtered_markers.index('1')  # index of predicate in tokens
        predicates[predicate_idx] = 'x'  # eval script requires predicate to be a character other than '-'
        props_by_predicate[predicate_idx] = chunk(filtered_labels, conll=True)  # assign SRL labels for this predicate

    if predicates:
        for key, val in sense_dict.items():
            predicates[key] = val
        yield predicates, props_by_predicate


def append_prediction_output(identifier, header, line, detailed, output_path, confusions=None):
    output_dir = os.path.dirname(output_path)
    summary_file = os.path.join(output_dir, '%s.%s.tsv' % (SUMMARY_FILE, identifier))
    eval_log = os.path.join(output_dir, EVAL_LOG)

    exists = tf.gfile.Exists(summary_file) and tf.gfile.Exists(eval_log)

    if not exists:
        with file_io.FileIO(summary_file, 'w') as summary:
            summary.write(header)
            summary.write('\n')
        with file_io.FileIO(eval_log, 'w') as log:
            log.write('%s\n\n' % output_dir)

    with file_io.FileIO(summary_file, 'a') as summary:
        summary.write(line)
        summary.write('\n')

    with file_io.FileIO(eval_log, 'a') as log:
        log.write('\nID: %s\n' % output_path)
        log.write(str(detailed) + '\n')
        if confusions:
            log.write('\n%s\n\n' % str(confusions))


def get_parse_prediction(arc_prob_matrix, rel_prob_tensor, rel_feat=None):
    arc_preds = nonprojective(arc_prob_matrix)
    arc_preds_one_hot = np.zeros([rel_prob_tensor.shape[0], rel_prob_tensor.shape[2]])
    arc_preds_one_hot[np.arange(len(arc_preds)), arc_preds] = 1.
    rel_preds = np.argmax(np.einsum('nrb,nb->nr', rel_prob_tensor, arc_preds_one_hot), axis=1)
    if rel_feat:
        rel_preds = [rel_feat.index_to_feat(rel) for rel in rel_preds]
    return arc_preds, rel_preds


def to_conllx_line(index, word, arc_pred, rel_pred):
    # ID FORM LEMMA CPOS POS FEAT HEAD DEPREL PHEAD PDEPREL
    fields = ['_'] * 10
    fields[0] = str(index + 1)
    fields[1] = word
    fields[2] = word
    fields[6] = str(arc_pred)
    fields[7] = rel_pred
    return fields


def to_conll09_line(index, word, arc_pred, rel_pred):
    # ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs
    fields = ['_'] * 15
    fields[0] = str(index + 1)
    fields[1] = word
    fields[2] = word
    fields[8] = str(arc_pred)
    fields[9] = str(arc_pred)
    fields[10] = rel_pred
    fields[11] = rel_pred
    return fields


def write_parse_result_to_file(sentence_heads, sentence_rels, file, line_func=to_conllx_line, words=None):
    if not words or len(words) == 0:
        words = ['x'] * (len(sentence_rels) - 1)
    for index, (word, arc_pred, rel_pred) in enumerate(zip(words, sentence_heads[1:], sentence_rels[1:])):
        fields = line_func(index, word, arc_pred, rel_pred)
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
