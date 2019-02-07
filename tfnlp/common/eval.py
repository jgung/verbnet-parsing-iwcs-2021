import os
import re
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import tensorflow as tf
from nltk import ConfusionMatrix
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io.file_io import get_matching_files

from tfnlp.common.chunk import chunk
from tfnlp.common.conlleval import conll_eval_lines
from tfnlp.common.parsing import nonprojective
from tfnlp.common.srleval import evaluate

SUMMARY_FILE = 'eval-summary.tsv'
EVAL_LOG = 'eval.log'
PREDICTIONS_FILE = 'predictions.txt'


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
            for label, prediction in zip(gold_seq, predicted_seq):
                yield "_ {} {}".format(label, prediction)
            yield ""  # sentence break

    if output_file:
        with file_io.FileIO(output_file, 'w') as output:
            for line in get_lines():
                output.write(line + '\n')

    result = conll_eval_lines(get_lines(), raw=True).to_conll_output()
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

    def _add_sentence(_props_by_predicate, _predicates):
        current_sentence = defaultdict(list)
        for tok, predicate in enumerate(_predicates):
            current_sentence[0].append(predicate)
            for i, prop in enumerate(_props_by_predicate):
                current_sentence[i + 1].append(prop[tok])
        sentences.append(current_sentence)

    prev_sent_idx = -1
    predicates = []
    props_by_predicate = []
    for labels, markers, curr_sent_idx in zip(ys, indices, ids):
        if prev_sent_idx != curr_sent_idx:
            prev_sent_idx = curr_sent_idx
            if predicates:
                _add_sentence(props_by_predicate, predicates)
            predicates = ["-"] * len(markers)
            props_by_predicate = []

        predicate_idx = markers.index('1')
        predicates[predicate_idx] = 'x'
        props_by_predicate.append(chunk(labels, conll=True))

    if predicates:
        _add_sentence(props_by_predicate, predicates)
    return sentences


def write_props_to_file(output_file, labels, markers, sentence_ids):
    """
    Write PropBank predictions to a file.
    :param output_file: output file
    :param labels: lists of labels
    :param markers: predicate markers
    :param sentence_ids: sentence indices
    """
    with file_io.FileIO(output_file, 'w') as output_file:

        def _write_props(_props_by_predicate, _predicates):
            # used to ensure proposition columns are in correct order (by appearance of predicate in sentence)
            prop_list = [arg for _, arg in sorted(_props_by_predicate.items(), key=lambda item: item[0])]
            line = ''
            for tok, predicate in enumerate(_predicates):
                line += '%s %s\n' % (predicate, ' '.join([prop[tok] for prop in prop_list]))
            output_file.write(line + '\n')

        prev_sent_idx = -1  # previous sentence's index
        predicates = []  # list of '-' or 'x', with one per token ('x' indicates the token is a predicate)
        props_by_predicate = {}  # dict from predicate indices to list of predicted or gold argument labels (1 per token)
        for labels, markers, curr_sent_idx in sorted(zip(labels, markers, sentence_ids), key=lambda x: x[2]):

            if prev_sent_idx != curr_sent_idx:  # either first sentence, or a new sentence
                prev_sent_idx = curr_sent_idx

                if predicates:
                    _write_props(props_by_predicate, predicates)

                predicates = ["-"] * len(markers)
                props_by_predicate = {}

            predicate_idx = markers.index('1')  # index of predicate in tokens
            predicates[predicate_idx] = 'x'  # official eval script requires predicate to be a character other than '-'
            props_by_predicate[predicate_idx] = (chunk(labels, conll=True))  # assign SRL labels for this predicate

        if predicates:
            _write_props(props_by_predicate, predicates)


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


def parser_write_and_eval(arc_probs, rel_probs, heads, rels, script_path, features=None, out_path=None, gold_path=None):
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
    ckpts = get_matching_files(os.path.join(model_dir, '*.index'))
    path_step_ckpts = []
    for ckpt in ckpts:
        match = CKPT_PATTERN.search(ckpt)
        if match:
            path_step_ckpts.append((match.group(1), int(match.group(2))))
    # noinspection PyTypeChecker
    return min(path_step_ckpts, key=lambda x: x[1], default=(None, None))[0]
