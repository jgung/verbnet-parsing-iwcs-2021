import subprocess
import unicodedata
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.python.lib.io import file_io

from tfnlp.common import constants
from tfnlp.common.bert import BERT_SUBLABEL
from tfnlp.common.config import append_label
from tfnlp.common.eval import conll_eval, conll_srl_eval, append_prediction_output
from tfnlp.common.eval import write_props_to_file, get_parse_prediction, \
    to_conll09_line, to_conllx_line, write_parse_result_to_file

PUNCT_CAT = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}


def get_evaluator(heads, feature_extractor, output_path, script_path):
    evaluators = {
        constants.TAGGER_KEY: TaggerEvaluator,
        constants.SRL_KEY: SrlEvaluator,
        constants.NER_KEY: TaggerEvaluator,
        constants.PARSER_KEY: DepParserEvaluator,
        constants.TOKEN_CLASSIFIER_KEY: TokenClassifierEvaluator,
        constants.CLASSIFIER_KEY: TokenClassifierEvaluator
    }

    evals = []
    for head in heads:
        if head.task not in evaluators:
            raise ValueError("Unsupported head type: " + head.task)
        try:
            evaluator = evaluators[head.task](target=feature_extractor.targets[head.name],
                                              output_path=output_path + '.' + head.name,
                                              script_path=script_path)
            evals.append(evaluator)
        except AssertionError:
            tf.logging.info("Skipping evaluation of '%s' since an evaluation script was not provided", head.task)

    return AggregateEvaluator(evals)


class Evaluator(object):

    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__()
        self.target = target
        self.output_path = output_path
        self.script_path = script_path

        self.metric = 0
        self.summary = ''

    def __call__(self, labeled_instances, results):
        """
        Perform standard evaluation on a given list of gold labeled instances.
        :param labeled_instances: labeled instances
        :param results: prediction results corresponding to labeled instances
        """
        self.start()
        count = 0
        for instance, result in zip(labeled_instances, results):
            self.accumulate(instance, result)
            count += 1
            if count % 1024 == 0:
                tf.logging.info("...Accumulated %d instances for evaluation.", count)
        tf.logging.info("Evaluating on %d instances...", count)
        self.evaluate(self.output_path)

    def start(self):
        pass

    def accumulate(self, instance, result):
        pass

    def evaluate(self, identifier=None):
        pass


class AggregateEvaluator(Evaluator):
    def __init__(self, evaluators: List[Evaluator]) -> None:
        super().__init__()
        self._evaluators = evaluators

    def start(self):
        for evaluator in self._evaluators:
            evaluator.start()

    def accumulate(self, instance, result):
        for evaluator in self._evaluators:
            evaluator.accumulate(instance, result)

    def evaluate(self, identifier=None):
        total, summaries = 0, []
        for evaluator in self._evaluators:
            evaluator.evaluate(identifier if identifier is not None else evaluator.output_path)
            total += evaluator.metric
            summaries.append(str(evaluator.summary))
        self.metric, self.summary = total / len(self._evaluators), '\n'.join(summaries)


class TokenClassifierEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)
        self.labels = None
        self.gold = None
        self.indices = None
        self.token_indices = None
        self.target_key = constants.LABEL_KEY if not self.target.name else self.target.name
        self.labels_key = constants.LABEL_KEY if not self.target.key else self.target.key
        self.scores_name = append_label(constants.LABEL_SCORES, self.target_key)

    def start(self):
        self.labels = []
        self.gold = []
        self.indices = []
        self.token_indices = []

    def accumulate(self, instance, result):
        if self.target.constraints:
            scores = {self.target.index_to_feat(i): score for i, score in enumerate(result[self.scores_name])}
            ck = instance[self.target.constraint_key]
            valid_scores = {label: score for label, score in scores.items() if label in self.target.constraints.get(ck, [])}
            label = max(valid_scores.items(), key=lambda x: x[1], default=(self.target.unknown_word, 0))[0]
            self.labels.append(label)
        else:
            result = result[self.target_key]
            if not isinstance(result, str):
                result = result.decode('utf-8')
            self.labels.append(result)

        # apply mappings to gold labels?
        label = instance[self.labels_key]
        for func in self.target.mapping_funcs:
            label = func(label)

        self.gold.append(label)
        self.indices.append(instance[constants.SENTENCE_INDEX])
        if constants.INSTANCE_INDEX in instance:
            self.token_indices.append(instance[constants.INSTANCE_INDEX])

    def evaluate(self, identifier='.'):
        output_file = self.output_path + '.txt'

        if len(self.gold) != len(self.labels):
            raise ValueError("Predictions and gold labels must have the same length.")
        if output_file:
            with file_io.FileIO(output_file, 'w') as _out_file:
                if len(self.token_indices) > 0:
                    for predicted, index, token_idx, gold in sorted(
                            zip(self.labels, self.indices, self.token_indices, self.gold), key=lambda k: (k[1], k[2])):
                        _out_file.write("{}\t{}\t{}\t{}\n".format(token_idx, gold, predicted, '-' if gold != predicted else ''))
                else:
                    # sort by sentence index to maintain original order of instances
                    for predicted, index, gold in sorted(zip(self.labels, self.indices, self.gold), key=lambda k: k[1]):
                        _out_file.write("{}\t{}\t{}\t{}\n".format(index, gold, predicted, '-' if gold != predicted else ''))

        report = classification_report(y_true=self.gold, y_pred=self.labels, digits=4, zero_division=0)
        tf.logging.info('\n%s' % report)

        correct = sum(x == y for x, y in zip(self.gold, self.labels))
        total = len(self.labels)
        accuracy = correct / total
        tf.logging.info("Accuracy: %f (%d/%d)" % (accuracy, correct, total))

        append_prediction_output(identifier=self.target.name,
                                 header='ID\tCorrect\tTotal\tAccuracy',
                                 line='%s\t%d\t%d\t%f' % (identifier, correct, total, accuracy),
                                 detailed=report,
                                 output_path=self.output_path)
        self.metric, self.summary = accuracy, report


class TaggerEvaluator(Evaluator):

    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)
        self.labels = None
        self.gold = None
        self.indices = None
        self.target_key = constants.LABEL_KEY if not self.target.name else self.target.name
        self.labels_key = constants.LABEL_KEY if not self.target.key else self.target.key

    def start(self):
        self.labels = []
        self.gold = []
        self.indices = []

    def accumulate(self, instance, result):
        self.labels.append([label for label in result[self.target_key] if label != BERT_SUBLABEL])
        self.gold.append([label for label in instance[self.labels_key] if label != BERT_SUBLABEL])
        self.indices.append(instance[constants.SENTENCE_INDEX])

    def evaluate(self, identifier=None):
        f1, result_str = conll_eval(self.gold, self.labels, self.indices, output_file=self.output_path + '.txt')
        tf.logging.info(result_str)

        append_prediction_output(identifier=self.target.name,
                                 header='ID\tF1',
                                 line='%s\t%f' % (identifier, f1),
                                 detailed=result_str,
                                 output_path=self.output_path)
        self.metric, self.summary = f1, result_str


class DepParserEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)
        self._sent_index = None
        self._sent_arc_probs, self._sent_rel_probs = None, None
        self._sent_gold_arc, self._sent_gold_rel = None, None
        self._sent_words = None
        self._count = None
        self._sys_arcs = None
        self._sys_rels = None
        self._gold_arcs = None
        self._gold_rels = None
        self._words = None

        self._gold_path = self.output_path + '.gold.txt'
        self._system_path = self.output_path + '.txt'
        self._system_file = None
        self._gold_file = None
        if not self.script_path:
            self.script_path = ''
        self._line_func = to_conllx_line if 'conllx' in self.script_path else to_conll09_line

    def start(self):
        self._system_file = file_io.FileIO(self._system_path, 'w')
        self._gold_file = file_io.FileIO(self._gold_path, 'w')
        self._sent_index = -1
        self._sent_arc_probs, self._sent_rel_probs = None, None
        self._sent_gold_arc, self._sent_gold_rel = None, None
        self._sent_words = []
        self._count = 0
        self._sys_arcs = []
        self._sys_rels = []
        self._gold_arcs = []
        self._gold_rels = []
        self._words = []

    def accumulate(self, instance, result):
        # here, we aggregate multiple predictions over the same sentence, as in a product of experts

        # plus 1 for head
        seq_len = instance[constants.LENGTH_KEY] if constants.LENGTH_KEY in instance else 1 + len(instance[constants.WORD_KEY])

        arc_probs = result[constants.ARC_PROBS][:seq_len, :seq_len]
        rel_probs = result[constants.REL_PROBS][:seq_len, :, :seq_len]

        if instance[constants.SENTENCE_INDEX] != self._sent_index:
            self._sent_index = instance[constants.SENTENCE_INDEX]
            self._write_result()
            self._sent_arc_probs = arc_probs
            self._sent_rel_probs = rel_probs
            self._sent_gold_arc = [0] + instance[constants.HEAD_KEY]
            self._sent_gold_rel = ['<ROOT>'] + instance[constants.DEPREL_KEY]
            if constants.WORD_KEY in instance:
                self._sent_words = instance[constants.WORD_KEY]
        else:
            self._sent_arc_probs += arc_probs
            self._sent_rel_probs += rel_probs

    def _write_result(self):
        if self._sent_arc_probs is not None:
            self._sent_arc_probs /= np.sum(self._sent_arc_probs, axis=1, keepdims=True)
            self._sent_rel_probs /= np.sum(self._sent_rel_probs, axis=1, keepdims=True)
            sys_arc, sys_rel = get_parse_prediction(self._sent_arc_probs, self._sent_rel_probs, self.target)

            write_parse_result_to_file(sys_arc, sys_rel, self._system_file, self._line_func, words=self._sent_words)
            write_parse_result_to_file(self._sent_gold_arc, self._sent_gold_rel, self._gold_file, self._line_func,
                                       words=self._sent_words)

            if not self.script_path:
                self._sys_arcs.append(sys_arc)
                self._sys_rels.append(sys_rel)
                self._gold_arcs.append(self._sent_gold_arc)
                self._gold_rels.append(self._sent_gold_rel)
                if len(self._sent_words) > 0:
                    self._words.append(self._sent_words)
                else:
                    self._words.append(['x'] * (len(self._sent_gold_arc) - 1))

    @staticmethod
    def _all_punct(word):
        return all(unicodedata.category(x) in PUNCT_CAT for x in word)

    def evaluate(self, identifier=None):
        self._write_result()

        self._system_file.close()
        self._gold_file.close()

        if self.script_path:
            res = subprocess.check_output(['perl', self.script_path, '-g', self._gold_path, '-s', self._system_path, '-q'],
                                          universal_newlines=True)
            lines = res.split('\n')
            las = float(lines[0].strip().split()[9])
            uas = float(lines[1].strip().split()[9])
            la = float(lines[2].strip().split()[9])
        else:
            corr_uas = 0
            corr_la = 0
            corr_las = 0
            total = 0
            for words, sys_arc, sys_rel, gold_arc, gold_rel in zip(self._words, self._sys_arcs, self._sys_rels, self._gold_arcs,
                                                                   self._gold_rels):
                for word, arc, rel, garc, grel in zip(words, sys_arc[1:], sys_rel[1:], gold_arc[1:], gold_rel[1:]):
                    if DepParserEvaluator._all_punct(word):
                        continue
                    total += 1
                    if arc == garc:
                        corr_uas += 1
                    if rel == grel:
                        corr_la += 1
                    if arc == garc and rel == grel:
                        corr_las += 1
            la = 100 * (corr_la / total)
            las = 100 * (corr_las / total)
            uas = 100 * (corr_uas / total)
            res = 'LA: %f\nUAS: %f\nLAS: %f' % (la, uas, las)

        tf.logging.info('\n%s', res)
        append_prediction_output(identifier=self.target.name,
                                 header='ID\tLA\tUAS\tLAS',
                                 line='%s\t%f\t%f\t%f' % (identifier, la, uas, las),
                                 detailed=res,
                                 output_path=self.output_path)
        self.metric, self.summary = las, res


class SrlEvaluator(TaggerEvaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)
        self.markers = None

    def start(self):
        super().start()
        self.markers = []

    def accumulate(self, instance, result):
        super().accumulate(instance, result)
        self.markers.append(instance[constants.MARKER_KEY])

    def evaluate(self, identifier='.'):
        write_props_to_file(self.output_path + '.gold.txt', self.gold, self.markers, self.indices)
        write_props_to_file(self.output_path + '.txt', self.labels, self.markers, self.indices)

        result = conll_srl_eval(self.gold, self.labels, self.markers, self.indices)
        res = str(result)
        tf.logging.info(res)
        p, r, f1 = result.evaluation.prec_rec_f1()
        if self.output_path is not None:
            line = '%s\t%d\t%f\t%f\t%f\t%f' % (identifier,
                                               result.ntargets,
                                               result.perfect_props(),
                                               p, r, f1)
            append_prediction_output(identifier=self.target.name,
                                     header='ID\t# Props\t% Perfect\tPrecision\tRecall\tF1',
                                     line=line,
                                     detailed=res,
                                     output_path=self.output_path,
                                     confusions=result.confusion_matrix())

        self.metric, self.summary = f1, result
