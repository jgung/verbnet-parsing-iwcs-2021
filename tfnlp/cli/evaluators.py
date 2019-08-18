import subprocess
from typing import List

import numpy as np
import tensorflow as tf
from common.config import append_label
from tensorflow.python.lib.io import file_io
from tfnlp.common import constants
from tfnlp.common.bert import BERT_SUBLABEL
from tfnlp.common.eval import conll_eval, conll_srl_eval
from tfnlp.common.eval import write_props_to_file, accuracy_eval, get_parse_prediction, \
    to_conll09_line, to_conllx_line, write_parse_result_to_file


def get_evaluator(heads, feature_extractor, output_path, script_path):
    evaluators = {
        constants.TAGGER_KEY: TaggerEvaluator,
        constants.SRL_KEY: SrlEvaluator,
        constants.NER_KEY: TaggerEvaluator,
        constants.PARSER_KEY: DepParserEvaluator,
        constants.TOKEN_CLASSIFIER_KEY: TokenClassifierEvaluator
    }

    evals = []
    for head in heads:
        if head.task not in evaluators:
            raise ValueError("Unsupported head type: " + head.task)
        evaluator = evaluators[head.task](target=feature_extractor.targets[head.name],
                                          output_path=output_path + '.' + head.name,
                                          script_path=script_path)
        evals.append(evaluator)
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
        self.evaluate()

    def start(self):
        pass

    def accumulate(self, instance, result):
        pass

    def evaluate(self):
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

    def evaluate(self):
        total, summaries = 0, []
        for evaluator in self._evaluators:
            evaluator.evaluate()
            total += evaluator.metric
            summaries.append(str(evaluator.summary))
        self.metric, self.summary = total / len(self._evaluators), '\n'.join(summaries)


class TokenClassifierEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)
        self.labels = None
        self.gold = None
        self.indices = None
        self.target_key = constants.LABEL_KEY if not self.target.name else self.target.name
        self.labels_key = constants.LABEL_KEY if not self.target.key else self.target.key
        self.scores_name = append_label(constants.LABEL_SCORES, self.target_key)

    def start(self):
        self.labels = []
        self.gold = []
        self.indices = []

    def accumulate(self, instance, result):
        if self.target.constraints:
            scores = {self.target.index_to_feat(i): score for i, score in enumerate(result[self.scores_name])}
            ck = instance[self.target.constraint_key]
            valid_scores = {label: score for label, score in scores.items() if label in self.target.constraints.get(ck, [])}
            label = max(valid_scores.items(), key=lambda x: x[1], default=(constants.UNKNOWN_WORD, 0))[0]
            self.labels.append(label)
        else:
            self.labels.append(result[self.target_key].decode('utf-8'))
        self.gold.append(instance[self.labels_key])
        self.indices.append(instance[constants.SENTENCE_INDEX])

    def evaluate(self):
        self.metric, self.summary = accuracy_eval(self.gold, self.labels, self.indices, output_file=self.output_path + '.txt'), ''


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

    def evaluate(self):
        f1, result_str = conll_eval(self.gold, self.labels, self.indices, output_file=self.output_path + '.txt')
        tf.logging.info(result_str)
        self.metric, self.summary = f1, result_str


class DepParserEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)
        if not script_path:
            raise AssertionError('Dependency parse evaluation requires a script script to be provided.')
        self._sent_index = None
        self._sent_arc_probs, self._sent_rel_probs = None, None
        self._sent_gold_arc, self._sent_gold_rel = None, None
        self._count = None

        self._gold_path = self.output_path + '.gold.txt'
        self._system_path = self.output_path + '.txt'
        self._system_file = None
        self._gold_file = None
        self._line_func = to_conllx_line if 'conllx' in self.script_path else to_conll09_line

    def start(self):
        self._system_file = file_io.FileIO(self._system_path, 'w')
        self._gold_file = file_io.FileIO(self._gold_path, 'w')
        self._sent_index = -1
        self._sent_arc_probs, self._sent_rel_probs = None, None
        self._sent_gold_arc, self._sent_gold_rel = None, None
        self._count = 0

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
        else:
            self._sent_arc_probs += arc_probs
            self._sent_rel_probs += rel_probs

    def _write_result(self):
        if self._sent_arc_probs is not None:
            self._sent_arc_probs /= np.sum(self._sent_arc_probs, axis=1, keepdims=True)
            self._sent_rel_probs /= np.sum(self._sent_rel_probs, axis=1, keepdims=True)
            sys_arc, sys_rel = get_parse_prediction(self._sent_arc_probs, self._sent_rel_probs, self.target)

            write_parse_result_to_file(sys_arc, sys_rel, self._system_file, self._line_func)
            write_parse_result_to_file(self._sent_gold_arc, self._sent_gold_rel, self._gold_file, self._line_func)

    def evaluate(self):
        self._write_result()

        self._system_file.close()
        self._gold_file.close()

        res = subprocess.check_output(['perl', self.script_path, '-g', self._gold_path, '-s', self._system_path, '-q'],
                                      universal_newlines=True)
        tf.logging.info('\n%s', res)
        lines = res.split('\n')
        las = float(lines[0].strip().split()[9])
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

    def evaluate(self):
        write_props_to_file(self.output_path + '.gold.txt', self.gold, self.markers, self.indices)
        write_props_to_file(self.output_path + '.txt', self.gold, self.markers, self.indices)

        result = conll_srl_eval(self.gold, self.labels, self.markers, self.indices)
        tf.logging.info(str(result))

        self.metric, self.summary = result.evaluation.prec_rec_f1()[2], result
