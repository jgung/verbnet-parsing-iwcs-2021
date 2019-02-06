import os

import tensorflow as tf

from tfnlp.common import constants
from tfnlp.common.eval import append_srl_prediction_output, write_props_to_file
from tfnlp.common.eval import conll_eval, conll_srl_eval, parser_write_and_eval
from tfnlp.common.utils import binary_np_array_to_unicode


def get_evaluator(config):
    head_type = [head.type for head in config.heads][0]
    evaluators = {
        constants.TAGGER_KEY: tagger_evaluator,
        constants.SRL_KEY: srl_evaluator,
        constants.NER_KEY: tagger_evaluator,
        constants.PARSER_KEY: dep_evaluator,
    }
    if head_type not in evaluators:
        raise ValueError("Unsupported head type: " + head_type)
    return EvaluatorWrapper(evaluator=evaluators[head_type], target=config.heads[0].name)


class EvaluatorWrapper(object):

    def __init__(self, evaluator, target):
        super().__init__()
        self.evaluator = evaluator
        self.target = target

    def __call__(self, labeled_instances, results, output_path=None, script_path=None):
        """
        Perform standard evaluation on a given list of gold labeled instances.
        :param labeled_instances: labeled instances
        :param results: prediction results corresponding to labeled instances
        :param output_path: path to output results to, or if none, use stdout
        """
        return self.evaluator(labeled_instances, results, output_path, target_key=self.target, script_path=None)


def tagger_evaluator(labeled_instances, results, output_path=None, target_key=None, script_path=None):
    target_key = constants.LABEL_KEY if not target_key else target_key
    labels = []
    gold = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append(binary_np_array_to_unicode(result[target_key]))
        gold.append(instance[constants.LABEL_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])
    f1, result_str = conll_eval(gold, labels, indices, output_file=output_path)
    tf.logging.info(result_str)


def dep_evaluator(labeled_instances, results, output_path=None, target_key=None, script_path=None):
    arc_probs = []
    rel_probs = []
    gold_arcs = []
    gold_rels = []

    for instance, result in zip(labeled_instances, results):
        seq_len = 1 + len(instance[constants.WORD_KEY])  # plus 1 for head
        gold_arcs.append([0] + instance[constants.HEAD_KEY])
        gold_rels.append(['<ROOT>'] + instance[constants.DEPREL_KEY])

        arc_probs.append(result[constants.ARC_PROBS][:seq_len, :seq_len])
        rel_probs.append(result[constants.REL_PROBS])

    parser_write_and_eval(arc_probs=arc_probs,
                          rel_probs=rel_probs,
                          heads=gold_arcs,
                          rels=gold_rels,
                          script_path=script_path)


def srl_evaluator(labeled_instances, results, output_path=None, target_key=None, script_path=None):
    labels = []
    gold = []
    markers = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append(binary_np_array_to_unicode(result[target_key]))
        gold.append(instance[constants.LABEL_KEY])
        markers.append(instance[constants.MARKER_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])

    write_props_to_file(output_path + '.gold.conll', gold, markers, indices)
    write_props_to_file(output_path + '.conll', gold, markers, indices)

    result = conll_srl_eval(gold, labels, markers, indices)
    tf.logging.info(result)

    job_dir = os.path.dirname(output_path)

    # append results to summary file
    append_srl_prediction_output(os.path.basename(output_path), result, job_dir, output_confusions=True)
