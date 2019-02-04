from tfnlp.common import constants
from tfnlp.common.constants import LABEL_KEY, SENTENCE_INDEX
from tfnlp.common.eval import conll_eval, conll_srl_eval
from tfnlp.common.utils import binary_np_array_to_unicode


def get_evaluator(config):
    head_type = [head.type for head in config.heads][0]
    evaluators = {
        constants.TAGGER_KEY: tagger_evaluator,
        constants.SRL_KEY: srl_evaluator,
        constants.NER_KEY: tagger_evaluator,
    }
    if head_type not in evaluators:
        raise ValueError("Unsupported head type: " + head_type)
    return lambda labeled_instances, results, output_path: evaluators[head_type](labeled_instances, results, output_path,
                                                                                 config.heads[0].name)


def tagger_evaluator(labeled_instances, results, output_path=None, target_key=None):
    """
    Perform standard tagger evaluation on a given list of gold labeled instances.
    :param labeled_instances: labeled instances
    :param results: prediction results corresponding to labeled instances
    :param output_path: path to output results to, or if none, use stdout
    :param target_key: result key for predictions
    """
    target_key = LABEL_KEY if not target_key else target_key
    labels = []
    gold = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append(binary_np_array_to_unicode(result[target_key]))
        gold.append(instance[LABEL_KEY])
        indices.append(instance[SENTENCE_INDEX])
    f1, result_str = conll_eval(gold, labels, indices, output_file=output_path)
    print(result_str)


def srl_evaluator(labeled_instances, results, output_path=None, target_key=None):
    """
    Perform SRL evaluation on a given list of gold labeled instances.
    :param labeled_instances: labeled instances
    :param results: prediction results corresponding to labeled instances
    :param output_path: path to output results to, or if none, use stdout
    :param target_key: result key for predictions
    """
    labels = []
    gold = []
    markers = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append(binary_np_array_to_unicode(result[target_key]))
        gold.append(instance[constants.LABEL_KEY])
        markers.append(instance[constants.MARKER_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])
    result = conll_srl_eval(gold, labels, markers, indices)
    print(str(result))
