import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tfnlp.common import constants
from tfnlp.common.constants import LABEL_KEY, SENTENCE_INDEX
from tfnlp.common.eval import conll_eval, conll_srl_eval, write_props_to_file, SUMMARY_FILE, EVAL_LOG
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
    return EvaluatorWrapper(evaluator=evaluators[head_type], target=config.heads[0].name)


class EvaluatorWrapper(object):

    def __init__(self, evaluator, target):
        super().__init__()
        self.evaluator = evaluator
        self.target = target

    def __call__(self, labeled_instances, results, output_path=None):
        """
        Perform standard evaluation on a given list of gold labeled instances.
        :param labeled_instances: labeled instances
        :param results: prediction results corresponding to labeled instances
        :param output_path: path to output results to, or if none, use stdout
        """
        return self.evaluator(labeled_instances, results, output_path, target_key=self.target)


def tagger_evaluator(labeled_instances, results, output_path=None, target_key=None):
    target_key = LABEL_KEY if not target_key else target_key
    labels = []
    gold = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append(binary_np_array_to_unicode(result[target_key]))
        gold.append(instance[LABEL_KEY])
        indices.append(instance[SENTENCE_INDEX])
    f1, result_str = conll_eval(gold, labels, indices, output_file=output_path)
    tf.logging.info(result_str)


def srl_evaluator(labeled_instances, results, output_path=None, target_key=None):
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
    p, r, f1 = result.evaluation.prec_rec_f1()

    job_dir = os.path.dirname(output_path)

    # append results to summary file
    summary_file = os.path.join(job_dir, SUMMARY_FILE)
    exists = tf.gfile.Exists(summary_file)
    with file_io.FileIO(summary_file, 'a') as summary:
        if not exists:
            summary.write('Path\t# Props\t% Perfect\tPrecision\tRecall\tF1\n')
        summary.write('%s\t%d\t%f\t%f\t%f\t%f\n' % (os.path.basename(output_path),
                                                    result.ntargets,
                                                    result.perfect_props(),
                                                    p, r, f1))

    # append evaluation log
    with file_io.FileIO(os.path.join(job_dir, EVAL_LOG), 'a') as eval_log:
        eval_log.write('\n%d\t%s\n' % os.path.basename(output_path))
        eval_log.write(str(result) + '\n')
        eval_log.write('\n%s\n\n' % result.confusion_matrix())
