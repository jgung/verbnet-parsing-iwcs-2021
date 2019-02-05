import os

import tensorflow as tf
from tensorflow.contrib.predictor import from_saved_model

from tfnlp.cli.formatters import get_formatter
from tfnlp.cli.parsers import get_parser
from tfnlp.common import constants
from tfnlp.common.config import get_network_config
from tfnlp.common.utils import read_json
from tfnlp.feature import get_feature_extractor


class Predictor(object):
    """
    General entry point for making predictions using saved models.
    """

    def __init__(self, predictor, parser_function, feature_function, formatter, batcher=None):
        super().__init__()
        self._predictor = predictor  # Tensorflow Predictor
        self._parser_function = parser_function  # given a text input, produces a list of inputs for features
        self._feature_function = feature_function  # feature extractor, which converts inputs to inputs for prediction
        self._formatter = formatter  # formatter for predictor output -- takes inputs from parser function and predictor output
        self._batcher = batcher if batcher else lambda x: x  # batching function

    def predict(self, text, formatted=True):
        inputs = self._parser_function(text)
        return self.predict_inputs(inputs, formatted)

    def predict_inputs(self, inputs, formatted=True):
        examples = []
        for processed_input in inputs:
            examples.append(self._feature_function(processed_input))

        predictions = self._batcher(examples, self._predictor)

        if not formatted:
            return predictions

        formatted = []
        for prediction, processed_input in zip(predictions, inputs):
            formatted.append(self._formatter(prediction, processed_input))
        return formatted


def default_batching_function(batch_size):
    def batch_fn(examples, predictor):
        results = []
        for i in range(0, len(examples), batch_size):
            batch_end = min(i + batch_size, len(examples))
            batch = examples[i:batch_end]
            result = predictor(batch)
            for idx in range(len(batch)):
                single_result = {}
                for key, val in result.items():
                    single_result[key] = val[idx]
                results.append(single_result)
        return results

    return batch_fn


def from_job_dir(job_dir):
    """
    Initialize a predictor from the output directory of a trainer.
    :param job_dir: output directory of trainer
    :return: initialized predictor
    """
    path_to_savedmodel = get_latest_savedmodel_from_jobdir(job_dir)
    path_to_vocab = os.path.join(job_dir, constants.VOCAB_PATH)
    path_to_config = os.path.join(job_dir, constants.CONFIG_PATH)
    return from_config_and_savedmodel(path_to_config, path_to_savedmodel, path_to_vocab)


def get_latest_savedmodel_from_jobdir(job_dir):
    """
    Return the latest saved model from a given output directory of a trainer.
    """
    export_dir = os.path.join(job_dir, constants.MODEL_PATH, 'export', 'best_exporter')
    latest = os.path.join(export_dir, max(os.listdir(export_dir)))
    return latest


def from_config_and_savedmodel(path_to_config, path_to_savedmodel, path_to_vocab):
    """
    Initialize a savedmodel from a configuration, saved model, and vocabulary.
    :param path_to_config: path to trainer configuration
    :param path_to_savedmodel: path to TF saved model
    :param path_to_vocab: path to vocabulary directory
    :return: initialized predictor
    """
    config = get_network_config(read_json(path_to_config))

    tf.logging.info("Loading predictor from saved model at %s" % path_to_savedmodel)
    tf_predictor = default_predictor(path_to_savedmodel)
    parser_function = get_parser(config)
    feature_function = _get_feature_function(config.features, path_to_vocab)
    formatter = get_formatter(config)

    return Predictor(tf_predictor, parser_function, feature_function, formatter, default_batching_function(config.batch_size))


def _get_feature_function(config, path_to_vocab):
    feature_extractor = get_feature_extractor(config)
    feature_extractor.read_vocab(path_to_vocab)

    return lambda instance: feature_extractor.extract(instance).SerializeToString()


def default_predictor(path_to_savedmodel):
    base_predictor = from_saved_model(path_to_savedmodel)
    return lambda features: base_predictor({"examples": features})
