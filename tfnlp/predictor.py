import os
from typing import Callable, Iterable, List, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.io import gfile

from tfnlp.cli.formatters import get_formatter
from tfnlp.cli.parsers import get_parser
from tfnlp.common import constants
from tfnlp.common.config import get_network_config
from tfnlp.common.utils import read_json, binary_np_array_to_unicode
from tfnlp.feature import get_feature_extractor


class Predictor(object):
    """
    General entry point for making predictions using saved models.

    :param predictor: TF Predictor
    :param parser_function: given a text input, produces a list of inputs for features
    :param feature_function: feature extractor, which converts raw inputs to inputs for prediction
    :param formatter: formatter for predictor output -- takes inputs from parser function and predictor output
    :param batcher: batching function, specifying logic for calling predictor
    """

    def __init__(self, predictor: Callable[[Iterable[str]], dict], parser_function: Callable[[str], Iterable[dict]],
                 feature_function: Callable[[dict], str], formatter: [[object, dict], str],
                 batcher: Callable[[Iterable[dict], Callable[[Iterable[dict]], dict]], Iterable[Tuple[dict, dict]]]) -> None:
        super().__init__()

        def _predictor_from_dict(raw_feats: Iterable[dict]) -> dict:
            processed = [feature_function(raw) for raw in raw_feats]
            return predictor(processed)

        self._predictor = _predictor_from_dict
        self._parser_function = parser_function
        self._formatter = formatter
        self._batcher = batcher

    def predict(self, text, formatted=True) -> Union[List[str], List[dict]]:
        """
        Predict from raw text, applying a parsing function to generate an input dictionary for each instance found.
        :param text: raw, un-tokenized text
        :param formatted: if True, return a textual representation of output
        :return: return a list of results for instances found in text
        """
        inputs = self._parser_function(text)
        return self.predict_parsed(inputs, formatted)

    def predict_parsed(self, inputs: Iterable[dict], formatted: bool = True) -> Union[List[str], Iterable[dict]]:
        """
        Predict from a list of pre-parsed instance dictionaries.
        :param inputs: input dictionaries
        :param formatted: if True, return a textual representation of output
        :return: return a list of results corresponding to each input instance dictionary
        """
        for processed_input, prediction in self._batcher(inputs, self._predictor):
            if not formatted:
                yield prediction
            else:
                yield self._formatter(prediction, processed_input)


def default_batching_function(batch_size: int) -> Callable[[Iterable[dict], Callable[[Iterable[dict]], dict]],
                                                           Iterable[Tuple[dict, dict]]]:
    """
    Returns a function that performs batching over a list of serialized examples, and returns a list of resulting dictionaries.
    :param batch_size: maximum batch size to use (limited by input Predictor's max batch size)
    """

    def _batch_fn(examples: Iterable[dict], predictor: Callable[[Iterable[dict]], dict]) -> Iterable[Tuple[dict, dict]]:

        curr_batch = []

        def _single_batch(_batch):
            result = predictor(_batch)
            for idx in range(len(_batch)):
                single_result = {}
                for key, val in result.items():
                    value = val[idx]
                    if isinstance(value, np.ndarray) and not np.issubdtype(value.dtype, np.number) and len(value.shape) > 0:
                        value = binary_np_array_to_unicode(value)
                    single_result[key] = value
                yield _batch[idx], single_result

        for example in examples:
            if len(curr_batch) == batch_size:
                yield from _single_batch(curr_batch)
                curr_batch = []
            curr_batch.append(example)

        if len(curr_batch) > 0:
            yield from _single_batch(curr_batch)

    return _batch_fn


def from_job_dir(job_dir: str) -> type(Predictor):
    """
    Initialize a predictor from the output directory of a trainer.
    :param job_dir: output directory of trainer
    :return: initialized predictor
    """
    path_to_savedmodel = get_latest_savedmodel_from_jobdir(job_dir)
    path_to_vocab = os.path.join(job_dir, constants.VOCAB_PATH)
    path_to_config = os.path.join(job_dir, constants.CONFIG_PATH)
    return from_config_and_savedmodel(path_to_config, path_to_savedmodel, path_to_vocab)


def get_latest_savedmodel_from_jobdir(job_dir: str) -> type(Predictor):
    """
    Return the latest saved model from a given output directory of a trainer.
    :param job_dir: output directory of trainer
    """
    export_dir = os.path.join(job_dir, constants.MODEL_PATH, 'export', 'best_exporter')
    latest = os.path.join(export_dir, max(
        [path for path in gfile.listdir(export_dir) if not path.startswith('temp')]))
    return latest


def from_config_and_savedmodel(path_to_config: str, path_to_savedmodel: str, path_to_vocab: str) -> type(Predictor):
    """
    Initialize a savedmodel from a configuration, saved model, and vocabulary.
    :param path_to_config: path to trainer configuration
    :param path_to_savedmodel: path to TF saved model
    :param path_to_vocab: path to vocabulary directory
    :return: initialized predictor
    """
    config = get_network_config(read_json(path_to_config))

    tf.logging.info("Loading predictor from saved model at %s" % path_to_savedmodel)
    tf_predictor = _default_predictor(path_to_savedmodel)
    parser_function = get_parser(config)
    feature_function = _get_feature_function(config.features, path_to_vocab)
    formatter = get_formatter(config)

    return Predictor(tf_predictor, parser_function, feature_function, formatter, default_batching_function(config.batch_size))


def _get_feature_function(config: object, path_to_vocab: str) -> Callable[[dict], str]:
    feature_extractor = get_feature_extractor(config)
    feature_extractor.read_vocab(path_to_vocab)

    return lambda instance: feature_extractor.extract(instance, train=False).SerializeToString()


def _default_predictor(path_to_savedmodel: str) -> Callable[[Iterable[str]], dict]:
    base_predictor = from_saved_model(path_to_savedmodel)
    return lambda features: base_predictor({"examples": features})
