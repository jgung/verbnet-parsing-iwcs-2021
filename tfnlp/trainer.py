import argparse
import os

import tensorflow as tf
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.export.export import ServingInputReceiver
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

from tfnlp.common.config import get_feature_extractor
from tfnlp.common.constants import LABEL_KEY, WORD_KEY
from tfnlp.common.eval import BestExporter
from tfnlp.common.utils import read_json
from tfnlp.datasets import make_dataset
from tfnlp.feature import write_features
from tfnlp.model.parser import parser_model_func
from tfnlp.model.tagger import tagger_model_func
from tfnlp.readers import get_reader

tf.logging.set_verbosity(tf.logging.INFO)


def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='File containing training data.')
    parser.add_argument('--valid', type=str, help='File containing validation data.')
    parser.add_argument('--test', type=str, help='File containing test data.')
    parser.add_argument('--save', type=str, required=True, help='Directory where models/checkpoints are saved.')
    parser.add_argument('--resources', type=str, help='Base path to shared resources, such as word embeddings')
    parser.add_argument('--vocab', type=str, required=True, help='Directory where vocabulary files are saved.')
    parser.add_argument('--mode', type=str, default="train", help='Command in [train, predict]')
    parser.add_argument('--features', type=str, required=True, help='JSON file for configuring feature extractors')
    parser.add_argument('--config', type=str, required=True, help='JSON file for configuring training')
    parser.add_argument('--script', type=str, help='(Optional) Path to evaluation script')
    parser.add_argument('--type', type=str, default='tagger', help='(Optional) Model type')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite previous trained models and vocabularies')
    parser.set_defaults(overwrite=False)
    return parser


class Trainer(object):
    def __init__(self, args=None):
        super().__init__()
        args = self._validate_and_parse_args(args)
        self._mode = args.mode
        self._raw_train = args.train
        self._raw_valid = args.valid
        self._raw_test = args.test
        self._overwrite = args.overwrite

        self._save_path = args.save
        self._vocab_path = args.vocab
        self._resources = args.resources
        self._eval_script_path = args.script
        self._feature_config = read_json(args.features)
        self._training_config = read_json(args.config)
        self._model_fn = get_model_func(args.type)

        self._feature_extractor = None
        self._estimator = None

        self._raw_instance_reader_fn = lambda raw_path: get_reader(self._training_config.reader).read_file(raw_path)
        self._data_path_fn = lambda orig: orig + ".tfrecords"

        self._parse_fn = default_parser
        self._predict_input_fn = default_input_fn
        self._prediction_formatter_fn = default_formatter

    def _get_arg_parser(self):
        """
        Initialize argument parser. Override this method when adding new arguments.
        :return: argument parser instance
        """
        return default_args()

    def _validate_and_parse_args(self, args):
        """
        Parse arguments using argument parser. Can override for additional validation or logic.
        :param args: command-line arguments
        :return: parsed arguments
        """
        return self._get_arg_parser().parse_args(args)

    def run(self):
        self._init_feature_extractor()
        self._init_estimator()
        if self._mode == "train":
            self.train()
        elif self._mode == "eval":
            self.eval()
        elif self._mode == "predict":
            self.predict()
        elif self._mode == "loop":
            self.itl()
        else:
            raise ValueError("Unexpected mode type: {}".format(self._mode))

    def train(self):
        train_input_fn = self._input_fn(self._raw_train, True)
        valid_input_fn = self._input_fn(self._raw_valid, False)
        exporter = BestExporter(serving_input_receiver_fn=self._serving_input_fn)
        train_and_evaluate(self._estimator, train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None),
                           eval_spec=tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None, exporters=[exporter],
                                                           throttle_secs=self._training_config.throttle_secs))

    def eval(self):
        self._extract_features(self._raw_test, train=False)
        eval_input_fn = self._input_fn(self._raw_test, False)
        self._estimator.evaluate(eval_input_fn)

    def predict(self):
        raise NotImplementedError

    def itl(self):
        predictor = from_saved_model(self._save_path)
        while True:
            sentence = input(">>> ")
            example = self._parse_fn(sentence)
            features = self._feature_extractor.extract(example, train=False).SerializeToString()
            serialized_feats = self._predict_input_fn(features)
            result = predictor(serialized_feats)
            print(self._prediction_formatter_fn(result))

    def _init_feature_extractor(self):
        self._feature_extractor = get_feature_extractor(self._feature_config)
        if self._mode == "train":
            if not self._overwrite:
                success = self._feature_extractor.read_vocab(self._vocab_path)
                if success:
                    tf.logging.info("Loaded pre-existing vocabulary at %s.", self._vocab_path)
                else:
                    tf.logging.info("Unable to load pre-existing vocabulary at %s.", self._vocab_path)
                    self._train_vocab()
            else:
                self._train_vocab()
        else:
            self._feature_extractor.read_vocab(self._vocab_path)
            tf.logging.info("Loaded pre-existing vocabulary at %s.", self._vocab_path)
        self._feature_extractor.test()

    def _train_vocab(self):
        tf.logging.info("Training new vocabulary using training data at %s", self._raw_train)
        self._feature_extractor.initialize(self._resources)
        self._extract_features(self._raw_train, train=True)
        self._extract_features(self._raw_valid, train=False)
        self._feature_extractor.write_vocab(self._vocab_path, overwrite=self._overwrite, resources=self._resources)

    def _extract_features(self, path, train=False):
        self._feature_extractor.train(train)

        tf.logging.info("Extracting features from %s", path)
        raw_instances = self._raw_instance_reader_fn(path)
        examples = [self._feature_extractor.extract(instance) for instance in raw_instances]

        output_path = self._data_path_fn(path)
        tf.logging.info("Writing extracted features to %s", output_path)
        write_features(examples, output_path)

    def _init_estimator(self):
        self._estimator = tf.estimator.Estimator(model_fn=self._model_fn, model_dir=self._save_path,
                                                 config=RunConfig(save_checkpoints_steps=self._training_config.checkpoint_steps),
                                                 params=self._params())

    def _serving_input_fn(self):
        serialized_tf_example = array_ops.placeholder(dtype=dtypes.string, name='input_example_tensor')
        features = self._feature_extractor.parse(serialized_tf_example, train=False)
        features = {key: tf.expand_dims(val, axis=0) for key, val in features.items()}
        return ServingInputReceiver(features, self._predict_input_fn(serialized_tf_example))

    def _params(self):
        return HParams(extractor=self._feature_extractor,
                       config=self._training_config,
                       script_path=self._eval_script_path,
                       label_vocab_path=os.path.join(self._vocab_path, LABEL_KEY))

    def _input_fn(self, dataset, train=False):
        return lambda: make_dataset(self._feature_extractor, paths=self._data_path_fn(dataset),
                                    batch_size=self._training_config.batch_size, evaluate=not train)


def default_parser(sentence):
    return {WORD_KEY: sentence.split()}


def default_formatter(result):
    return ' '.join([bstr.decode('utf-8') for bstr in result['output'][0].tolist()])


def default_input_fn(features):
    return {"examples": features}


def get_model_func(model_type):
    if "tagger" == model_type:
        return tagger_model_func
    elif "parser" == model_type:
        return parser_model_func
    else:
        raise ValueError("Unexpected model type: " + model_type)


if __name__ == '__main__':
    Trainer().run()
