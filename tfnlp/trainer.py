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
from tfnlp.common.constants import F1_METRIC_KEY, WORD_KEY, LABEL_KEY
from tfnlp.common.eval import BestExporter
from tfnlp.common.utils import read_json
from tfnlp.datasets import make_dataset
from tfnlp.feature import write_features
from tfnlp.model.tagger import model_func
from tfnlp.readers import conll_2003_reader

tf.logging.set_verbosity(tf.logging.INFO)


def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='File containing training data.')
    parser.add_argument('--valid', type=str, help='File containing validation data.')
    parser.add_argument('--test', type=str, help='File containing test data.')
    parser.add_argument('--save', type=str, required=True, help='Directory where models/checkpoints are saved.')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary base directory.')
    parser.add_argument('--mode', type=str, default="train", help='Command in [train, predict]')
    parser.add_argument('--features', type=str, required=True, help='JSON config file for initializing feature extractors')
    parser.add_argument('--config', type=str, required=True, help='JSON config file for additional training options')
    parser.add_argument('--script', type=str, help='Optional path to evaluation script')
    return parser


def default_parser(sentence):
    example = {WORD_KEY: sentence.split()}
    return example


def default_formatter(result):
    return ' '.join([bstr.decode('utf-8') for bstr in result['output'][0].tolist()])


def default_input_fn(features):
    return {"examples": features}


class Trainer(object):
    def __init__(self, args=None):
        super().__init__()
        args = self._validate_and_parse_args(args)
        self._mode = args.mode
        self._raw_train = args.train
        self._raw_valid = args.valid
        self._raw_test = args.test

        self._save_path = args.save
        self._vocab_path = args.vocab
        self._feature_config = read_json(args.features)
        self._training_config = read_json(args.config)
        self._eval_script_path = args.script
        self._output_path = None
        self._log_path = None

        self._parse_fn = default_parser
        self._feature_extractor = None
        self._prediction_formatter_fn = default_formatter
        self._predict_input_fn = default_input_fn
        self._model_fn = model_func
        self._data_path_fn = lambda orig: orig + ".tfr"
        self._raw_instance_reader_fn = lambda raw_path: conll_2003_reader().read_file(raw_path)

    # noinspection PyMethodMayBeStatic
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
        self._init_feature_extractor()
        estimator = tf.estimator.Estimator(model_fn=self._model_fn, model_dir=self._save_path,
                                           config=RunConfig(save_checkpoints_steps=2000),
                                           params=self._params())

        def train_input_fn():
            return make_dataset(self._feature_extractor, paths=self._data_path_fn(self._raw_train),
                                batch_size=self._training_config.batch_size)

        def eval_input_fn():
            return make_dataset(self._feature_extractor, paths=self._data_path_fn(self._raw_valid),
                                batch_size=self._training_config.batch_size, evaluate=True)

        exporter = BestExporter(serving_input_receiver_fn=self._serving_input_fn(), compare_key=F1_METRIC_KEY)
        train_and_evaluate(estimator, train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None),
                           eval_spec=tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, exporters=[exporter],
                                                           throttle_secs=60))

    def eval(self):
        self._feature_extractor = get_feature_extractor(self._feature_config)
        self._feature_extractor.read_vocab(self._vocab_path)
        self._feature_extractor.test()

        examples = []
        for instance in self._raw_instance_reader_fn(self._raw_test):
            examples.append(self._feature_extractor.extract(instance))
        write_features(examples, self._data_path_fn(self._raw_test))

        estimator = tf.estimator.Estimator(model_fn=self._model_fn, model_dir=self._save_path,
                                           config=RunConfig(save_checkpoints_steps=2000),
                                           params=self._params())

        def eval_input_fn():
            return make_dataset(self._feature_extractor, paths=self._data_path_fn(self._raw_test),
                                batch_size=self._training_config.batch_size, evaluate=True)

        estimator.evaluate(eval_input_fn)

    def predict(self):
        raise NotImplementedError

    def itl(self):
        self._feature_extractor = get_feature_extractor(self._feature_config)
        self._feature_extractor.read_vocab(self._vocab_path)
        self._feature_extractor.test()

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
        self._train_vocab()

    def _train_vocab(self):
        print("Reading/writing features...")
        self._feature_extractor.train()
        self._feature_extractor.initialize()

        examples = []
        for instance in self._raw_instance_reader_fn(self._raw_train):
            examples.append(self._feature_extractor.extract(instance))
        write_features(examples, self._data_path_fn(self._raw_train))

        self._feature_extractor.test()
        examples = []
        for instance in self._raw_instance_reader_fn(self._raw_valid):
            examples.append(self._feature_extractor.extract(instance))
        write_features(examples, self._data_path_fn(self._raw_valid))

        self._feature_extractor.write_vocab(self._vocab_path)

        self._feature_extractor.test()

    def _serving_input_fn(self):
        def serving_input_receiver_fn():
            serialized_tf_example = array_ops.placeholder(dtype=dtypes.string, shape=None, name='input_example_tensor')
            receiver_tensors = {'examples': serialized_tf_example}
            features = self._feature_extractor.parse(serialized_tf_example, train=False)
            features = {key: tf.expand_dims(val, axis=0) for key, val in features.items()}
            return ServingInputReceiver(features, receiver_tensors)

        return serving_input_receiver_fn

    def _params(self):
        params = HParams(extractor=self._feature_extractor,
                         config=self._training_config,
                         script_path=self._eval_script_path,
                         label_vocab_path=os.path.join(self._vocab_path, LABEL_KEY))
        return params


if __name__ == '__main__':
    Trainer().run()
