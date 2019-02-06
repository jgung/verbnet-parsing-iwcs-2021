import argparse
import os
import sys
from typing import Union, Iterable, Callable, Optional

import tensorflow as tf
from tensorflow.contrib.estimator import stop_if_no_increase_hook
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.export.export import ServingInputReceiver
from tensorflow.python.estimator.exporter import BestExporter
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops

from tfnlp.cli.evaluators import get_evaluator
from tfnlp.common import constants
from tfnlp.common.config import get_network_config
from tfnlp.common.eval import metric_compare_fn
from tfnlp.common.logging import set_up_logging
from tfnlp.common.utils import read_json, write_json
from tfnlp.datasets import make_dataset, padded_batch
from tfnlp.feature import get_default_buckets, get_feature_extractor, write_features
from tfnlp.model.model import multi_head_model_fn
from tfnlp.predictor import get_latest_savedmodel_from_jobdir, from_job_dir
from tfnlp.readers import get_reader

TF_MODEL_FN = Callable[[dict, str, type(HParams)], type(tf.estimator.EstimatorSpec)]


class Trainer(object):
    """
    Model trainer, used to train and evaluate models using the TF Estimator API.

    :param save_dir_path: directory from which to save/load model, metadata, logs, and other training files
    :param config_json_path: path to training configuration file
    :param resources_dir_path: path to base directory of resources, such as for pre-trained weights
    :param script_file_path: path to official evaluation scripts
    :param model_fn: TF model function ([features, mode, params] -> EstimatorSpec)
    """

    def __init__(self,
                 save_dir_path: str,
                 config_json_path: Optional[str] = None,
                 resources_dir_path: Optional[str] = '',
                 script_file_path: Optional[str] = None,
                 model_fn: Optional[TF_MODEL_FN] = multi_head_model_fn) -> None:
        super().__init__()
        self._job_dir = save_dir_path

        self._model_path = os.path.join(self._job_dir, constants.MODEL_PATH)
        self._vocab_path = os.path.join(self._job_dir, constants.VOCAB_PATH)
        self._resources = resources_dir_path
        self._eval_script_path = script_file_path
        self._model_fn = model_fn

        # read configuration file
        self.config_path = os.path.join(self._job_dir, constants.CONFIG_PATH)
        if not tf.gfile.Exists(self.config_path):
            if not config_json_path:
                raise AssertionError('trainer configuration is required when training for the first time')
            tf.gfile.MakeDirs(self._job_dir)
            tf.gfile.Copy(config_json_path, self.config_path, overwrite=True)
        self._training_config = get_network_config(read_json(self.config_path))

        self._data_path_fn = lambda orig: os.path.join(self._job_dir, os.path.basename(orig) + ".tfrecords")

        self._feature_extractor = None

    def train(self, train: str, valid: str) -> None:
        """
        Train a new model with this trainer, or if a model already exists in the save path for this trainer,
        resume training from a checkpoint.
        :param train: path to training corpus
        :param valid: path to validation corpus
        """
        if not self._feature_extractor:
            self._init_feature_extractor(train_path=train)
        estimator = self._init_estimator(test=False)

        tf.logging.info('Training on %s, validating on %s' % (train, valid))

        # read and extract features from training/validation data, serialize to disk
        self._extract_and_write(train)
        self._extract_and_write(valid)

        # train and evaluate using Estimator API
        # TODO: fixes issue https://github.com/tensorflow/tensorflow/issues/18394
        if not os.path.exists(estimator.eval_dir()):
            os.makedirs(estimator.eval_dir())

        early_stopping = stop_if_no_increase_hook(
            estimator,
            metric_name=self._training_config.metric,
            max_steps_without_increase=self._training_config.patience,
            min_steps=100,
            run_every_secs=None,
            run_every_steps=100,
        )

        train_spec = tf.estimator.TrainSpec(self._input_fn(train, True),
                                            max_steps=self._training_config.max_steps, hooks=[early_stopping])

        exporter = BestExporter(serving_input_receiver_fn=self._serving_input_fn,
                                compare_fn=metric_compare_fn(self._training_config.metric),
                                exports_to_keep=self._training_config.exports_to_keep)

        eval_spec = tf.estimator.EvalSpec(self._input_fn(valid, False),
                                          steps=None, exporters=[exporter], throttle_secs=0)

        train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)

    def eval(self, test_paths: Union[str, Iterable[str]]) -> None:
        """
        Evaluate a trained model on a given corpus or list of corpora.
        :param test_paths: paths to test corpora
        """
        if not isinstance(test_paths, Iterable):
            test_paths = [test_paths]

        if not self._feature_extractor:
            self._init_feature_extractor()

        # initialize predictor from saved trained model
        predictor = from_job_dir(self._job_dir)
        # get function that is used to evaluate predictions from configuration
        evaluation_fn = get_evaluator(self._training_config)

        for test_set in test_paths:
            tf.logging.info('Evaluating on %s' % test_set)
            output_path = os.path.join(self._model_path, os.path.basename(test_set) + '.eval')

            # extract instances from test file at given path--this is a generator, so wrap in a list
            instances = list(self._extract_raw(test_set, True))
            # predict from instances instead of raw text, so use .predict_inputs, don't format since we need the raw predictions
            processed_examples = predictor.predict_parsed(instances, formatted=False)
            # call evaluation function on predictions
            evaluation_fn(instances, processed_examples, output_path=output_path, script_path=self._eval_script_path)

    def predict(self, test_paths: Union[str, Iterable[str]]) -> None:
        """
        Generate predictions from plain text corpora at a given path or list of paths.
        :param test_paths: paths to documents for which to generate predictions
        """
        if not isinstance(test_paths, Iterable):
            test_paths = [test_paths]

        if not self._feature_extractor:
            self._init_feature_extractor()

        # initialize predictor from saved trained model
        predictor = from_job_dir(self._job_dir)

        for test_set in test_paths:
            prediction_path = os.path.join(self._job_dir, os.path.basename(test_set) + '.predictions.txt')
            tf.logging.info('Writing predictions on %s to %s' % (test_set, prediction_path))
            with file_io.FileIO(prediction_path, mode="w") as output:
                with file_io.FileIO(test_set, mode="r") as text_lines:
                    for line in text_lines:
                        line = line.strip()
                        if not line:
                            continue
                        predictions = predictor.predict(line)
                        for prediction in predictions:
                            output.write(str(prediction) + '\n')
                        output.write('\n')

    def itl(self) -> None:
        """
        Initiate a REPL-style interactive testing loop.
        """
        if not self._feature_extractor:
            self._init_feature_extractor()

        # initialize predictor from saved trained model
        predictor = from_job_dir(self._job_dir)

        while True:
            sentence = input(">>> ")
            if not sentence:
                continue
            if sentence.lower() in {'exit', 'quit'}:
                break
            predictions = predictor.predict(sentence)
            for prediction in predictions:
                print(str(prediction))

    def _get_predictor(self):
        latest = get_latest_savedmodel_from_jobdir(self._job_dir)
        tf.logging.info("Loading predictor from saved model at %s" % latest)
        return from_saved_model(latest)

    def _init_feature_extractor(self, train_path: str = None):
        self._feature_extractor = get_feature_extractor(self._training_config.features)
        if train_path:
            tf.logging.info("Checking for pre-existing vocabulary at vocabulary at %s", self._vocab_path)
            if self._feature_extractor.read_vocab(self._vocab_path):
                tf.logging.info("Loaded pre-existing vocabulary at %s", self._vocab_path)
            else:
                tf.logging.info("No valid pre-existing vocabulary found at %s "
                                "(this is normal when not loading from an existing model)", self._vocab_path)
                self._train_vocab(train_path)
        else:
            tf.logging.info("Checking for pre-existing vocabulary at vocabulary at %s", self._vocab_path)
            self._feature_extractor.read_vocab(self._vocab_path)
            tf.logging.info("Loaded pre-existing vocabulary at %s", self._vocab_path)

    def _extract_raw(self, path: str, test: bool = False):
        # TODO: allow for separate test reader configuration
        reader = get_reader(self._training_config.reader) if test else get_reader(self._training_config.reader,
                                                                                  self._training_config)
        raw_instances = reader.read_file(path)

        if not raw_instances:
            raise ValueError("No examples provided at path given by '{}'".format(path))
        return raw_instances

    def _train_vocab(self, train_path: str):
        tf.logging.info("Training new vocabulary using training data at %s", train_path)
        self._feature_extractor.initialize(self._resources)
        self._feature_extractor.train(self._extract_raw(train_path))
        self._feature_extractor.write_vocab(self._vocab_path, resources=self._resources, prune=True)

    def _extract_features(self, path: str, test: bool = False):
        tf.logging.info("Extracting features from %s", path)
        examples = self._feature_extractor.extract_all(self._extract_raw(path, test))
        return examples

    def _extract_and_write(self, path: str, test: bool = False):
        output_path = self._data_path_fn(path)
        if tf.gfile.Exists(output_path):
            tf.logging.info("Using existing features for %s from %s", path, output_path)
            return
        examples = self._extract_features(path, test)
        tf.logging.info("Writing extracted features from %s for %d instances to %s", path, len(examples), output_path)
        write_features(examples, output_path)

    def _init_estimator(self, test: bool = False):
        return tf.estimator.Estimator(model_fn=self._model_fn, model_dir=self._model_path,
                                      config=RunConfig(
                                          keep_checkpoint_max=self._training_config.keep_checkpoints,
                                          save_checkpoints_steps=self._training_config.checkpoint_steps),
                                      params=self._params(test=test))

    def _serving_input_fn(self):
        # input has been serialized to a TFRecord string (variable batch size)
        serialized_tf_example = array_ops.placeholder(dtype=dtypes.string, shape=[None], name=constants.SERVING_PLACEHOLDER)
        # create single padded batch
        batch = padded_batch(self._feature_extractor, serialized_tf_example, self._training_config.batch_size)
        return ServingInputReceiver(batch, {"examples": serialized_tf_example})

    def _params(self, test: bool = False):
        return HParams(extractor=self._feature_extractor,
                       config=self._training_config,
                       script_path=self._eval_script_path,
                       vocab_path=self._vocab_path,
                       job_dir=self._job_dir,
                       verbose_eval=test)

    def _input_fn(self, dataset: str, train: bool = False):
        bucket_sizes = self._training_config.bucket_sizes
        if not bucket_sizes and constants.LENGTH_KEY in self._feature_extractor.features:
            length_feat = self._feature_extractor.feature(constants.LENGTH_KEY)
            bucket_sizes = get_default_buckets(length_feat.counts, self._training_config.batch_size * 2,
                                               max_length=self._training_config.max_length)
            if not bucket_sizes:
                bucket_sizes = None
            else:
                # persist dynamically computed bucket sizes
                self._training_config.bucket_sizes = bucket_sizes
                write_json(self._training_config, self.config_path)

        return lambda: make_dataset(self._feature_extractor, paths=self._data_path_fn(dataset),
                                    batch_size=self._training_config.batch_size, evaluate=not train,
                                    bucket_sizes=bucket_sizes)


TRAINING_MODES = {'train', 'predict', 'test', 'itl'}


def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='save', type=str, required=True,
                        help='models/checkpoints/vocabularies save path')
    parser.add_argument('--config', type=str, help='training configuration JSON')
    parser.add_argument('--resources', type=str, help='shared resources directory (such as for word embeddings)')
    parser.add_argument('--train', type=str, help='training data path')
    parser.add_argument('--valid', type=str, help='validation/development data path')
    parser.add_argument('--test', type=str, help='test data paths, comma-separated')
    parser.add_argument('--mode', type=str, default="train", help='(optional) training command, "train" by default',
                        choices=list(TRAINING_MODES))
    parser.add_argument('--script', type=str, help='(optional) evaluation script path')
    return parser


def _validate_and_parse_args():
    parser = default_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def cli():
    opts = _validate_and_parse_args()

    trainer = Trainer(save_dir_path=opts.save,
                      config_json_path=opts.config,
                      resources_dir_path=opts.resources,
                      script_file_path=opts.script)

    mode = opts.mode
    test_paths = [t for t in opts.test.split(',') if t.strip()] if opts.test else None

    if mode not in TRAINING_MODES:
        raise ValueError("Unexpected mode type: {}".format(mode))

    set_up_logging(os.path.join(opts.save, '{}.log'.format(mode)))

    if mode == 'train' and not opts.train and test_paths:
        tf.logging.info('No training set provided, defaulting to test mode for %s' % opts.test)
        mode = 'test'

    if mode == "train":
        if not opts.train:
            tf.logging.warn('train mode was selected, but no training set path was provided (use "--train path/to/train")')
            return
        elif not opts.valid:
            tf.logging.warn('train mode was selected, but no validation set path was provided (use "--valid path/to/valid")')
            return

        trainer.train(opts.train, opts.valid)

        if test_paths:
            trainer.eval(test_paths)

    elif mode == "predict":
        if not test_paths:
            tf.logging.warn('predict mode was selected, but no test paths were provided (use "--test path1,path2")')
            return

        trainer.predict(test_paths)

    elif mode == "test":
        if not test_paths:
            tf.logging.warn('test mode was selected, but no test paths were provided (use "--test path1,path2")')
            return

        trainer.eval(test_paths)

    elif mode == "itl":
        trainer.itl()


if __name__ == '__main__':
    cli()
