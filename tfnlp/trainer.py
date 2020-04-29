import argparse
import os
import sys
from typing import Union, Iterable, Callable, Optional

import tensorflow as tf
import tensorflow_estimator as tfe
from tensorflow.compat.v1 import logging
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.contrib.training import HParams
from tensorflow.io import gfile
from tensorflow.python.estimator.export.export import ServingInputReceiver
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops

from tfnlp.cli.evaluators import get_evaluator
from tfnlp.common import constants
from tfnlp.common.config import get_network_config
from tfnlp.common.eval_hooks import metric_compare_fn
from tfnlp.common.export import BesterExporter
from tfnlp.common.logging_utils import set_up_logging
from tfnlp.common.utils import read_json, write_json
from tfnlp.config_builder import read_config
from tfnlp.datasets import make_dataset, padded_batch
from tfnlp.feature import get_default_buckets, get_feature_extractor, write_features
from tfnlp.model.model import multi_head_model_fn
from tfnlp.predictor import get_latest_savedmodel_from_jobdir, from_job_dir
from tfnlp.readers import get_reader

TF_MODEL_FN = Callable[[dict, str, type(HParams)], type(tfe.estimator.EstimatorSpec)]


class Trainer(object):
    """
    Model trainer, used to train and evaluate models using the TF Estimator API.

    :param save_dir_path: directory from which to save/load model, metadata, logs, and other training files
    :param config: training configuration JSON
    :param resources_dir_path: path to base directory of resources, such as for pre-trained weights
    :param script_file_path: path to official evaluation scripts
    :param model_fn: TF model function ([features, mode, params] -> EstimatorSpec)
    :param debug: debug mode for training
    """

    def __init__(self,
                 save_dir_path: str,
                 config: Optional[dict] = None,
                 resources_dir_path: Optional[str] = '',
                 script_file_path: Optional[str] = None,
                 model_fn: Optional[TF_MODEL_FN] = multi_head_model_fn,
                 debug: bool = False) -> None:
        super().__init__()
        self._job_dir = save_dir_path

        config_path = os.path.join(self._job_dir, constants.CONFIG_PATH)
        if not gfile.exists(config_path):
            write_json(config, config_path)
        if not config:
            config = read_json(config_path)
        self._training_config = get_network_config(config)

        self._model_path = os.path.join(self._job_dir, constants.MODEL_PATH)
        self._vocab_path = os.path.join(self._job_dir, constants.VOCAB_PATH)
        self._resources = resources_dir_path
        self._eval_script_path = script_file_path
        self._model_fn = model_fn
        self._debug = debug

        self._data_path_fn = lambda orig: os.path.join(self._job_dir, os.path.basename(orig) + ".tfrecords")

        self._feature_extractor = None

    def train(self, train: str, valid: str, feats_only: bool = False) -> None:
        """
        Train a new model with this trainer, or if a model already exists in the save path for this trainer,
        resume training from a checkpoint.
        :param train: path to training corpus
        :param valid: path to validation corpus
        :param feats_only: only extract features, don't train
        """
        if not self._feature_extractor:
            self._init_feature_extractor(train_path=train)

        # read and extract features from training/validation data, serialize to disk
        self._extract_and_write(train)
        self._extract_and_write(valid, test=True)

        if feats_only:
            return

        # compute steps per epoch/checkpoint and early stopping steps
        max_steps, patience, checkpoint_steps, steps_per_epoch = self._compute_steps(train, valid)
        self._training_config.max_steps = max_steps  # update config value for learning rate calculation
        self._training_config.steps_per_epoch = steps_per_epoch

        # train and evaluate using Estimator API
        estimator = self._init_estimator(checkpoint_steps)

        train_spec = self._train_spec(train, max_steps, self._training_hooks(estimator, patience, checkpoint_steps))
        eval_spec = self._eval_spec(valid)

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

        for test_set in test_paths:
            logging.info('Evaluating on %s' % test_set)
            output_path = os.path.join(self._job_dir, os.path.basename(test_set) + '.eval')

            evaluation_fn = get_evaluator(self._training_config.heads,
                                          self._feature_extractor, output_path, self._eval_script_path)

            # extract instances from test file at given path--this is a generator, so wrap in a list
            # predict from instances instead of raw text, so use .predict_inputs, don't format since we need the raw predictions
            processed_examples = predictor.predict_parsed(self._extract_raw(test_set, True), formatted=False)
            # call evaluation function on predictions
            evaluation_fn(self._extract_raw(test_set, True), processed_examples)

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
            logging.info('Writing predictions on %s to %s' % (test_set, prediction_path))
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
        logging.info("Loading predictor from saved model at %s" % latest)
        return from_saved_model(latest)

    def _init_feature_extractor(self, train_path: str = None):
        self._feature_extractor = get_feature_extractor(self._training_config.features, self._training_config.heads)
        logging.info("Checking for pre-existing vocabulary at vocabulary at %s", self._vocab_path)
        if self._feature_extractor.read_vocab(self._vocab_path):
            logging.info("Loaded pre-existing vocabulary at %s", self._vocab_path)
        elif train_path:
            logging.info("No valid pre-existing vocabulary found at %s "
                         "(this is normal when not loading from an existing model)", self._vocab_path)
            self._train_vocab(train_path)
        else:
            raise ValueError('No feature vocabulary available at %s and unable to train new vocabulary' % self._vocab_path)

    def _extract_raw(self, path: str, test: bool = False):
        reader = get_reader(self._training_config.reader, self._training_config, is_test=test)
        raw_instances = reader.read_file(path)

        if not raw_instances:
            raise ValueError("No examples provided at path given by '%s'" % path)
        return raw_instances

    def _train_vocab(self, train_path: str):
        logging.info("Creating new vocabulary using training data at %s", train_path)
        self._feature_extractor.initialize(self._resources)
        self._feature_extractor.train(self._extract_raw(train_path))
        logging.info("Writing new feature/label vocabulary to %s", self._vocab_path)
        self._feature_extractor.write_vocab(self._vocab_path, resources=self._resources, prune=True)

    def _extract_features(self, path: str, test: bool = False):
        logging.info("Extracting features from %s", path)
        examples = self._feature_extractor.extract_all(self._extract_raw(path, test))
        return examples

    def _extract_and_write(self, path: str, test: bool = False):
        output_path = self._data_path_fn(path)
        if gfile.exists(output_path):
            logging.info("Using pre-existing features for %s from %s", path, output_path)
            return
        examples = self._extract_features(path, test)
        write_features(examples, output_path)

    def _compute_steps(self, train, valid):
        train_count = sum(1 for _ in tf.python_io.tf_record_iterator(self._data_path_fn(train)))
        valid_count = sum(1 for _ in tf.python_io.tf_record_iterator(self._data_path_fn(valid)))

        steps_per_epoch = train_count // self._training_config.batch_size
        if not self._training_config.max_epochs:
            if not self._training_config.max_steps:
                self._training_config.max_epochs = 100
            else:
                self._training_config.max_epochs = self._training_config.max_steps // steps_per_epoch
        if not self._training_config.patience_epochs:
            self._training_config.patience_epochs = 5
        if not self._training_config.checkpoint_epochs:
            self._training_config.checkpoint_epochs = 1

        max_steps = self._training_config.max_epochs * steps_per_epoch
        patience = self._training_config.patience_epochs * steps_per_epoch
        checkpoint_steps = self._training_config.checkpoint_epochs * steps_per_epoch

        logging.info('Training on %d instances at %s, validating on %d instances at %s'
                     % (train_count, train, valid_count, valid))
        logging.info('Training for a maximum of %d epoch(s) (%d steps w/ batch_size=%d)'
                     % (self._training_config.max_epochs, max_steps, self._training_config.batch_size))
        if patience < max_steps:
            logging.info('Early stopping after %d epoch(s) (%d steps) with no improvement on validation set'
                         % (self._training_config.patience_epochs, patience))
        logging.info('Evaluating every %d steps, %d epoch(s)' % (checkpoint_steps, self._training_config.checkpoint_epochs))

        return max_steps, patience, checkpoint_steps, steps_per_epoch

    def _init_estimator(self, checkpoint_steps):
        return tfe.estimator.Estimator(model_fn=self._model_fn,
                                       model_dir=self._model_path,
                                       config=RunConfig(
                                           log_step_count_steps=checkpoint_steps // 10,
                                           save_summary_steps=checkpoint_steps,
                                           keep_checkpoint_max=self._training_config.keep_checkpoints,
                                           save_checkpoints_steps=checkpoint_steps),
                                       params=self._params(test=False))

    def _training_hooks(self, estimator, patience, checkpoint_steps):
        early_stopping = tf.estimator.experimental.stop_if_no_increase_hook(
            estimator,
            metric_name=self._training_config.metric,
            max_steps_without_increase=patience,
            min_steps=checkpoint_steps,
            run_every_secs=None,
            # reduce how often we check if we should stop to when it makes sense--when we evaluate
            run_every_steps=checkpoint_steps,
        )

        hooks = [early_stopping]

        if self._debug:
            hooks.append(tf.train.ProfilerHook(save_steps=10,
                                               output_dir=self._job_dir,
                                               show_memory=True))
        return hooks

    def _train_spec(self, train, max_steps, hooks):
        return tfe.estimator.TrainSpec(self._input_fn(train, True),
                                       max_steps=max_steps,
                                       hooks=hooks)

    def _eval_spec(self, valid):
        exporter = BesterExporter(serving_input_receiver_fn=self._serving_input_fn,
                                  compare_fn=metric_compare_fn(self._training_config.metric),
                                  exports_to_keep=self._training_config.exports_to_keep)

        return tfe.estimator.EvalSpec(self._input_fn(valid, False),
                                      steps=None,  # evaluate on full validation set
                                      exporters=[exporter],
                                      throttle_secs=0)

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
                write_json(self._training_config, os.path.join(self._job_dir, constants.CONFIG_PATH))

        return lambda: make_dataset(self._feature_extractor,
                                    paths=self._data_path_fn(dataset),
                                    batch_size=self._training_config.batch_size,
                                    evaluate=not train,
                                    bucket_sizes=bucket_sizes,
                                    buffer_size=self._training_config.buffer_size,
                                    batch_buffer_size=self._training_config.batch_buffer_size,
                                    caching=self._training_config.dataset_caching)


TRAINING_MODES = {'train', 'predict', 'test', 'itl', 'features-only'}


def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='save', type=str, required=True,
                        help='models/checkpoints/vocabularies save path')
    parser.add_argument('--config', type=str, help='training configuration JSON')
    parser.add_argument('--resources', type=str, help='shared resources directory (such as for word embeddings)')
    parser.add_argument('--train', type=str, help='training data path')
    parser.add_argument('--valid', type=str, help='validation/development data path')
    parser.add_argument('--test', type=str, nargs="*", help='test data paths, space-separated')
    parser.add_argument('--mode', type=str, default="train", help='(optional) training command, "train" by default',
                        choices=list(TRAINING_MODES))
    parser.add_argument('--script', type=str, help='(optional) evaluation script path')
    parser.add_argument('--debug', action='store_true', help='Activate profiling/debug mode')
    parser.set_defaults(debug=False)

    parser.add_argument('--config_overrides', nargs="*", type=str,
                        help='space-separated list of keys and corresponding JSON configuration files')
    parser.add_argument('--param_overrides', type=str,
                        help='comma-separated list of parameters with subfields separated by periods, e.g. "optimizer.lr=0.1"')
    return parser


def _validate_and_parse_args():
    parser = default_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def cli():
    opts = _validate_and_parse_args()

    # write configuration file to model path, applying any updates/overrides if this is the first time training
    config_path = os.path.join(opts.save, constants.CONFIG_PATH)
    if not gfile.exists(config_path):
        if not opts.config:
            raise AssertionError('trainer configuration is required when training for the first time')
        config = read_config(opts.config, opts.config_overrides, opts.param_overrides)
        gfile.makedirs(opts.save)
        write_json(config, config_path)

    trainer = Trainer(save_dir_path=opts.save,
                      resources_dir_path=opts.resources,
                      script_file_path=opts.script,
                      debug=opts.debug)

    mode = opts.mode

    set_up_logging(os.path.join(opts.save, '{}.log'.format(mode)))

    test_paths = None
    if opts.test:
        test_paths = []
        for test_path in opts.test:
            test_paths.extend([t for t in test_path.split(',') if t.strip()])
        test_paths = sorted(test_paths)

    if mode not in TRAINING_MODES:
        raise ValueError("Unexpected mode type: {}".format(mode))

    if mode == 'train' and not opts.train and test_paths:
        logging.info('No training set provided, defaulting to test mode for %s' % opts.test)
        mode = 'test'

    if mode == "train":
        if not opts.train:
            logging.warn('train mode was selected, but no training set path was provided (use "--train path/to/train")')
            return
        elif not opts.valid:
            logging.warn('train mode was selected, but no validation set path was provided (use "--valid path/to/valid")')
            return

        trainer.train(opts.train, opts.valid)

        if test_paths:
            trainer.eval(test_paths)

    elif mode == "predict":
        if not test_paths:
            logging.warn('predict mode was selected, but no test paths were provided (use "--test path1,path2")')
            return

        trainer.predict(test_paths)

    elif mode == "test":
        if not test_paths:
            logging.warn('test mode was selected, but no test paths were provided (use "--test path1,path2")')
            return

        trainer.eval(test_paths)

    elif mode == "itl":
        trainer.itl()

    elif mode == "features-only":
        trainer.train(opts.train, opts.valid, True)


if __name__ == '__main__':
    cli()
