import argparse
import re
import subprocess
import tempfile

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from tensorflow.contrib.layers import optimize_loss
from tensorflow.contrib.learn import Experiment, RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from tfnlp.common.config import get_feature_extractor
from tfnlp.common.constants import LABEL_KEY, LENGTH_KEY, PREDICT_KEY, WORD_KEY
from tfnlp.common.utils import read_json
from tfnlp.datasets import make_dataset
from tfnlp.feature import write_features
from tfnlp.layers.layers import input_layer
from tfnlp.readers import conll_2003_reader

tf.logging.set_verbosity(tf.logging.INFO)

TRANSITIONS = "transitions"
SCORES_KEY = "scores"


def decode(transition_matrix, logits, sequence_length):
    return viterbi_decode(score=logits[:sequence_length], transition_params=transition_matrix)[0]


class ConllEvalHook(session_run_hook.SessionRunHook):
    def __init__(self, script_path, predict_tensor, gold_tensor, length_tensor, vocab, scores_tensor=None, transitions=None):
        self._script_path = script_path
        self._predict_tensor = predict_tensor
        self._scores_tensor = scores_tensor
        self._gold_tensor = gold_tensor
        self._transitions = transitions
        self._length_tensor = length_tensor
        self._vocab = vocab

        self._predictions = None
        self._gold = None

    def begin(self):
        self._predictions = []
        self._gold = []

    def before_run(self, run_context):  # pylint: disable=unused-argument
        fetches = {LABEL_KEY: self._gold_tensor,
                   PREDICT_KEY: self._predict_tensor,
                   LENGTH_KEY: self._length_tensor}
        if self._scores_tensor is not None:
            fetches[SCORES_KEY] = self._scores_tensor
            fetches[TRANSITIONS] = self._transitions
        return SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        transition_matrix = run_values.results[TRANSITIONS]
        if self._scores_tensor is not None:
            for gold, scores, seq_len in zip(run_values.results[LABEL_KEY],
                                             run_values.results[SCORES_KEY],
                                             run_values.results[LENGTH_KEY]):
                self._gold.append([self._vocab.index_to_feat(val) for val in gold][:seq_len])
                self._predictions.append([self._vocab.index_to_feat(val) for val in decode(transition_matrix, scores, seq_len)])

        else:
            for gold, predictions, seq_len in zip(run_values.results[LABEL_KEY],
                                                  run_values.results[PREDICT_KEY],
                                                  run_values.results[LENGTH_KEY]):
                self._gold.append([self._vocab.index_to_feat(val) for val in gold][:seq_len])
                self._predictions.append([self._vocab.index_to_feat(val) for val in predictions][:seq_len])

    def end(self, session):
        evaluate(self._gold, self._predictions, self._script_path)


def test(args):
    # load and process data -----------------------------------------------------------------------------------------------------
    feat_config = read_json("data/feats.json")
    net_config = read_json("data/network.json")

    if args.mode == "train":
        extractor = get_feature_extractor(feat_config)
        extractor.train()
        examples = []
        print("Reading/writing features...")
        for instance in conll_2003_reader().read_file(args.train):
            examples.append(extractor.extract(instance))
        write_features(examples, args.train + ".tfr")
        extractor.test()
        examples = []
        for instance in conll_2003_reader().read_file(args.valid):
            examples.append(extractor.extract(instance))
        write_features(examples, args.valid + ".tfr")
        extractor.write_vocab(args.vocab)

        print("Beginning training...")
        estimator = tf.estimator.Estimator(model_fn=model_func, model_dir=args.save,
                                           config=RunConfig(save_checkpoints_steps=2000),
                                           params=HParams(extractor=extractor, config=net_config))
        experiment = Experiment(estimator=estimator,
                                train_input_fn=lambda: make_dataset(extractor, paths=args.train + ".tfr", batch_size=10),
                                eval_input_fn=lambda: make_dataset(extractor, paths=args.valid + ".tfr", batch_size=10,
                                                                   evaluate=True),
                                eval_steps=None)
        experiment.train_and_evaluate()
        print("Done training")

    else:
        extractor = get_feature_extractor(feat_config)
        extractor.read_vocab(args.vocab)
        examples = []
        for instance in conll_2003_reader().read_file(args.valid):
            examples.append(extractor.extract(instance))
        write_features(examples, args.valid + ".tfr")
        estimator = tf.estimator.Estimator(model_fn=model_func, model_dir=args.save,
                                           config=RunConfig(save_checkpoints_secs=60),
                                           params=HParams(extractor=extractor, config=net_config))
        labels = extractor.target(LABEL_KEY)
        predicted = []
        gold = []
        for prediction in estimator.predict(lambda: make_dataset(extractor, paths=args.valid + ".tfr", evaluate=True)):
            predicted.append([labels.index_to_feat(val) for val in prediction[PREDICT_KEY]][:prediction[LENGTH_KEY]])
            gold.append([labels.index_to_feat(val) for val in prediction[LABEL_KEY]][:prediction[LENGTH_KEY]])
        evaluate(gold, predicted)


def model_func(features, mode, params):
    feats = input_layer(features, params.extractor.features, mode == tf.estimator.ModeKeys.TRAIN)
    targets = tf.identity(features[LABEL_KEY], name=LABEL_KEY)

    def cell():
        _cell = tf.nn.rnn_cell.LSTMCell(params.config.state_size)
        keep_prob = (1.0 - params.config.dropout) if mode == tf.estimator.ModeKeys.TRAIN else 1.0
        return DropoutWrapper(_cell, variational_recurrent=True, dtype=tf.float32,
                              output_keep_prob=keep_prob, state_keep_prob=keep_prob)

    outputs, final = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell(),
                                                     cell_bw=cell(),
                                                     inputs=feats,
                                                     sequence_length=features[LENGTH_KEY],
                                                     dtype=tf.float32)
    outputs = tf.concat(values=outputs, axis=-1)
    time_steps = tf.shape(outputs)[1]
    rnn_outputs = tf.reshape(outputs, [-1, params.config.state_size * 2], name="flatten_rnn_outputs_for_linear_projection")

    num_labels = params.extractor.targets[LABEL_KEY].vocab_size()
    initializer = None
    logits = tf.layers.dense(rnn_outputs, num_labels, kernel_initializer=initializer)
    logits = tf.reshape(logits, [-1, time_steps, num_labels], name="unflatten_logits")

    transition_matrix = tf.get_variable("transitions", [num_labels, num_labels])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.argmax(logits, 2, name=PREDICT_KEY)
        return tf.estimator.EstimatorSpec(mode=mode, predictions={PREDICT_KEY: predictions,
                                                                  LENGTH_KEY: features[LENGTH_KEY],
                                                                  WORD_KEY: features[WORD_KEY],
                                                                  SCORES_KEY: logits,
                                                                  LABEL_KEY: targets,
                                                                  TRANSITIONS: transition_matrix
                                                                  })

    if params.config.crf:
        log_likelihood, _ = crf_log_likelihood(logits, targets, sequence_lengths=features[LENGTH_KEY],
                                               transition_params=transition_matrix)
        losses = -log_likelihood
    else:
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        mask = tf.sequence_mask(features[LENGTH_KEY], name="padding_mask")
        losses = tf.boolean_mask(losses, mask, name="mask_padding_from_loss")

    loss = tf.reduce_mean(losses)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, epsilon=1e-6)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        optimize_loss(loss=loss, global_step=tf.train.get_global_step(), learning_rate=1.0,
                      optimizer=optimizer, clip_gradients=1.0)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(logits, 2, name=PREDICT_KEY)
        eval_metrics_ops = {
            "accuracy": tf.metrics.accuracy(labels=targets, predictions=predictions),
            "precision": tf.metrics.precision(labels=targets, predictions=predictions),
            "recall": tf.metrics.recall(labels=targets, predictions=predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops,
                                          evaluation_hooks=[ConllEvalHook(script_path="data/scripts/conlleval.pl",
                                                                          gold_tensor=targets,
                                                                          predict_tensor=predictions,
                                                                          scores_tensor=logits,
                                                                          length_tensor=features[LENGTH_KEY],
                                                                          transitions=transition_matrix,
                                                                          vocab=params.extractor.targets[LABEL_KEY])])

    raise AssertionError


def evaluate(gold_batches, predicted_batches, script_path="data/scripts/conlleval.pl"):
    with tempfile.NamedTemporaryFile(mode='wt') as temp:
        for gold_seq, predicted_seq in zip(gold_batches, predicted_batches):
            for label, prediction in zip(gold_seq, predicted_seq):
                temp.write("_ {} {}\n".format(label, prediction))
            temp.write("\n")  # sentence break
        temp.flush()
        temp.seek(0)
        result = subprocess.check_output(["perl", script_path], stdin=temp, universal_newlines=True)
        print(result)
        return float(re.split('\s+', re.split('\n', result)[1].strip())[7])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, required=True, help='Directory where models/checkpoints are saved.')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary base directory.')
    parser.add_argument('--train', type=str, help='File containing training data.')
    parser.add_argument('--valid', type=str, help='File containing validation data.')
    parser.add_argument('--mode', type=str, default="train", help='Command in [train, predict]')
    test(parser.parse_args())
