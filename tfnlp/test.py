import argparse

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers import optimize_loss
from tensorflow.contrib.layers.python.layers.optimizers import adaptive_clipping_fn
from tensorflow.contrib.learn import Experiment, RunConfig
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.export.export import ServingInputReceiver
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper

from tfnlp.common.config import get_feature_extractor
from tfnlp.common.constants import LABEL_KEY, LENGTH_KEY, PREDICT_KEY, WORD_KEY
from tfnlp.common.eval import SequenceEvalHook, make_best_model_export_strategy
from tfnlp.common.utils import read_json
from tfnlp.datasets import make_dataset
from tfnlp.feature import write_features
from tfnlp.layers.layers import input_layer
from tfnlp.readers import conll_2003_reader

tf.logging.set_verbosity(tf.logging.INFO)

TRANSITIONS = "transitions"
SCORES_KEY = "scores"


def test(args):
    # load and process data -----------------------------------------------------------------------------------------------------
    feat_config = read_json("data/feats.json")
    net_config = read_json("data/network.json")

    extractor = get_feature_extractor(feat_config)
    params = HParams(extractor=extractor, config=net_config)
    if args.mode == "train":
        extractor.train()
        examples = []
        print("Reading/writing features...")
        for instance in conll_2003_reader().read_file(args.train):
            examples.append(extractor.extract(instance))
        write_features(examples, args.train + ".tfr")
        examples = []
        for instance in conll_2003_reader().read_file(args.valid):
            examples.append(extractor.extract(instance))
        write_features(examples, args.valid + ".tfr")
        extractor.write_vocab(args.vocab)
        extractor.test()

        print("Beginning training...")
        extractor = get_feature_extractor(feat_config)
        extractor.read_vocab(args.vocab)
        extractor.test()

        def serving_input_receiver_fn():
            """An input_fn that expects a serialized tf.Example."""
            serialized_tf_example = array_ops.placeholder(dtype=dtypes.string,
                                                          shape=None,
                                                          name='input_example_tensor')
            receiver_tensors = {'examples': serialized_tf_example}
            features = extractor.parse(serialized_tf_example, train=False)
            features = {key: tf.expand_dims(val, axis=0) for key, val in features.items()}
            return ServingInputReceiver(features, receiver_tensors)

        estimator = tf.estimator.Estimator(model_fn=model_func, model_dir=args.save,
                                           config=RunConfig(save_checkpoints_steps=2000),
                                           params=params)
        experiment = Experiment(estimator=estimator,
                                train_input_fn=lambda: make_dataset(extractor, paths=args.train + ".tfr", batch_size=10),
                                eval_input_fn=lambda: make_dataset(extractor, paths=args.valid + ".tfr", batch_size=10,
                                                                   evaluate=True),
                                eval_steps=None,
                                export_strategies=[make_best_model_export_strategy(
                                    serving_input_fn=serving_input_receiver_fn,
                                    strip_default_attrs=True)],
                                checkpoint_and_export=True)

        experiment.train_and_evaluate()

    if args == "cli":
        predictor = from_saved_model(args.save)
        while True:
            sentence = input(">>> ").split()
            example = {WORD_KEY: sentence}
            result = predictor({"examples": extractor.extract(example, train=False).SerializeToString()})
            print(' '.join([bstr.decode('utf-8') for bstr in result['output'][0].tolist()]))
    else:
        extractor.read_vocab(args.vocab)
        extractor.test()
        examples = []
        for instance in conll_2003_reader().read_file(args.valid):
            examples.append(extractor.extract(instance))
        write_features(examples, args.valid + ".tfr")
        estimator = tf.estimator.Estimator(model_fn=model_func, model_dir=args.save, params=params)
        estimator.evaluate(lambda: make_dataset(extractor, paths=args.valid + ".tfr", evaluate=True))


def model_func(features, mode, params):
    feats = input_layer(features, params.extractor.features, mode == tf.estimator.ModeKeys.TRAIN)

    def cell():
        _cell = tf.nn.rnn_cell.LSTMCell(params.config.state_size)
        keep_prob = (1.0 - params.config.dropout) if mode == tf.estimator.ModeKeys.TRAIN else 1.0
        return DropoutWrapper(_cell, variational_recurrent=True, dtype=tf.float32,
                              output_keep_prob=keep_prob, state_keep_prob=keep_prob)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell(), cell_bw=cell(), inputs=feats,
                                                 sequence_length=features[LENGTH_KEY], dtype=tf.float32)
    outputs = tf.concat(values=outputs, axis=-1)
    time_steps = tf.shape(outputs)[1]
    rnn_outputs = tf.reshape(outputs, [-1, params.config.state_size * 2], name="flatten_rnn_outputs_for_linear_projection")

    target = params.extractor.targets[LABEL_KEY]
    num_labels = target.vocab_size()
    logits = tf.reshape(tf.layers.dense(rnn_outputs, num_labels), [-1, time_steps, num_labels], name="unflatten_logits")
    transition_matrix = tf.get_variable(TRANSITIONS, [num_labels, num_labels])

    targets = None
    predictions = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None
    evaluation_hooks = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        targets = tf.identity(features[LABEL_KEY], name=LABEL_KEY)

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
        train_op = optimize_loss(loss=loss,
                                 global_step=tf.train.get_global_step(),
                                 learning_rate=1.0,
                                 optimizer=optimizer, clip_gradients=adaptive_clipping_fn())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        predictions, _ = tf.contrib.crf.crf_decode(logits, transition_matrix, features[LENGTH_KEY])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=targets, predictions=predictions),
            "precision": tf.metrics.precision(labels=targets, predictions=predictions),
            "recall": tf.metrics.recall(labels=targets, predictions=predictions)
        }
        evaluation_hooks = [SequenceEvalHook(script_path="data/scripts/conlleval.pl",
                                             gold_tensor=targets,
                                             predict_tensor=predictions,
                                             length_tensor=features[LENGTH_KEY],
                                             vocab=params.extractor.targets[LABEL_KEY])]

    if mode == tf.estimator.ModeKeys.PREDICT:
        index_to_label = index_to_string_table_from_file("data/vocab/gold",
                                                         default_value=target.index_to_feat(target.unknown_index))
        predictions = index_to_label.lookup(tf.cast(predictions, dtype=tf.int64))
        export_outputs = {PREDICT_KEY: PredictOutput(predictions)}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      evaluation_hooks=evaluation_hooks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, required=True, help='Directory where models/checkpoints are saved.')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary base directory.')
    parser.add_argument('--train', type=str, help='File containing training data.')
    parser.add_argument('--valid', type=str, help='File containing validation data.')
    parser.add_argument('--mode', type=str, default="train", help='Command in [train, predict]')
    test(parser.parse_args())
