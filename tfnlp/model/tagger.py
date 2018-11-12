import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow.python.saved_model import signature_constants

import tfnlp.common.constants as constants
from tfnlp.common.config import get_gradient_clip, get_optimizer
from tfnlp.common.eval import SequenceEvalHook, SrlEvalHook, log_trainable_variables
from tfnlp.common.metrics import tagger_metrics
from tfnlp.layers.layers import encoder, input_layer


def tagger_model_func(features, mode, params):
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)
    outputs, output_size, _ = encoder(features, inputs, mode, params.config)

    outputs = tf.concat(values=outputs, axis=-1)
    time_steps = tf.shape(outputs)[1]
    rnn_outputs = tf.reshape(outputs, [-1, output_size], name="flatten_rnn_outputs_for_linear_projection")

    all_targets = params.extractor.targets
    transitions = {}
    logits = {}

    for target_key, target in all_targets.items():
        with tf.variable_scope("inference_layer_%s" % target_key):
            num_labels = target.vocab_size()
            if params.config.zero_init:
                initializer = tf.zeros_initializer
            else:
                initializer = tf.random_normal_initializer(stddev=0.01)

            logits[target_key] = tf.layers.dense(rnn_outputs, num_labels, kernel_initializer=initializer,
                                                 name="softmax_projection")
            logits[target_key] = tf.reshape(logits[target_key], [-1, time_steps, num_labels], name="unflatten_logits")

            if params.config.crf:
                transitions[target_key] = tf.get_variable("transitions", [num_labels, num_labels])
            else:
                transitions[target_key] = tf.get_variable("transitions", [num_labels, num_labels],
                                                          trainable=False, initializer=_create_transition_matrix(target))

    targets = {}
    loss = None
    train_op = None
    scores = {}

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        log_trainable_variables()

        def _loss(_target_key):
            targets[_target_key] = tf.identity(features[_target_key], name=_target_key)

            if params.config.crf:
                log_likelihood, _ = crf_log_likelihood(logits[_target_key], targets[_target_key],
                                                       sequence_lengths=tf.cast(features[constants.LENGTH_KEY], tf.int32),
                                                       transition_params=transitions[_target_key])
                _losses = -log_likelihood
            else:
                _losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[_target_key], labels=targets[_target_key])
                mask = tf.sequence_mask(features[constants.LENGTH_KEY], name="padding_mask_%s" % target_key)
                _losses = tf.boolean_mask(_losses, mask, name="mask_padding_from_loss_%s" % target_key)
            return tf.reduce_mean(_losses)

        if len(all_targets) == 1:
            loss = _loss(next(iter(all_targets.keys())))
        else:
            loss = tf.reduce_mean(tf.stack([_loss(target_key) for target_key in all_targets.keys()]))

        if params.config.type == constants.SRL_KEY:
            for target_key in all_targets.keys():
                scores[target_key] = tf.Variable(0, name=_metric_key(constants.SRL_METRIC_KEY, target_key), dtype=tf.float32,
                                                 trainable=False)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params.config)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=get_gradient_clip(params.config))
        train_op = optimizer.apply_gradients(zip(gradients, parameters), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    predictions = {}
    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        for target_key, target in all_targets.items():
            _predictions, _ = crf.crf_decode(logits[target_key], transitions[target_key],
                                             tf.cast(features[constants.LENGTH_KEY], tf.int32))
            predictions[target_key] = _predictions

    eval_metric_ops = {}
    evaluation_hooks = []
    if mode == tf.estimator.ModeKeys.EVAL:
        for target_key, target in all_targets.items():
            _predictions = predictions[target_key]
            _labels = targets[target_key]
            if params.config.type == constants.SRL_KEY:
                f1_score = tf.identity(scores[target_key])
                eval_metric_ops[_metric_key(constants.SRL_METRIC_KEY, target_key)] = f1_score, f1_score
                eval_placeholder = tf.placeholder(dtype=tf.float32)
                evaluation_hooks.append(SrlEvalHook(
                    tensors={
                        _metric_key(constants.LABEL_KEY, target_key): _labels,
                        _metric_key(constants.PREDICT_KEY, target_key): _predictions,
                        constants.LENGTH_KEY: features[constants.LENGTH_KEY],
                        constants.MARKER_KEY: features[constants.MARKER_KEY],
                        constants.SENTENCE_INDEX: features[constants.SENTENCE_INDEX]
                    },
                    vocab=target,
                    eval_tensor=f1_score, eval_update=tf.assign(scores[target_key], eval_placeholder),
                    eval_placeholder=eval_placeholder,
                    label_key=_metric_key(constants.LABEL_KEY, target_key),
                    predict_key=_metric_key(constants.PREDICT_KEY, target_key),
                    output_confusions=params.verbose_eval,
                ))
            else:
                eval_metric_ops.update(tagger_metrics(predictions=tf.cast(_predictions, dtype=tf.int64), labels=_labels,
                                                      ns=target_key if target_key != constants.LABEL_KEY else None))
                eval_metric_ops[_metric_key(constants.ACCURACY_METRIC_KEY, target_key)] = tf.metrics.accuracy(
                    labels=_labels, predictions=_predictions, name=_metric_key(constants.ACCURACY_METRIC_KEY, target_key))
                evaluation_hooks.append(SequenceEvalHook(
                    script_path=params.script_path,
                    tensors={
                        _metric_key(constants.LABEL_KEY, target_key): _labels,
                        _metric_key(constants.PREDICT_KEY, target_key): _predictions,
                        constants.LENGTH_KEY: features[constants.LENGTH_KEY],
                        constants.SENTENCE_INDEX: features[constants.SENTENCE_INDEX]
                    },
                    vocab=target,
                    output_file=params.output,
                    label_key=_metric_key(constants.LABEL_KEY, target_key),
                    predict_key=_metric_key(constants.PREDICT_KEY, target_key)
                ))

    export_outputs = {}
    if mode == tf.estimator.ModeKeys.PREDICT:
        for target_key, target in all_targets.items():
            index_to_label = index_to_string_table_from_file(vocabulary_file=os.path.join(params.vocab_path, target_key),
                                                             default_value=target.unknown_word)
            _string_predictions = index_to_label.lookup(tf.cast(predictions[target_key], dtype=tf.int64))
            export_key = _metric_key(constants.PREDICT_KEY, target_key, signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            export_outputs[export_key] = PredictOutput(_string_predictions)
            predictions[target_key] = _string_predictions

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      evaluation_hooks=evaluation_hooks)


def _create_transition_matrix(labels):
    """
    Return a numpy matrix to enforce valid transitions for IOB-style tagging problems.
    :param labels: label feature extractor
    """
    labels = [labels.index_to_feat(i) for i in range(len(labels.indices))]
    num_tags = len(labels)
    transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)
    for i, prev_label in enumerate(labels):
        for j, curr_label in enumerate(labels):
            if i != j and curr_label[:2] == 'I-' and not prev_label == 'B' + curr_label[1:]:
                transition_params[i, j] = np.NINF
    return tf.initializers.constant(transition_params)


def _metric_key(metric_key, target_key, default_val=None):
    if target_key == constants.LABEL_KEY:
        if default_val:
            return default_val
        return metric_key
    return '%s-%s' % (metric_key, target_key)
