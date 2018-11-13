import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow.python.saved_model import signature_constants

import tfnlp.common.constants as constants
from tfnlp.common.config import train_op_from_config
from tfnlp.common.eval import SequenceEvalHook, SrlEvalHook, log_trainable_variables
from tfnlp.common.metrics import tagger_metrics
from tfnlp.layers.layers import encoder, input_layer


def tagger_model_func(features, mode, params):
    config = params.config

    # reduce all features to a single input vector (per token per batch)
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)
    # produce a context-dependent vector representation of token (such as from hidden states of a biLSTM)
    encoder_outputs, encoder_dim, _ = encoder(features, inputs, mode, config)

    time_steps = tf.shape(encoder_outputs)[1]  # max sequence length (number of tokens) for this batch
    # flatten encoder outputs to a (batch_size * time_steps x encoder_dim) Tensor for batch matrix multiplication
    encoder_outputs = tf.reshape(encoder_outputs, [-1, encoder_dim], name="flatten")

    heads = params.extractor.targets
    transitions = {}
    logits = {}

    for target_key, target in heads.items():
        with tf.variable_scope(_append_label("inference_layer", target_key)):
            num_labels = target.vocab_size()
            initializer = tf.zeros_initializer if config.zero_init else tf.random_normal_initializer(stddev=0.01)
            _dense = tf.layers.dense(encoder_outputs, num_labels, kernel_initializer=initializer)
            # batch multiplication complete, convert back to a (batch_size x time_steps x num_labels) Tensor
            logits[target_key] = tf.reshape(_dense, [-1, time_steps, num_labels], name="unflatten")

            if config.crf:
                # explicitly train a transition matrix
                transitions[target_key] = tf.get_variable("transitions", [num_labels, num_labels])
            else:
                # use constrained decoding based on IOB labels
                transitions[target_key] = tf.get_variable("transitions", [num_labels, num_labels],
                                                          trainable=False, initializer=_create_transition_matrix(target))

    targets = {}
    scores = {}
    loss = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        def _loss(_target_key):
            with tf.variable_scope("%s_loss" % _target_key):
                targets[_target_key] = features[_target_key]

                if config.crf:
                    _losses = -crf_log_likelihood(logits[_target_key], targets[_target_key],
                                                  sequence_lengths=tf.cast(features[constants.LENGTH_KEY], tf.int32),
                                                  transition_params=transitions[_target_key])[0]
                else:
                    _losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[_target_key],
                                                                             labels=targets[_target_key])
                    _mask = tf.sequence_mask(features[constants.LENGTH_KEY], name="padding_mask")
                    _losses = tf.boolean_mask(_losses, _mask, name="mask_padding_from_loss")
                return tf.reduce_mean(_losses)  # just average over batch/token-specific losses

        # compute loss for each target
        losses = [_loss(target_key) for target_key in heads.keys()]
        # just compute mean over losses (possibly consider a more sophisticated strategy?)
        loss = losses[0] if len(losses) == 0 else tf.reduce_mean(tf.stack(losses))

        if config.type == constants.SRL_KEY:
            # used to score official CoNLL script scores computed off-graph in eval hooks in the graph
            scores = _get_score_tensors(heads.keys())

    if mode == tf.estimator.ModeKeys.TRAIN:
        log_trainable_variables()
        train_op = train_op_from_config(config, loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # end of TRAIN configuration -----------------------------------------------------------------------------------------------

    predictions = {}
    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        for target_key, target in heads.items():
            predictions[target_key] = crf.crf_decode(logits[target_key],
                                                     transitions[target_key],
                                                     tf.cast(features[constants.LENGTH_KEY], tf.int32))[0]

    # EVAL-only configuration --------------------------------------------------------------------------------------------------
    eval_metric_ops = {}
    evaluation_hooks = []
    if mode == tf.estimator.ModeKeys.EVAL:
        for target_key, target in heads.items():
            _predictions = predictions[target_key]
            _labels = targets[target_key]
            predictions_key = _append_label(constants.PREDICT_KEY, target_key)
            labels_key = _append_label(constants.LABEL_KEY, target_key)

            eval_tensors = {  # tensors necessary for evaluation hooks (such as sequence length)
                constants.LENGTH_KEY: features[constants.LENGTH_KEY],
                constants.SENTENCE_INDEX: features[constants.SENTENCE_INDEX],
                labels_key: _labels,
                predictions_key: _predictions,
            }

            if config.type == constants.SRL_KEY:
                eval_tensors[constants.MARKER_KEY] = features[constants.MARKER_KEY]

                # https://github.com/tensorflow/tensorflow/issues/20418 -- metrics don't accept variables, so we create a tensor
                srl_score = tf.identity(scores[target_key])
                srl_key = _append_label(constants.SRL_METRIC_KEY, target_key)
                eval_metric_ops[srl_key] = (srl_score, srl_score)

                eval_placeholder = tf.placeholder(dtype=tf.float32, name='update_%s' % srl_key)
                evaluation_hooks.append(SrlEvalHook(
                    tensors=eval_tensors,
                    vocab=target,
                    label_key=labels_key,
                    predict_key=predictions_key,
                    eval_tensor=srl_score,
                    eval_update=tf.assign(scores[target_key], eval_placeholder),
                    eval_placeholder=eval_placeholder,
                    output_confusions=params.verbose_eval)
                )
            else:
                _tagger_metrics = tagger_metrics(predictions=tf.cast(_predictions, dtype=tf.int64), labels=_labels,
                                                 ns=None if target_key == constants.LABEL_KEY else target_key)
                eval_metric_ops.update(_tagger_metrics)
                acc = _append_label(constants.ACCURACY_METRIC_KEY, target_key)
                eval_metric_ops[acc] = tf.metrics.accuracy(labels=_labels, predictions=_predictions, name=acc)
                evaluation_hooks.append(SequenceEvalHook(
                    tensors=eval_tensors,
                    vocab=target,
                    label_key=labels_key,
                    predict_key=predictions_key,
                    script_path=params.script_path,
                    output_file=params.output)
                )

    # PREDICT-only configuration ----------------------------------------------------------------------------------------------
    export_outputs = {}
    if mode == tf.estimator.ModeKeys.PREDICT:
        for target_key, target in heads.items():
            index_to_label = index_to_string_table_from_file(vocabulary_file=os.path.join(params.vocab_path, target_key),
                                                             default_value=target.unknown_word)
            predictions[target_key] = index_to_label.lookup(tf.cast(predictions[target_key], dtype=tf.int64))
            # if there are multiple targets, there must be at least one 'gold' target to use as a default
            export_key = _append_label(constants.PREDICT_KEY, target_key, signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            export_outputs[export_key] = PredictOutput(predictions[target_key])

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
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


def _append_label(metric_key, target_key, default_val=None):
    """
    Append a label to a pre-existing key. If the label is the default 'gold' label, don't modify the pre-existing key.
    :param metric_key: pre-existing key
    :param target_key: label
    :param default_val: value to use conditionally if the label is a default value
    :return:
    """
    if target_key == constants.LABEL_KEY:
        if default_val:
            return default_val
        return metric_key
    return '%s-%s' % (metric_key, target_key)


def _get_score_tensors(_target_keys):
    score_tensors = {}
    for target_key in _target_keys:
        score_tensors[target_key] = tf.Variable(0, name=_append_label(constants.SRL_METRIC_KEY, target_key), dtype=tf.float32,
                                                trainable=False)
    return score_tensors
