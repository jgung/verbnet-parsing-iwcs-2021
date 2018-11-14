import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

from tfnlp.common.config import train_op_from_config
from tfnlp.common.eval import log_trainable_variables
from tfnlp.layers.heads import model_head
from tfnlp.layers.layers import encoder, input_layer


def multi_head_model_func(features, mode, params):
    config = params.config

    # reduce all features to a single input vector (per token per batch)
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)
    # produce a context-dependent vector representation of token (such as from hidden states of a biLSTM)
    encoder_output = encoder(features, inputs, mode, config)

    heads = [model_head(head, encoder_output, features, mode, params) for head in config.heads]

    # combine losses
    loss = None
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        # compute loss for each target
        losses = [head.loss for head in heads]
        # just compute mean over losses (possibly consider a more sophisticated strategy?)
        loss = losses[0] if len(losses) == 0 else tf.reduce_mean(tf.stack(losses))

    if mode == tf.estimator.ModeKeys.TRAIN:
        log_trainable_variables()
        train_op = train_op_from_config(config, loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # EVAL/PREDICT -------------------------------------------------------------------------------------------------------------

    # combine predictions
    predictions = {}
    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        for head in heads:
            predictions[head.name] = head.predictions

    # combine evaluation hooks and metrics
    eval_metric_ops = {}
    evaluation_hooks = []
    if mode == tf.estimator.ModeKeys.EVAL:
        for head in heads:
            eval_metric_ops.update(head.metric_ops)
            evaluation_hooks.extend(head.evaluation_hooks)

    # combine export outputs
    export_outputs = {}
    if mode == tf.estimator.ModeKeys.PREDICT:
        for head in heads:
            export_outputs.update(head.export_outputs)
        if len(export_outputs) > 1:
            export_outputs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs[heads[0].name]

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      evaluation_hooks=evaluation_hooks)
