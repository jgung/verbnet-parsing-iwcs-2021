import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

from tfnlp.common.config import train_op_from_config
from tfnlp.common.eval import log_trainable_variables
from tfnlp.layers.heads import model_head
from tfnlp.layers.layers import encoder, embedding


def build(features, mode, params):
    config = params.config
    training = mode == tf.estimator.ModeKeys.TRAIN

    encoder_configs = {enc.name: enc for enc in config.encoders}
    inputs = {feat: embedding(features, feat_conf, training) for feat, feat_conf in params.extractor.features.items()}
    heads = {}
    encoders = {}

    def build_head(_head):
        head_encoder = encoders.get(_head.encoder)
        if not head_encoder:
            head_encoder = build_encoder(encoder_configs[_head.encoder])
            encoders[_head.encoder] = head_encoder
        return model_head(_head, head_encoder, features, mode, params)

    def build_encoder(_encoder):
        encoder_features = {}
        for encoder_input in _encoder.inputs:
            if encoder_input in inputs:
                encoder_features[encoder_input] = inputs[encoder_input]
            if encoder_input in encoders:
                encoder_features[encoder_input] = encoders[encoder_input]
            elif encoder_input in heads:
                encoder_features[encoder_input] = heads[encoder_input].predictions
        return encoder(features, encoder_features.values(), mode, _encoder)

    return [build_head(head) for head in config.heads]


def multi_head_model_func(features, mode, params):
    config = params.config

    heads = build(features, mode, params)

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
