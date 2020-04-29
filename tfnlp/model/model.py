from collections import OrderedDict

import tensorflow as tf
import tensorflow_estimator as tfe
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.saved_model import signature_constants

from tfnlp.common import constants
from tfnlp.common.config import train_op_from_config
from tfnlp.common.eval import log_trainable_variables
from tfnlp.common.training_utils import assign_ema_weights
from tfnlp.layers.heads import ClassifierHead, TaggerHead, TokenClassifierHead, BiaffineSrlHead
from tfnlp.layers.layers import encoder, embedding, get_embedding_input
from tfnlp.model.parser import ParserHead


def build(features, mode, params):
    config = params.config
    training = mode == tfe.estimator.ModeKeys.TRAIN

    encoder_configs = {enc.name: enc for enc in config.encoders}
    head_configs = {head.name: head for head in config.heads}

    inputs = {feat: embedding(features, feat_conf, training) for feat, feat_conf in params.extractor.features.items()}
    heads = {}
    encoders = {}

    def get_head(_head_config):
        if _head_config.name in heads:
            return heads[_head_config.name]

        head_encoder = get_encoder(encoder_configs[_head_config.encoder])
        head = model_head(_head_config, head_encoder, features, mode, params)

        heads[_head_config.name] = head
        return head

    def get_encoder(_encoder_config):
        if _encoder_config.name in encoders:
            return encoders[_encoder_config.name]

        # build encoder recursively
        encoder_features = OrderedDict()
        for encoder_input in _encoder_config.inputs:
            if encoder_input in inputs:
                # input from embedding/feature input
                encoder_features[encoder_input] = inputs[encoder_input]
            elif encoder_input in encoder_configs:
                # input from another encoder
                encoder_config = encoder_configs[encoder_input]
                encoder_features[encoder_input] = get_encoder(encoder_config)
            elif encoder_input in head_configs:
                # input from a model head
                head_config = head_configs[encoder_input]
                head = get_head(head_config)

                weights = None
                if training and head_config.teacher_forcing:
                    predictions = features[head.name]
                else:
                    if head_config.weighted_embedding:
                        weights = head.scores
                    predictions = head.predictions

                encoder_features[encoder_input] = get_embedding_input(predictions, head.extractor, training, weights=weights)
            else:
                raise ValueError('Missing encoder input: %s' % encoder_input)

        result = encoder(features, list(encoder_features.values()), mode, _encoder_config)

        encoders[_encoder_config.name] = result
        return result

    return [get_head(head) for head in config.heads]


def multi_head_model_fn(features, mode, params):
    config = params.config

    heads = build(features, mode, params)

    # combine losses
    loss = None
    if mode in [tfe.estimator.ModeKeys.TRAIN, tfe.estimator.ModeKeys.EVAL]:
        # compute loss for each target
        losses = [head.weight * head.loss for head in heads]
        # just compute mean over losses (possibly consider a more sophisticated strategy?)
        loss = losses[0] if len(losses) == 1 else tf.reduce_mean(tf.stack(losses))

    dependencies = []
    # optionally setup exponential moving average of parameters
    if config.ema_decay > 0:
        dependencies.append(_exponential_moving_average_op(mode, config.ema_decay))
    else:
        dependencies.append(tf.no_op())

    with tf.control_dependencies(dependencies):
        # make sure we have properly assigned averaged variables if we are evaluating

        if mode == tfe.estimator.ModeKeys.TRAIN:
            log_trainable_variables()
            train_op = train_op_from_config(config, loss)
            return tfe.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        # EVAL/PREDICT -----------------------------------------------------------------------------------------------------------

        # combine predictions
        predictions = {}
        if mode in [tfe.estimator.ModeKeys.EVAL, tfe.estimator.ModeKeys.PREDICT]:
            for head in heads:
                predictions[head.name] = head.predictions

        # combine evaluation hooks and metrics
        eval_metric_ops = {}
        evaluation_hooks = []
        if mode == tfe.estimator.ModeKeys.EVAL:
            for head in heads:
                eval_metric_ops.update(head.metric_ops)
                evaluation_hooks.extend(head.evaluation_hooks)

        # combine export outputs
        export_outputs = None
        if mode == tfe.estimator.ModeKeys.PREDICT:
            export_outputs = {}
            combined_outputs = {}
            for head in heads:
                export_outputs[head.name] = PredictOutput(head.export_outputs)
                combined_outputs.update(head.export_outputs)
            # combined signature with all relevant outputs
            export_outputs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = PredictOutput(combined_outputs)

        return tfe.estimator.EstimatorSpec(mode=mode,
                                           predictions=predictions,
                                           loss=loss,
                                           eval_metric_ops=eval_metric_ops,
                                           export_outputs=export_outputs,
                                           evaluation_hooks=evaluation_hooks)


def model_head(config, inputs, features, mode, params):
    """
    Initialize a model head from a given configuration.
    :param config: head configuration
    :param inputs: output from encoder (e.g. biLSTM), input to head
    :param features: all model inputs
    :param mode: Estimator mode type (TRAIN, EVAL, or PREDICT)
    :param params: HParams input to Estimator
    :return: initialized model head
    """
    heads = {
        constants.CLASSIFIER_KEY: ClassifierHead,
        constants.TAGGER_KEY: TaggerHead,
        constants.NER_KEY: TaggerHead,
        constants.SRL_KEY: TaggerHead,
        constants.BIAFFINE_SRL_KEY: BiaffineSrlHead,
        constants.TOKEN_CLASSIFIER_KEY: TokenClassifierHead,
        constants.PARSER_KEY: ParserHead
    }
    if config.type not in heads:
        raise AssertionError('Unsupported head type: %s' % config.type)

    head = heads[config.type](inputs=inputs, config=config, features=features, params=params,
                              training=mode == tfe.estimator.ModeKeys.TRAIN)
    if mode == tfe.estimator.ModeKeys.TRAIN:
        head.training()
    elif mode == tfe.estimator.ModeKeys.EVAL:
        head.evaluation()
    elif mode == tfe.estimator.ModeKeys.PREDICT:
        head.prediction()
    return head


def _exponential_moving_average_op(mode, ema_decay):
    ema = tf.train.ExponentialMovingAverage(ema_decay, num_updates=tf.train.get_global_step(), zero_debias=True)
    ema_op = ema.apply(tf.trainable_variables())
    tf.logging.debug("Using EMA for variables: %s" % str([v.name for v in tf.trainable_variables()]))

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    # only use EMA averages when evaluating
    ema_dep = tf.cond(tf.equal(mode, tfe.estimator.ModeKeys.TRAIN),
                      lambda: tf.no_op(),
                      lambda: assign_ema_weights(ema))
    return ema_dep
