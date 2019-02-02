import numbers

import tensorflow as tf
from tensorflow.contrib.opt import LazyAdamOptimizer
from tensorflow.python.training.learning_rate_decay import exponential_decay, inverse_time_decay

from tfnlp.common import constants
from tfnlp.common.utils import Params
from tfnlp.optim.lazy_adam import LazyAdamOptimizer as LazyNadamOptimizer
from tfnlp.optim.nadam import NadamOptimizerSparse


class BaseNetworkConfig(Params):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.reader = config.get('reader')

        # training hyperparameters
        self.optimizer = config.get('optimizer')
        self.batch_size = config.get('batch_size')
        if not self.batch_size:
            self.batch_size = 10
            tf.logging.warn("No 'batch_size' parameter provided. Using default value of %d", self.batch_size)
        self.checkpoint_steps = config.get('checkpoint_steps')
        if not self.checkpoint_steps:
            self.checkpoint_steps = self.batch_size * 100
            tf.logging.warn("No 'checkpoint_steps' parameter provided. Using default value of %d", self.checkpoint_steps)
        self.patience = config.get('patience')
        if not self.patience:
            self.patience = self.checkpoint_steps * 5
            tf.logging.warn("No 'patience' parameter provided. Using default value of %d", self.patience)
        self.max_steps = config.get('max_steps')
        if not self.max_steps:
            self.max_steps = self.checkpoint_steps * 100
            tf.logging.warn("No 'max_steps' parameter provided. Using default value of %d", self.max_steps)
        self.exports_to_keep = config.get('exports_to_keep', 1)
        self.keep_checkpoints = config.get('checkpoints_to_keep', 5)

        # feature/input settings
        self.features = config.get('features')
        self.buckets = config.get('buckets')
        self.max_length = config.get('max_length', 100)

        # encoder settings
        self.encoders = [EncoderConfig(val) for val in config.get('encoders', [])]
        if not self.encoders:
            raise ValueError('Must have at least one encoder')

        # head configuration validation
        self.heads = [HeadConfig(head) for head in config.get('heads', [])]
        targets = {}
        for target in self.features.targets:
            if target.name not in {head.name for head in self.heads}:
                tf.logging.warning("Missing head configuration for target '%s'" % target.name)
            targets[target.name] = target
        for head in self.heads:
            if head.name not in targets:
                raise ValueError("Missing feature configuration for target '%s'" % head.name)
        if len(self.heads) == 0:
            raise ValueError("Must have at least one head/target in configuration")

        self.metric = config.get('metric')
        if not self.metric:
            metrics = [append_label(head.metric, head.name) for head in self.heads]
            self.metric = metrics[0]


class EncoderConfig(Params):
    def __init__(self, config):
        super().__init__(**config)
        self.name = config.get('name')
        self.inputs = config.get('inputs', [])
        if not self.inputs:
            raise ValueError("Encoders must have at least one input")
        self.options = config.get('options', {})
        self.encoder_type = config.get('type', constants.ENCODER_BLSTM)
        if self.encoder_type not in constants.ENCODERS:
            raise ValueError("Invalid encoder type: %s" % self.encoder_type)

        self.input_dropout = config.get('input_dropout', 0)

        self.forget_bias = config.get('forget_bias', 1)
        self.encoder_dropout = config.get('encoder_dropout', 0)
        self.encoder_input_dropout = config.get('encoder_input_dropout', 0)
        self.encoder_output_dropout = config.get('encoder_output_dropout', 0)
        self.encoder_layers = config.get('encoder_layers', 1)
        self.state_size = config.get('state_size', 100)

        # transformer encoder settings
        self.num_heads = config.get('num_heads', 8)
        self.head_dim = config.get('head_dim', 25) * self.num_heads
        self.attention_dropout = config.get('attention_dropout', 0.1)
        self.relu_hidden_size = config.get('relu_hidden_size', 0.1)
        self.relu_dropout = config.get('relu_dropout', 0.1)
        self.prepost_dropout = config.get('prepost_dropout', 0.1)


class HeadConfig(Params):
    def __init__(self, config):
        super().__init__(**config)
        self.name = config.get('name', constants.LABEL_KEY)
        self.encoder = config.get('encoder')
        if not self.encoder:
            raise ValueError('Must specify an input "encoder" for this head')
        self.crf = config.get('crf', False)
        self.type = config.get('type', constants.TAGGER_KEY)
        self.zero_init = config.get('zero_init', True)
        self.metric = config.get('metric', constants.OVERALL_KEY)
        self.label_smoothing = config.get('label_smoothing', 0)


def get_network_config(config):
    return BaseNetworkConfig(config)


def get_learning_rate(lr_config, global_step):
    """
    Instantiate a learning rate operation given a configuration with learning rate, name, and parameters.
    :param lr_config:  learning rate configuration
    :param global_step: global step `Tensor`
    :return: learning rate operation
    """
    lr = lr_config.rate
    name = lr_config.name
    if "exponential_decay" == name:
        decay = exponential_decay(lr, global_step, **lr_config.params)
    elif "inverse_time_decay" == name:
        decay = inverse_time_decay(lr, global_step, **lr_config.params)
    elif "vaswani" == name:
        decay = _transformer_learning_rate(lr_config, global_step)
    else:
        raise ValueError("Unknown learning rate schedule: {}".format(name))
    return decay


def _transformer_learning_rate(lr_config, global_step):
    lr = lr_config.rate
    warmup_steps = lr_config.warmup_steps
    decay_rate = lr_config.decay_rate
    if warmup_steps > 0:
        # add 1 to global_step so that we start at 1 instead of 0
        global_step_float = tf.cast(global_step, tf.float32) + 1.
        lr *= tf.minimum(tf.rsqrt(global_step_float),
                         tf.multiply(global_step_float, warmup_steps ** -decay_rate))
        return lr
    else:
        decay_steps = lr_config.decay_steps
        if decay_steps > 0:
            return lr * decay_rate ** (global_step / decay_steps)
        else:
            return lr


def get_optimizer(network_config, default_optimizer=tf.train.AdadeltaOptimizer(learning_rate=1.0)):
    """
    Return the optimizer given by the input network configuration, or a default optimizer.
    :param network_config: network configuration
    :param default_optimizer: default optimization algorithm
    :return: configured optimizer
    """
    try:
        optimizer = network_config.optimizer
    except KeyError:
        tf.logging.info("Using Adadelta as default optimizer.")
        return default_optimizer
    if isinstance(optimizer.lr, numbers.Number):
        lr = optimizer.lr
    else:
        lr = get_learning_rate(optimizer.lr, tf.train.get_global_step())

    name = optimizer.name
    if "Adadelta" == name:
        opt = tf.train.AdadeltaOptimizer(lr, **optimizer.params)
    elif "Adam" == name:
        opt = tf.train.AdamOptimizer(lr, **optimizer.params)
    elif "LazyAdam" == name:
        opt = LazyAdamOptimizer(lr, **optimizer.params)
    elif "LazyNadam" == name:
        opt = LazyNadamOptimizer(lr, **optimizer.params)
    elif "SGD" == name:
        opt = tf.train.GradientDescentOptimizer(lr)
    elif "Momentum" == name:
        opt = tf.train.MomentumOptimizer(lr, **optimizer.params)
    elif "Nadam" == name:
        opt = NadamOptimizerSparse(lr, **optimizer.params)
    else:
        raise ValueError("Invalid optimizer name: {}".format(name))
    return opt


def get_gradient_clip(network_config, default_val=5.0):
    """
    Given a configuration, return the clip norm, or a given default value.
    :param network_config: network configuration
    :param default_val: default clip norm
    :return: clip norm
    """
    clip = network_config.optimizer.get('clip')
    if clip is None:
        tf.logging.info("Using default global norm of gradient clipping threshold of %f", default_val)
        clip = default_val
    return clip


def train_op_from_config(config, loss):
    optimizer = get_optimizer(config)
    clip_norm = get_gradient_clip(config)

    parameters = tf.trainable_variables()
    gradients = tf.gradients(loss, parameters)
    gradients = tf.clip_by_global_norm(gradients, clip_norm=clip_norm)[0]
    return optimizer.apply_gradients(grads_and_vars=zip(gradients, parameters), global_step=tf.train.get_global_step())


def append_label(metric_key, target_key, default_val=None):
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
