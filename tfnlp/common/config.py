import numbers

import tensorflow as tf
from tensorflow.python.training.learning_rate_decay import exponential_decay, inverse_time_decay

from tfnlp.common.constants import TAGGER_KEY
from tfnlp.common.utils import Params
from tfnlp.optim.nadam import NadamOptimizerSparse


class BaseNetworkConfig(Params):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.features = config.get('features')
        self.optimizer = config.get('optimizer')
        self.reader = config.get('reader')

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
        self.exports_to_keep = config.get('exports_to_keep', 5)

        self.input_dropout = config.get('input_dropout', 0)
        self.buckets = config.get('buckets', [10, 15, 25, 30, 75])
        self.metric = config.get('metric', 'Accuracy')
        self.encoder = config.get('encoder', 'lstm')
        self.forget_bias = config.get('forget_bias', 1)
        self.encoder_dropout = config.get('encoder_dropout', 0)
        self.encoder_input_dropout = config.get('encoder_input_dropout', 0)
        self.encoder_output_dropout = config.get('encoder_output_dropout', 0)
        self.encoder_layers = config.get('encoder_layers', 1)
        self.state_size = config.get('state_size', 100)
        self.crf = config.get('crf', False)
        self.mlp_dropout = config.get('mlp_dropout', 0)
        self.zero_init = config.get('zero_init', True)
        self.type = config.get('type', TAGGER_KEY)


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
    else:
        raise ValueError("Unknown learning rate schedule: {}".format(name))
    return decay


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
    try:
        return network_config.optimizer.clip
    except KeyError:
        tf.logging.info("Using default global norm of gradient clipping threshold of %f", default_val)
        return default_val
