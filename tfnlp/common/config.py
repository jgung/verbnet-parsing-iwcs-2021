import numbers

import tensorflow as tf
from tensorflow.python.training.learning_rate_decay import exponential_decay, inverse_time_decay

import tfnlp.feature
from tfnlp.common.constants import ELMO_KEY
from tfnlp.layers.reduce import ConvNet
from tfnlp.optim.nadam import NadamOptimizerSparse


def get_reduce_function(config, dim, length):
    """
    Return a neural sequence reduction op given a configuration, input dimensionality, and max length.
    :param config: reduction op configuration
    :param dim: input dimensionality
    :param length: input max length
    :return: neural sequence reduction operation
    """
    if config.name == "ConvNet":
        return ConvNet(input_size=dim, kernel_size=config.kernel_size, num_filters=config.num_filters, max_length=length)
    else:
        raise AssertionError("Unexpected feature function: {}".format(config.name))


def get_feature(feature):
    """
    Create an individual feature from an input feature configuration.
    :param feature: feature configuration
    :return: feature
    """
    numeric = feature.get('numeric')

    if feature.name == ELMO_KEY:
        feat = tfnlp.feature.TextExtractor
    elif feature.rank == 3:
        feat = tfnlp.feature.SequenceListFeature
        feature.config.func = get_reduce_function(feature.config.function, feature.config.dim, feature.max_len)
    elif feature.rank == 2:
        feat = tfnlp.feature.SequenceExtractor if numeric else tfnlp.feature.SequenceFeature
    elif feature.rank == 1:
        feat = tfnlp.feature.Extractor if numeric else tfnlp.feature.Feature
    else:
        raise AssertionError("Unexpected feature rank: {}".format(feature.rank))
    return feat(**feature)


def get_feature_extractor(config):
    """
    Create a `FeatureExtractor` from a given feature configuration.
    :param config: feature configuration
    :return: feature extractor
    """
    features = []
    for feature in config.features:
        features.append(get_feature(feature))
    targets = []
    for target in config.targets:
        targets.append(get_feature(target))

    features.append(tfnlp.feature.LengthFeature(config.seq_feat))
    features.append(tfnlp.feature.index_feature())

    return tfnlp.feature.FeatureExtractor(features, targets)


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
