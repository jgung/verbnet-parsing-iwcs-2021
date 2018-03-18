import numbers

import tensorflow as tf
from tensorflow.python.training.learning_rate_decay import exponential_decay, inverse_time_decay

from tfnlp.feature import Feature, FeatureExtractor, LengthFeature, SequenceFeature, SequenceListFeature
from tfnlp.layers.reduce import ConvNet


def get_reduce_function(func, dim, length):
    if func.name == "ConvNet":
        return ConvNet(input_size=dim, kernel_size=func.kernel_size, num_filters=func.num_filters, max_length=length)
    else:
        raise AssertionError("Unexpected feature function: {}".format(func.name))


def get_feature(feature):
    if feature.rank == 3:
        feat = SequenceListFeature
        feature.config.func = get_reduce_function(feature.config.function, feature.config.dim, feature.max_len)
    elif feature.rank == 2:
        feat = SequenceFeature
    elif feature.rank == 1:
        feat = Feature
    else:
        raise AssertionError("Unexpected feature rank: {}".format(feature.rank))
    return feat(**feature)


def get_feature_extractor(config):
    features = []
    for feature in config.features:
        features.append(get_feature(feature))
    targets = []
    for target in config.targets:
        targets.append(get_feature(target))

    features.append(LengthFeature(config.seq_feat))

    return FeatureExtractor(features, targets)


def get_learning_rate(lr_config, global_step):
    lr = lr_config.rate
    name = lr_config.name
    if "exponential_decay" == name:
        decay = exponential_decay(lr, global_step, **lr_config.params)
    elif "inverse_time_decay" == name:
        decay = inverse_time_decay(lr, global_step, **lr_config.params)
    else:
        raise ValueError("Unknown learning rate schedule: {}".format(name))
    return decay


def get_optimizer(network_config):
    try:
        optimizer = network_config.optimizer
    except KeyError:
        tf.logging.info("Using Adadelta as default optimizer.")
        return tf.train.AdadeltaOptimizer(learning_rate=1.0)
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
        opt = tf.contrib.opt.NadamOptimizer(lr, **optimizer.params)
    else:
        raise ValueError("Invalid optimizer name: {}".format(name))
    return opt


def get_gradient_clip(network_config):
    try:
        return network_config.optimizer.clip
    except KeyError:
        tf.logging.info("Using default global norm of gradient clipping threshold of 5.0")
        return 5.0
