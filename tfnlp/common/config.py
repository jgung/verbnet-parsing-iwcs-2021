import numbers
import re

import tensorflow as tf
from tensorflow.python.training.learning_rate_decay import exponential_decay, inverse_time_decay

import tfnlp.feature
from tfnlp.common.constants import ELMO_KEY, INITIALIZER, LOWER, NORMALIZE_DIGITS, UNKNOWN_WORD
from tfnlp.common.utils import Params
from tfnlp.layers.reduce import ConvNet
from tfnlp.optim.nadam import NadamOptimizerSparse


def _get_reduce_function(config, dim, length):
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


def _get_mapping_function(func, rank=2):
    if func == LOWER:
        if rank == 3:
            return lambda x: [word.lower() for word in x]
        return lambda x: x.lower()
    elif func == NORMALIZE_DIGITS:
        if rank == 3:
            return lambda x: [re.sub("\d", "#", word) for word in x]
        return lambda x: re.sub("\d", "#", x)
    raise AssertionError("Unexpected function name: {}".format(func))


class FeatureInitializer(Params):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # number of entries (words) in initializer (embedding) to include in vocabulary
        self.include_in_vocab = config.get('include_in_vocab', 0)
        # path to raw embedding file
        self.embedding = config.get('embedding')
        if not self.embedding:
            raise AssertionError("Missing 'embedding' parameter, which provides the path to raw embedding file")
        # name of serialized initializer after vocabulary training
        self.pkl_path = config.get('pkl_path')
        if not self.pkl_path:
            raise AssertionError("Missing 'pkl_path' parameter, which provides the path to the resulting serialized initializer")


class FeatureHyperparameters(Params):
    def __init__(self, config, feature, **kwargs):
        super().__init__(**kwargs)
        # dropout specific to this feature
        self.dropout = config.get('dropout', 0)
        # word-level dropout for this feature
        self.word_dropout = config.get('word_dropout', 0)
        # indicates whether variables for this feature should be trainable (`True`) or fixed (`False`)
        self.trainable = config.get('trainable', True)
        # dimensionality for this feature if an initializer is not provided
        self.dim = config.get('dim', 0)
        # variable initializer used to initialize lookup table for this feature (such as word embeddings)
        initializer = config.get(INITIALIZER)
        self.initializer = FeatureInitializer(initializer) if initializer else None
        reduce_func = config.get('function')
        if reduce_func:
            self.func = _get_reduce_function(reduce_func, self.dim, feature.max_len)


class FeatureConfig(Params):
    def __init__(self, feature, **kwargs):
        # name used to instantiate this feature
        super().__init__(**kwargs)
        self.name = feature.get('name')
        # key used for lookup during feature extraction
        self.key = feature.get('key')
        # number of tokens to use for left padding
        self.left_padding = feature.get('left_padding', 0)
        # number of tokens to use for right padding
        self.right_padding = feature.get('right_padding', 0)
        # 2 most common feature rank for our NLP applications (word/token-level features)
        self.rank = feature.get('rank', 2)
        # string mapping functions applied during extraction
        self.mapping_funcs = [_get_mapping_function(mapping_func, self.rank) for mapping_func in feature.get('mapping_funcs', [])]
        # maximum sequence length of feature
        self.max_len = feature.get('max_len')
        # word used to replace OOV tokens
        self.unknown_word = feature.get('unknown_word', UNKNOWN_WORD)
        # pre-initialized vocabulary
        self.indices = feature.get('indices')
        self.numeric = feature.get('numeric')
        # hyperparameters related to this feature
        config = feature.get('config', Params())
        self.config = FeatureHyperparameters(config, self)


class FeaturesConfig(object):
    def __init__(self, features):
        self.seq_feat = features.get('seq_feat', 'word')

        targets = features.get('targets')
        if not targets:
            tf.logging.warn("No 'targets' parameter provided in feature configuration--this could be an error if training.")
            self.targets = []
        else:
            self.targets = [_get_feature(target) for target in targets]

        features = features.get('features')
        if not features:
            raise AssertionError("No 'features' parameter provided in feature configuration, requires at least one feature")
        else:
            self.features = [_get_feature(feature) for feature in features]


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
            self.checkpoint_steps = 1000
            tf.logging.warn("No 'checkpoint_steps' parameter provided. Using default value of %d", self.checkpoint_steps)
        self.input_dropout = config.get('input_dropout', 0)
        self.buckets = config.get('buckets', [10, 15, 25, 30, 75])
        self.metric = config.get('metric', 'Accuracy')
        self.encoder = config.get('encoder', 'lstm')
        self.encoder_dropout = config.get('encoder_dropout', 0)
        self.encoder_layers = config.get('encoder_layers', 1)
        self.state_size = config.get('state_size', 100)
        self.crf = config.get('crf', False)
        self.mlp_dropout = config.get('mlp_dropout', 0)


def _get_feature(feature):
    """
    Create an individual feature from an input feature configuration.
    :param feature: feature configuration
    :return: feature
    """
    feature = FeatureConfig(feature)
    if feature.name == ELMO_KEY:
        feat = tfnlp.feature.TextExtractor
    elif feature.rank == 3:
        feat = tfnlp.feature.SequenceListFeature
    elif feature.rank == 2:
        feat = tfnlp.feature.SequenceExtractor if feature.numeric else tfnlp.feature.SequenceFeature
    elif feature.rank == 1:
        feat = tfnlp.feature.Extractor if feature.numeric else tfnlp.feature.Feature
    else:
        raise AssertionError("Unexpected feature rank: {}".format(feature.rank))
    return feat(**feature)


def get_feature_extractor(config):
    """
    Create a `FeatureExtractor` from a given feature configuration.
    :param config: feature configuration
    :return: feature extractor
    """
    # load default configurations
    config = FeaturesConfig(config)

    # use this feature to keep track of instance indices for error analysis
    config.features.append(tfnlp.feature.index_feature())
    # used to establish sequence length for bucketed batching
    config.features.append(tfnlp.feature.LengthFeature(config.seq_feat))

    return tfnlp.feature.FeatureExtractor(features=config.features, targets=config.targets)


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
