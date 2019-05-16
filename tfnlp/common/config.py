import numbers
import re

import tensorflow as tf
from bert.optimization import AdamWeightDecayOptimizer
from tensorflow.contrib.opt import LazyAdamOptimizer
from tensorflow.python.training.learning_rate_decay import exponential_decay, inverse_time_decay

from tfnlp.common import constants
from tfnlp.common.utils import Params
from tfnlp.optim.lazy_adam import LazyAdamOptimizer as LazyNadamOptimizer
from tfnlp.optim.nadam import NadamOptimizerSparse

_TYPE_TASK_MAP = {
    constants.BIAFFINE_SRL_KEY: constants.SRL_KEY
}


class BaseNetworkConfig(Params):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.reader = config.get('reader')

        # training hyperparameters
        self.batch_size = config.get('batch_size')
        if not self.batch_size:
            self.batch_size = 10
            tf.logging.warn("No 'batch_size' parameter provided. Using default value of %d", self.batch_size)
        self.buffer_size = config.get('buffer_size', 999999)  # in other words, read full dataset into memory by default
        self.batch_buffer_size = config.get('batch_buffer_size', 512)  # number of consecutive batches to shuffle
        self.dataset_caching = config.get('dataset_caching', True)

        self.checkpoint_steps = config.get('checkpoint_steps')
        self.patience = config.get('patience')
        self.max_steps = config.get('max_steps')

        self.max_epochs = config.get('max_epochs')
        self.patience_epochs = config.get('patience_epochs')
        self.checkpoint_epochs = config.get('checkpoint_epochs')

        self.exports_to_keep = config.get('exports_to_keep', 1)
        self.keep_checkpoints = config.get('checkpoints_to_keep', 1)

        # feature/input settings
        self.features = config.get('features')
        self.bucket_sizes = config.get('bucket_sizes')
        self.max_length = config.get('max_length', 100)

        # Decay for exponential moving average (EMA) of parameters -- 0.998 or 0.999 is standard
        # "Temporal averaging for semi-supervised learning", Laine and Aila 2017. https://arxiv.org/abs/1610.02242
        self.ema_decay = config.get('ema_decay', 0)

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

        optimizer_config = config.get('optimizer')
        if optimizer_config:
            self.optimizer = OptimizerConfig(optimizer_config)


class OptimizerConfig(Params):
    def __init__(self, optimizer_config, **kwargs):
        super().__init__(**optimizer_config, **kwargs)
        self.name = optimizer_config.name
        self.params = optimizer_config.params if optimizer_config.get('params') else {}

        clip = optimizer_config.get('clip')
        if not clip:
            clip = 5.0
            tf.logging.info("Using default global norm of gradient clipping threshold of %f", clip)
        self.clip = clip


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
        self.sequence_length_key = config.get('sequence_length_key', constants.LENGTH_KEY)

        # transformer encoder settings
        self.num_heads = config.get('num_heads', 8)
        self.head_dim = config.get('head_dim', 25)
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

        self.task = config.get('task')
        if not self.task:
            self.task = constants.TAGGER_KEY
            if 'type' in config:
                self.task = _TYPE_TASK_MAP.get(config.type, config.type)
            tf.logging.warn("No 'task' parameter provided for head %s. Using default of %s", self.name, self.task)

        self.type = config.get('type')
        if not self.type:
            tf.logging.warn("No 'type' parameter provided for head %s. Using default of %s", self.name, self.task)

        self.zero_init = config.get('zero_init', True)
        self.metric = config.get('metric', constants.OVERALL_KEY)
        # "Rethinking the Inception Architecture for Computer Vision", Szegedy et al. 2015 -- 0.1 is default
        self.label_smoothing = config.get('label_smoothing', 0)
        # "Regularizing Neural Networks by Penalizing Confident Predictions", Pereyra et al. 2017 -- 1.0 is default
        self.confidence_penalty = config.get('confidence_penalty', 0)


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
    elif "bert" == name:
        decay = _bert_learning_rate(lr_config, global_step)
    else:
        raise ValueError("Unknown learning rate schedule: {}".format(name))
    return decay


def _bert_learning_rate(lr_config, global_step):
    # adapted from https://github.com/google-research/bert/blob/master/optimization.py
    init_lr = lr_config.rate
    warmup_steps = int(lr_config.warmup_proportion * lr_config.num_train_steps)
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        lr_config.num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)
    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    return learning_rate


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
        optimizer.lr.num_train_steps = network_config.max_steps
        lr = get_learning_rate(optimizer.lr, tf.train.get_global_step())

    name = optimizer.name
    params = optimizer.params
    if "Adadelta" == name:
        opt = tf.train.AdadeltaOptimizer(lr, **params)
    elif "Adam" == name:
        opt = tf.train.AdamOptimizer(lr, **params)
    elif "LazyAdam" == name:
        opt = LazyAdamOptimizer(lr, **params)
    elif "LazyNadam" == name:
        opt = LazyNadamOptimizer(lr, **params)
    elif "SGD" == name:
        opt = tf.train.GradientDescentOptimizer(lr)
    elif "Momentum" == name:
        opt = tf.train.MomentumOptimizer(lr, **params)
    elif "Nadam" == name:
        opt = NadamOptimizerSparse(lr, **params)
    elif "bert" == name:
        opt = AdamWeightDecayOptimizer(lr, weight_decay_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-6,
                                       exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    else:
        raise ValueError("Invalid optimizer name: {}".format(name))
    return opt


def get_l2_loss(network_config, variables):
    if not network_config.optimizer or not network_config.optimizer.get('l2_loss'):
        return 0
    l2_loss = network_config.optimizer.get('l2_loss')

    if isinstance(l2_loss, numbers.Number):
        return tf.add_n([tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name]) * l2_loss
    if not isinstance(l2_loss, dict):
        raise ValueError("'l2_loss' expects a dictionary from regular expressions matching variable names to L2 terms,"
                         " e.g. {\".*scalar.*\": 0.001}, or a single L2 term to be applied globally to non-bias weights.")

    all_losses = []
    for var_pattern, alpha in l2_loss.items():
        for var in [v for v in variables if re.match(var_pattern, v.name)]:
            tf.logging.info('Adding L2 regularization with alpha=%f to %s' % (alpha, var.name))
            all_losses.append(alpha * tf.nn.l2_loss(var))

    return tf.add_n(all_losses)


def train_op_from_config(config, loss):
    optimizer = get_optimizer(config)
    clip_norm = config.optimizer.clip

    parameters = tf.trainable_variables()

    # optionally add L2 loss to specific weights, or globally
    l2_loss = get_l2_loss(config, parameters)
    if l2_loss is not None:
        loss += l2_loss

    gradients = tf.gradients(loss, parameters)
    gradients = tf.clip_by_global_norm(gradients, clip_norm=clip_norm)[0]

    global_step = tf.train.get_global_step()
    result = optimizer.apply_gradients(grads_and_vars=zip(gradients, parameters), global_step=global_step)
    if isinstance(optimizer, AdamWeightDecayOptimizer):
        # AdamWeightDecayOptimizer does not update the global step, unlike other optimizers
        new_global_step = global_step + 1
        train_op = tf.group(result, [global_step.assign(new_global_step)])
        return train_op
    return result


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
