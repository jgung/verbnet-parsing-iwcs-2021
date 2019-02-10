from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensor2tensor.layers.common_attention import add_timing_signal_1d, attention_bias_ignore_padding, multihead_attention
from tensorflow.contrib.layers import layer_norm
from tensorflow.contrib.lookup import index_table_from_tensor
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper, LSTMStateTuple, LayerRNNCell

from tfnlp.common import constants

ELMO_URL = "https://tfhub.dev/google/elmo/2"


def embedding(features, feature_config, training):
    if feature_config.name == constants.ELMO_KEY:
        tf.logging.info("Using ELMo module at %s", ELMO_URL)
        elmo_module = hub.Module(ELMO_URL, trainable=True)
        elmo_embedding = elmo_module(inputs={'tokens': features[constants.ELMO_KEY],
                                             'sequence_len': tf.cast(features[constants.LENGTH_KEY], dtype=tf.int32)},
                                     signature="tokens",
                                     as_dict=True)['elmo']
        return elmo_embedding
    elif feature_config.has_vocab():
        feature_embedding = _get_embedding_input(features[feature_config.name], feature_config, training)
        return feature_embedding


def input_layer(features, params, training):
    """
    Declare an input layers that compose multiple features into a single tensor across multiple time steps.
    :param features: input dictionary from feature names to 3D/4D Tensors
    :param params: feature/network configurations/metadata
    :param training: true if training
    :return: 3D tensor ([batch_size, time_steps, input_dim])
    """
    add_group_feats = defaultdict(list)
    inputs = []
    for feature_config in params.extractor.features.values():
        feature_embedding = embedding(features, feature_config, training)
        if feature_config.config.add_group:
            add_group_feats[feature_config.config.add_group].append(feature_embedding)
            continue
        inputs.append(feature_embedding)

    # add together any reduce sum feats (such as adding different encodings of the same word)
    for feature_list in add_group_feats.values():
        inputs.append(reduce_sum(feature_list))

    return concat(inputs, training, params.config)


def string2index(feature_strings, feature):
    """
    Convert a `Tensor` of type `tf.string` to a corresponding Tensor of ids (`tf.int32`)
    :param feature_strings: string `Tensor`
    :param feature: feature extractor with string to index vocabulary
    :return: feature id Tensor
    """
    with tf.variable_scope('lookup'):
        feats = list(feature.ordered_feats())
        lookup = index_table_from_tensor(mapping=tf.constant(feats), default_value=feature.unk_index())
        return lookup.lookup(feature_strings)


def _get_embedding_input(inputs, feature, training):
    config = feature.config

    with tf.variable_scope(feature.name):
        feature_ids = string2index(inputs, feature)

        with tf.variable_scope('embedding'):
            initializer = None
            if training:
                if feature.embedding is not None:
                    initializer = embedding_initializer(feature.embedding)
                elif config.initializer.zero_init:
                    tf.logging.info("Zero init for feature embedding: %s", feature.name)
                    initializer = tf.zeros_initializer
                else:
                    tf.logging.info("Xavier Uniform init for feature embedding: %s", feature.name)
                    initializer = tf.glorot_uniform_initializer

            embedding_matrix = tf.get_variable(name='parameters',
                                               shape=[feature.vocab_size(), config.dim],
                                               initializer=initializer,
                                               trainable=config.trainable)
            result = tf.nn.embedding_lookup(params=embedding_matrix, ids=feature_ids,
                                            name='lookup')  # wrapper of gather

            if config.dropout > 0:
                result = tf.layers.dropout(result,
                                           rate=config.dropout,
                                           training=training,
                                           name='dropout')

        if feature.rank == 3:  # reduce multiple vectors per token to a single vector
            with tf.name_scope('reduce'):
                result = config.func.apply(result)

        if config.word_dropout > 0 and training:
            shape = tf.shape(result)
            result = tf.layers.dropout(result,
                                       rate=config.word_dropout,
                                       training=training,
                                       name='word_dropout',
                                       noise_shape=[shape[0], shape[1], 1])

        return result


def encoder(features, inputs, mode, config):
    training = mode == tf.estimator.ModeKeys.TRAIN
    sequence_lengths = features[constants.LENGTH_KEY]

    with tf.variable_scope("encoder"):
        if constants.ENCODER_CONCAT == config.encoder_type:
            return concat(inputs, training, config)
        elif constants.ENCODER_SUM == config.encoder_type:
            return reduce_sum(inputs)
        elif constants.ENCODER_DBLSTM == config.encoder_type:
            return highway_dblstm(inputs[0], sequence_lengths, training, config)
        elif constants.ENCODER_BLSTM == config.encoder_type:
            return stacked_bilstm(inputs[0], sequence_lengths, training, config)
        elif constants.ENCODER_TRANSFORMER == config.encoder_type:
            return transformer_encoder(inputs[0], sequence_lengths, training, config)
        else:
            raise ValueError('No encoder of type "{}" available'.format(config.encoder_type))


def concat(inputs, training, config):
    result = tf.concat(inputs, -1, name="inputs")
    # apply dropout across entire layer
    result = tf.layers.dropout(result, rate=config.input_dropout, training=training, name='input_layer_dropout')
    return result


def reduce_sum(inputs):
    result = inputs[0]
    for _input in inputs[1:]:
        result += _input
    return result


def highway_dblstm(inputs, sequence_lengths, training, config):
    """
    Initialize a deep bidirectional highway LSTM (with interleaved forward/backward layers) as described in
    "Zhou, J. and Xu, W. End-to-end learning of semantic role labeling using recurrent neural networks" and
    "He, Luheng, et al. Deep semantic role labeling: What works and what’s next."
    :param inputs: batch major input tensor of shape: `[batch_size, max_time, input_dim]`
    :param sequence_lengths:  A vector of size `[batch_size]` containing the actual lengths for each of sequence in the batch.
    :param training: boolean used to indicate if dropout and other training-specific configurations should be enabled
    :param config: network configuration
    :return: tuple with encoder outputs and output dimensionality
    """
    input_keep_prob = (1.0 - config.encoder_input_dropout) if training else 1.0
    keep_prob = (1.0 - config.encoder_dropout) if training else 1.0
    output_keep_prob = (1.0 - config.encoder_output_dropout) if training else 1.0

    def highway_lstm_cell(size):
        _cell = HighwayLSTMCell(size, highway=True, initializer=numpy_orthogonal_initializer)
        return DropoutWrapper(_cell, variational_recurrent=True, dtype=tf.float32,
                              state_keep_prob=keep_prob,
                              input_keep_prob=input_keep_prob,
                              output_keep_prob=output_keep_prob)

    def _reverse(_input):
        return array_ops.reverse_sequence(input=_input, seq_lengths=sequence_lengths, seq_axis=1, batch_axis=0)

    outputs = None
    final_state = None
    with tf.variable_scope("dblstm"):
        cells = [highway_lstm_cell(config.state_size) for _ in range(config.encoder_layers)]

        for i, cell in enumerate(cells):
            odd = i % 2 == 1
            with tf.variable_scope("%s-%s" % ('bw' if odd else 'fw', i // 2)) as layer_scope:
                inputs = _reverse(inputs) if odd else inputs

                outputs, final_state = dynamic_rnn(cell=cell, inputs=inputs,
                                                   sequence_length=sequence_lengths,
                                                   dtype=tf.float32,
                                                   scope=layer_scope)

                outputs = _reverse(outputs) if odd else outputs
            inputs = outputs

    return outputs, config.state_size, final_state


def stacked_bilstm(inputs, sequence_lengths, training, config):
    keep_prob = (1.0 - config.encoder_dropout) if training else 1.0
    input_keep_prob = (1.0 - config.encoder_input_dropout) if training else 1.0
    output_keep_prob = (1.0 - config.encoder_output_dropout) if training else 1.0

    def cell(_size, name=None):
        _cell = tf.nn.rnn_cell.LSTMCell(config.state_size, name=name, initializer=orthogonal_initializer(4),
                                        forget_bias=config.forget_bias)
        return DropoutWrapper(_cell, variational_recurrent=True, dtype=tf.float32,
                              input_size=_size,
                              output_keep_prob=output_keep_prob,
                              state_keep_prob=keep_prob,
                              input_keep_prob=input_keep_prob)

    outputs = inputs
    fw_state, bw_state = None, None
    for i in range(config.encoder_layers):
        with tf.variable_scope("biRNN_%d" % i):
            size = outputs.get_shape().as_list()[-1]
            outputs, (fw_state, bw_state) = bidirectional_dynamic_rnn(cell_fw=cell(size), cell_bw=cell(size), inputs=outputs,
                                                                      sequence_length=sequence_lengths, dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1, name="Concatenate_biRNN_outputs_%d" % i)
    return outputs, config.state_size * 2, tf.concat([fw_state.h, bw_state.h], axis=1)


def embedding_initializer(np_embedding):
    # noinspection PyUnusedLocal
    def _initializer(name, dtype=None, partition_info=None):
        return np_embedding

    return _initializer


def numpy_orthogonal_matrix(shape):
    flat = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = (u if u.shape == flat else v).reshape(shape)
    return q[:shape[0], :shape[1]]


# noinspection PyUnusedLocal
def numpy_orthogonal_initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(numpy_orthogonal_matrix(shape), dtype=dtype)


def orthogonal_initializer(num_splits):
    # noinspection PyUnusedLocal
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if num_splits == 1:
            return numpy_orthogonal_initializer(shape, dtype, partition_info)
        shape = (shape[0], (shape[1] // num_splits))
        matrices = []
        for i in range(num_splits):
            matrices.append(numpy_orthogonal_initializer(shape, dtype, partition_info))
        return tf.concat(axis=1, values=matrices)

    return _initializer


# noinspection PyAbstractClass
class HighwayLSTMCell(LayerRNNCell):

    def __init__(self, num_units,
                 highway=False,
                 cell_clip=None,
                 initializer=None,
                 forget_bias=1.0,
                 activation=None, reuse=None, name=None):
        """Initialize the parameters for an LSTM cell with simplified highway connections as described in
        'Deep Semantic Role Labeling: What works and what's next' (He et al. 2017).

        Args:
          num_units: int, The number of units in the LSTM cell.
          highway: (optional) Python boolean describing whether to include highway connections
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight matrices.
            Uses an orthonormal initializer if none is provided.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
        """
        super(HighwayLSTMCell, self).__init__(_reuse=reuse, name=name)
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units

        self._highway = highway
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        self._state_size = (LSTMStateTuple(num_units, num_units))
        self._output_size = num_units

        # initialized in self.build
        self._input_kernel = None
        self._hidden_kernel = None
        self._bias = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units

        num_splits = self._highway and 6 or 4

        self._input_kernel = tf.concat([self.add_variable(
            "input_kernel_{}".format(i),
            shape=[input_depth, size],
            initializer=self._initializer) for i, size in enumerate(num_splits * [self._num_units])], axis=1)
        self._bias = self.add_variable(
            "bias",
            shape=[num_splits * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        num_splits = self._highway and 5 or 4
        self._hidden_kernel = tf.concat([self.add_variable(
            "hidden_kernel_{}".format(i),
            shape=[h_depth, size],
            initializer=self._initializer) for i, size in enumerate(num_splits * [self._num_units])], axis=1)

        self.built = True

    # noinspection PyMethodOverriding
    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid

        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate, r = transform gate
        input_matrix = math_ops.matmul(inputs, self._input_kernel)
        input_matrix = nn_ops.bias_add(input_matrix, self._bias)

        hidden_matrix = math_ops.matmul(m_prev, self._hidden_kernel)

        if self._highway:
            i, j, f, o, r = array_ops.split(hidden_matrix + input_matrix[:, :-self._num_units], num_or_size_splits=5, axis=1)
            hx = input_matrix[:, -self._num_units:]

            i = sigmoid(i)
            o = sigmoid(o)
            f = sigmoid(f + self._forget_bias)
            j = self._activation(j)
            c = f * c_prev + i * j
            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            t = sigmoid(r)
            _m = o * self._activation(c)
            m = t * _m + (1 - t) * hx

        else:
            i, j, f, o = array_ops.split(value=input_matrix + hidden_matrix, num_or_size_splits=4, axis=1)

            i = sigmoid(i)
            o = sigmoid(o)
            f = sigmoid(f + self._forget_bias)
            c = i * self._activation(j) + f * c_prev

            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            m = o * self._activation(c)

        new_state = (LSTMStateTuple(c, m))
        return m, new_state


def transformer_encoder(inputs, sequence_lengths, training, config):
    # nonlinear projection of input to dimensionality of transformer (head size x num heads)
    with tf.variable_scope("encoder_input_proj"):
        inputs = tf.nn.leaky_relu(tf.layers.dense(inputs, config.head_dim * config.num_heads), alpha=0.1)

    with tf.variable_scope('transformer'):
        mask = tf.sequence_mask(sequence_lengths, name="padding_mask", dtype=tf.int32)
        # e.g. give attention bias [0 0 0 0 -inf -inf -inf] for a sequence length of 4 -- don't attend to padding nodes
        attention_bias = attention_bias_ignore_padding(tf.cast(1 - mask, tf.float32))
        # add sinusoidal timing signal to give position information to inputs
        inputs = add_timing_signal_1d(inputs)
        for i in range(config.encoder_layers):
            with tf.variable_scope('layer%d' % i):
                inputs = transformer(inputs, attention_bias, training, config)

    # apply final layer norm
    inputs = layer_norm(inputs)

    return inputs, config.head_dim * config.num_heads, None


def transformer(inputs, attention_bias, training, config):
    def _layer_norm(_x):
        return layer_norm(_x, begin_norm_axis=-1, begin_params_axis=-1)

    def _residual(_x, _y):
        return tf.add(_x, tf.layers.dropout(_y, rate=config.prepost_dropout, training=training))

    self_attention_dim = config.head_dim * config.num_heads
    with tf.name_scope('transformer_layer'):
        with tf.variable_scope("self_attention"):
            # apply layer norm before self attention layer
            x = _layer_norm(inputs)

            # multi-head self-attention
            y = multihead_attention(query_antecedent=x, memory_antecedent=None,
                                    bias=attention_bias,
                                    total_key_depth=self_attention_dim,
                                    total_value_depth=self_attention_dim,
                                    output_depth=self_attention_dim,
                                    num_heads=config.num_heads,
                                    dropout_rate=config.attention_dropout if training else 0,
                                    attention_type="dot_product")
            x = _residual(x, y)

        with tf.variable_scope("ffnn"):
            # apply layer norm after self attention layer
            x = layer_norm(x, begin_norm_axis=-1, begin_params_axis=-1)

            y = _mlp(x,
                     hidden_size=config.relu_hidden_size,
                     output_size=self_attention_dim,
                     keep_prob=(1 - config.relu_dropout) if training else 1.0)
            # residual connection
            x = _residual(x, y)

        return x


def _mlp(inputs, hidden_size, output_size, keep_prob):
    with tf.variable_scope("conv_mlp", [inputs]):
        inputs = tf.expand_dims(inputs, 1)
        input_dim = inputs.get_shape().as_list()[-1]

        def ff(name, x, in_dim, out_dim, last=False):
            weights = tf.get_variable(name, [1, 1, in_dim, out_dim])

            h = tf.nn.conv2d(x, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
            if not last:
                h = tf.nn.leaky_relu(h, alpha=0.1)
                h = tf.nn.dropout(h, keep_prob)
            else:
                h = tf.squeeze(h, 1)
            return h

        y = ff("ff1", inputs, input_dim, hidden_size)
        y = ff("ff2", y, hidden_size, hidden_size)
        y = ff("ff3", y, hidden_size, output_size, last=True)

        return y
