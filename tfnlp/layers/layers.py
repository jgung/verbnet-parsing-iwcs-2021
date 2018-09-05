import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper, LSTMStateTuple, LayerRNNCell

from tfnlp.common.constants import ELMO_KEY, LENGTH_KEY

ELMO_URL = "https://tfhub.dev/google/elmo/2"


def input_layer(features, params, training, elmo=False):
    """
    Declare an input layers that compose multiple features into a single tensor across multiple time steps.
    :param features: input dictionary from feature names to 3D/4D Tensors
    :param params: feature/network configurations/metadata
    :param training: true if training
    :param elmo: if true, use elmo embedding
    :return: 3D tensor ([batch_size, time_steps, input_dim])
    """
    inputs = []
    for feature_config in params.extractor.features.values():
        if feature_config.has_vocab():
            inputs.append(_get_input(features[feature_config.name], feature_config, training))
    if elmo:
        tf.logging.info("Using ELMo module at %s", ELMO_URL)
        elmo_module = hub.Module(ELMO_URL, trainable=True)
        lengths = tf.cast(features[LENGTH_KEY], dtype=tf.int32)
        elmo_embedding = elmo_module(inputs={'tokens': features[ELMO_KEY], 'sequence_len': lengths},
                                     signature="tokens",
                                     as_dict=True)['elmo']
        inputs.append(elmo_embedding)
    inputs = tf.concat(inputs, -1, name="inputs")
    inputs = tf.layers.dropout(inputs, rate=params.config.input_dropout, training=training, name='input_layer_dropout')
    return inputs


def encoder(features, inputs, mode, params):
    if params.config.encoder == 'dblstm':
        lengths = tf.identity(features[LENGTH_KEY])
        keep_prob = (1.0 - params.config.encoder_dropout) if mode == tf.estimator.ModeKeys.TRAIN else 1.0
        _encoder = deep_bidirectional_dynamic_rnn([highway_lstm_cell(params.config.state_size, keep_prob)
                                                   for _ in range(params.config.encoder_layers)], inputs, sequence_length=lengths)
        return _encoder, params.config.state_size
    return stacked_bilstm(features, inputs, mode, params), params.config.state_size * 2


def highway_lstm_cell(size, keep_prob):
    return DropoutWrapper(HighwayLSTMCell(size, highway=True, initializer=numpy_orthogonal_initializer),
                          variational_recurrent=True, dtype=tf.float32, output_keep_prob=keep_prob)


def stacked_bilstm(features, inputs, mode, params):
    def cell(name=None):
        _cell = tf.nn.rnn_cell.LSTMCell(params.config.state_size, name=name, initializer=orthogonal_initializer(4))
        keep_prob = (1.0 - params.config.encoder_dropout) if mode == tf.estimator.ModeKeys.TRAIN else 1.0
        return DropoutWrapper(_cell, variational_recurrent=True, dtype=tf.float32,
                              output_keep_prob=keep_prob, state_keep_prob=keep_prob)

    outputs = inputs
    for i in range(params.config.encoder_layers):
        with tf.variable_scope("biRNN_%d" % i):
            outputs, _ = bidirectional_dynamic_rnn(cell_fw=cell(), cell_bw=cell(), inputs=outputs,
                                                   sequence_length=features[LENGTH_KEY], dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1, name="Concatenate_biRNN_outputs_%d" % i)
    return outputs


def _get_input(feature_ids, feature, training):
    initializer = None
    if training and feature.embedding is not None:
        # noinspection PyUnusedLocal
        def initialize(name, dtype=None, partition_info=None):
            return feature.embedding

        initializer = initialize

    embedding_matrix = tf.get_variable(name='{}_embedding'.format(feature.name),
                                       shape=[feature.vocab_size(), feature.config.dim],
                                       initializer=initializer,
                                       trainable=feature.config.trainable)
    result = tf.nn.embedding_lookup(params=embedding_matrix, ids=feature_ids,
                                    name='{}_lookup'.format(feature.name))  # wrapper of gather

    if feature.config.dropout > 0:
        result = tf.layers.dropout(result,
                                   rate=feature.config.dropout,
                                   training=training,
                                   name='{}_dropout'.format(feature.name))

    if feature.rank == 3:  # reduce multiple vectors per token to a single vector
        with tf.name_scope('{}_reduction_op'.format(feature.name)):
            result = feature.config.func.apply(result)

    if 'word_dropout' in feature.config and feature.config.word_dropout > 0:
        shape = tf.shape(result)
        result = tf.layers.dropout(result,
                                   rate=feature.config.word_dropout,
                                   training=training,
                                   name='{}_dropout'.format(feature.name),
                                   noise_shape=[shape[0], shape[1], 1])

    return result


# noinspection PyUnusedLocal
def numpy_orthogonal_initializer(shape, dtype=tf.float32, partition_info=None):
    flat = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = (u if u.shape == flat else v).reshape(shape)
    return tf.constant(q[:shape[0], :shape[1]], dtype=dtype)


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


def deep_bidirectional_dynamic_rnn(cells, inputs, sequence_length):
    def _reverse(_input, seq_lengths):
        return array_ops.reverse_sequence(input=_input, seq_lengths=seq_lengths, seq_axis=1, batch_axis=0)

    outputs = None
    with tf.variable_scope("dblstm"):
        for i, cell in enumerate(cells):
            if i % 2 == 1:
                with tf.variable_scope("bw-%s" % (i // 2)) as bw_scope:
                    inputs_reverse = _reverse(inputs, seq_lengths=sequence_length)
                    outputs, _ = dynamic_rnn(cell=cell, inputs=inputs_reverse, sequence_length=sequence_length, dtype=tf.float32,
                                             scope=bw_scope)
                    outputs = _reverse(outputs, seq_lengths=sequence_length)
            else:
                with tf.variable_scope("fw-%s" % (i // 2)) as fw_scope:
                    outputs, _ = dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length, dtype=tf.float32,
                                             scope=fw_scope)
            inputs = outputs
    return outputs


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
            ih, jh, fh, oh, rh = array_ops.split(value=hidden_matrix, num_or_size_splits=5, axis=1)
            ix, jx, fx, ox, rx, hx = array_ops.split(value=input_matrix, num_or_size_splits=6, axis=1)

            i = sigmoid(ih + ix)
            o = sigmoid(oh + ox)
            f = sigmoid(fh + fx + self._forget_bias)
            j = self._activation(jh + jx)
            c = f * c_prev + i * j
            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            t = sigmoid(rh + rx)
            _m = o * self._activation(c)
            m = t * _m + (1 - t) * hx

        else:
            ix, jx, fx, ox = array_ops.split(value=input_matrix, num_or_size_splits=4, axis=1)
            ih, jh, fh, oh = array_ops.split(value=hidden_matrix, num_or_size_splits=4, axis=1)

            i = sigmoid(ix + ih)
            o = sigmoid(ox + oh)
            f = sigmoid(fx + fh + self._forget_bias)
            c = i * self._activation(jx + jh) + f * c_prev

            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            m = o * self._activation(c)

        new_state = (LSTMStateTuple(c, m))
        return m, new_state
