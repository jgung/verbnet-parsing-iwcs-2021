import tensorflow as tf
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper, LSTMStateTuple, LayerRNNCell

from tfnlp.common.constants import LENGTH_KEY


def input_layer(features, params, training):
    """
    Declare an input layers that compose multiple features into a single tensor across multiple time steps.
    :param features: input dictionary from feature names to 3D/4D Tensors
    :param params: feature/network configurations/metadata
    :param training: true if training
    :return: 3D tensor ([batch_size, time_steps, input_dim])
    """
    inputs = []
    for feature_config in params.extractor.features.values():
        if feature_config.has_vocab():
            inputs.append(_get_input(features[feature_config.name], feature_config, training))
    inputs = tf.concat(inputs, -1, name="inputs")
    inputs = tf.layers.dropout(inputs, rate=params.config.input_dropout, training=training, name='input_layer_dropout')
    return inputs


def encoder(features, inputs, mode, params):
    def cell(name=None):
        _cell = tf.nn.rnn_cell.LSTMCell(params.config.state_size, name=name, initializer=orthogonal_initializer(4))
        keep_prob = (1.0 - params.config.dropout) if mode == tf.estimator.ModeKeys.TRAIN else 1.0
        return DropoutWrapper(_cell, variational_recurrent=True, dtype=tf.float32,
                              output_keep_prob=keep_prob, state_keep_prob=keep_prob)

    outputs = inputs
    for i in range(params.config.encoder_layers):
        with tf.variable_scope("biRNN_%d" % i):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell(), cell_bw=cell(),
                                                         inputs=outputs, sequence_length=features[LENGTH_KEY], dtype=tf.float32)
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


def orthogonal_initializer(num_splits):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if num_splits == 1:
            return tf.orthogonal_initializer(seed=None, dtype=dtype).__call__(shape, dtype, partition_info)
        shape = (shape[0], (shape[1] // num_splits))
        matrices = []
        for i in range(num_splits):
            matrices.append(tf.orthogonal_initializer(seed=None, dtype=dtype).__call__(shape, dtype, partition_info))
        return tf.concat(axis=1, values=matrices)

    return _initializer


# noinspection PyAbstractClass
class HighwayLSTMCell(LayerRNNCell):

    def __init__(self, num_units,
                 highway=True,
                 cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 forget_bias=1.0,
                 activation=None, reuse=None, name=None):
        """Initialize the parameters for an LSTM cell with simplified highway connections as described in
        'Deep Semantic Role Labeling: What works and what's next' (He et al. 2017).

        Args:
          num_units: int, The number of units in the LSTM cell.
          highway: (optional) Python boolean describing whether to include highway connections
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices. Uses an orthonormal initializer if none is provided.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
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
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units))
            self._output_size = num_units

        # initialized in self.build
        self._input_kernel = None
        self._hidden_kernel = None
        self._bias = None
        self._proj_kernel = None

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
        h_depth = self._num_units if self._num_proj is None else self._num_proj

        num_splits = self._highway and 5 or 4
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape=[input_depth, num_splits * self._num_units],
            initializer=self._initializer if self._initializer else orthogonal_initializer(num_splits))
        self._bias = self.add_variable(
            "bias",
            shape=[num_splits * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))
        num_splits = self._highway and 6 or 4
        self._hidden_kernel = self.add_variable(
            "hidden_kernel",
            shape=[h_depth, num_splits * self._num_units],
            initializer=self._initializer if self._initializer else orthogonal_initializer(num_splits))
        if self._num_proj is not None:
            self._proj_kernel = self.add_variable(
                "projection/%s" % "kernel",
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer if self._initializer else orthogonal_initializer(1))

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

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)
            if self._proj_clip is not None:
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)

        new_state = (LSTMStateTuple(c, m))
        return m, new_state
