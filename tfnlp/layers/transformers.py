# Copyright 2019 The Tensor2Tensor Authors.
#
# Modifications Copyright 2020 James Gung.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# # limitations under the License.

import math

import tensorflow as tf


def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        x_name = "(eager Tensor)"
        try:
            x_name = x.name
        except AttributeError:
            pass
        tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                           x.device, cast_x.device)
    return cast_x


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def large_compatible_negative(tensor_type):
    """Large negative number as Tensor.

    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16

    Args:
      tensor_type: a dtype to determine the type.

    Returns:
      a large negative number.
    """
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1e9


def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
      x: a Tensor with shape [batch, length, channels]
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position

    Returns:
      a Tensor the same shape as x.
    """
    length = shape_list(x)[1]
    channels = shape_list(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale, start_index)
    return x + cast_like(signal, x)


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
      length: scalar, length of timing signal sequence.
      channels: scalar, size of timing embeddings to create. The number of
          different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position

    Returns:
      a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            tf.maximum(tf.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    # Please note that this slightly differs from the published paper.
    # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.

    Args:
      memory_padding: a float `Tensor` with shape [batch, memory_length].

    Returns:
      a `Tensor` with shape [batch, 1, 1, memory_length].
    """
    ret = memory_padding * large_compatible_negative(memory_padding.dtype)
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          name="dot_product_attention"):
    """Dot-product attention.

    Args:
      q: Tensor with shape [..., length_q, depth_k].
      k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
        match with q.
      v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
        match with q.
      bias: bias Tensor (see attention_bias())
      dropout_rate: a float.
      name: an optional string

    Returns:
      Tensor with shape [..., length_q, depth_v].
    """
    with tf.variable_scope(name, values=[q, k, v]):
        logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
        if bias is not None:
            bias = cast_like(bias, logits)
            logits += bias
        # If logits are fp16, upcast before softmax
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = cast_like(weights, q)
        # Drop out attention links for each head.
        weights = tf.nn.dropout(weights, keep_prob=1 - dropout_rate)
        return tf.matmul(weights, v)


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.

    Args:
      x: a Tensor with shape [..., a, b]

    Returns:
      a Tensor with shape [..., ab]
    """
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


def combine_heads(x):
    """Inverse of split_heads.

    Args:
      x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

    Returns:
      a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.

    The first of these two dimensions is n.

    Args:
      x: a Tensor with shape [..., m]
      n: an integer.

    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).

    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer

    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def compute_attention_component(antecedent,
                                total_depth,
                                name):
    """Computes attention component (query, key or value).

    Args:
      antecedent: a Tensor with shape [batch, length, channels]
      total_depth: an integer
      name: a string specifying scope name.

    Returns:
      c : [batch, length, depth] tensor
    """
    return tf.layers.dense(antecedent, total_depth, use_bias=False, name=name)


def compute_qkv(query_antecedent,
                total_key_depth,
                total_value_depth):
    """Computes query, key and value.

    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      total_key_depth: an integer
      total_value_depth: an integer

    Returns:
      q, k, v : [batch, length, depth] tensors
    """
    q = compute_attention_component(query_antecedent, total_key_depth, "q")
    k = compute_attention_component(query_antecedent, total_key_depth, "k")
    v = compute_attention_component(query_antecedent, total_value_depth, "v")
    return q, k, v


def multihead_attention(query_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        name="multihead_attention"):
    """Multihead scaled-dot-product attention with input/output transformations.

    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      bias: bias Tensor (see attention_bias())
      total_key_depth: an integer
      total_value_depth: an integer
      output_depth: an integer
      num_heads: an integer dividing total_key_depth and total_value_depth
      dropout_rate: a floating point number
      name: an optional string.

    Returns:
      The result of the attention transformation. The output shape is
          [batch_size, length_q, hidden_dim]

    Raises:
      ValueError: if the key depth or value depth are not divisible by the
        number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))

    with tf.variable_scope(name, values=[query_antecedent]):

        q, k, v = compute_qkv(query_antecedent, total_key_depth, total_value_depth)

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head ** -0.5

        x = dot_product_attention(q, k, v, bias, dropout_rate)

        x = combine_heads(x)

        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

        x = tf.layers.dense(x, output_depth, use_bias=False, name="output_transform")

        return x
