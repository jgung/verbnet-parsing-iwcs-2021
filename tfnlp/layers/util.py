import numpy as np
import tensorflow as tf

from tfnlp.layers.layers import numpy_orthogonal_matrix


def get_shape(tensor):
    shape = []
    dynamic_shape = tf.shape(tensor)
    static_shape = tensor.get_shape().as_list()
    for i in range(len(static_shape) - 1):
        shape.append(dynamic_shape[i])
    shape.append(static_shape[-1])
    return shape


def select_logits(logits, targets, n_steps):
    _one_hot = tf.one_hot(targets, n_steps)  # (b x n x n) one-hot Tensor
    # don't include gradients for arcs that aren't predicted
    # (b x n x r x n) (b x n x n x 1)
    # a.k.a. b*n (r x n) (n, 1) matrix multiplications
    _select_logits = tf.matmul(logits, tf.expand_dims(_one_hot, -1))  # (b x n x r x 1)
    _select_logits = tf.squeeze(_select_logits, axis=-1)  # (b x n x r)
    return _select_logits


def mlp(inputs, inputs_shape, dropout_rate, output_size, training, name, n_splits=1):
    def _leaky_relu(features):
        return tf.nn.leaky_relu(features, alpha=0.1)

    def _dropout(_inputs):
        size = _inputs.get_shape().as_list()[-1]
        dropped = tf.layers.dropout(_inputs, dropout_rate, noise_shape=[inputs_shape[0], 1, size], training=training)
        dropped = tf.reshape(dropped, [-1, size])
        return dropped

    inputs = _dropout(inputs)

    initializer = None
    if training and not tf.get_variable_scope().reuse:
        # initialize each split of the MLP into an individual orthonormal matrix
        mat = numpy_orthogonal_matrix([inputs_shape[-1], output_size])
        mat = np.concatenate([mat] * n_splits, axis=1)
        initializer = tf.constant_initializer(mat)

    output_size *= n_splits

    result = tf.layers.dense(inputs, output_size, activation=_leaky_relu, kernel_initializer=initializer, name=name)
    result = tf.reshape(result, [inputs_shape[0], inputs_shape[1], output_size])
    result = _dropout(result)

    if n_splits == 1:
        return result
    return tf.split(result, num_or_size_splits=n_splits, axis=1)


def bilinear(input1, input2, output_size, timesteps, include_bias1=True, include_bias2=True):
    """
    Compute bilinear attention as described in 'Deep Biaffine Attention for Neural Dependency Parsing' (Dozat and Manning, 2017).
    :param input1: (bn x d1) `Tensor`
    :param input2: (bn x d2) `Tensor`
    :param output_size: number of outputs
    :param timesteps: number of timesteps in input, n
    :param include_bias1: if `True`, make first transformation affine
    :param include_bias2: if `True`, add biases to both linear transformations
    :return: bilinear mapping (b x n x r x n) `Tensor` between `input1` and `input2`, or (b x n x n) if `output_size` == 1
    """

    def _add_bias(_input):
        batches_and_tokens = tf.shape(_input)[0]
        bias = tf.ones([batches_and_tokens, 1])
        return tf.concat([_input, bias], -1)

    if include_bias1:
        # (bn x d) -> (bn x d+1)
        input1 = _add_bias(input1)
    if include_bias2:
        input2 = _add_bias(input2)

    input1dim = input1.get_shape()[-1]
    input2dim = input2.get_shape()[-1]

    weights = tf.get_variable("weights", shape=[input1dim, output_size, input2dim], initializer=tf.orthogonal_initializer)

    # (d x r x d) -> (d x rd)
    weights = tf.reshape(weights, [input1dim, -1])
    # (b n x d) (d x r d) -> b n x r d
    linear_mapping = tf.matmul(input1, weights)
    # (bn x rd) -> (b x nr x d)
    linear_mapping = tf.reshape(linear_mapping, [-1, output_size * timesteps, input2dim])
    # (b x n x d)
    input2 = tf.reshape(input2, [-1, timesteps, input2dim])
    # (b x nr x d) (b x n x d)T -> (b x n x rn) (batch matrix multiplication)
    bilinear_result = tf.matmul(linear_mapping, input2, transpose_b=True)

    if output_size != 1:
        # (b x n x rn) -> (b x n x r x n)
        bilinear_result = tf.reshape(bilinear_result, [-1, timesteps, output_size, timesteps])

    return bilinear_result
