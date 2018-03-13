import math

import tensorflow as tf


class ConvNet(object):
    def __init__(self, input_size, kernel_size, num_filters, max_length):
        """
        Initialize 1D CNN with max-over-time pooling reduction op.
        :param input_size: input/channel dimensionality
        :param kernel_size: size of 1D window, length of first dimension of each filter
        :param num_filters: number of [kernel_size, num_channels] filters, the dimensionality of output
        :param max_length: maximum length of 3rd dimension of input tensors
        """
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.filters = num_filters
        self.sequence_length = max_length

    def apply(self, tensor):
        return self.max_over_time_pooling_cnn(tensor=tensor, input_size=self.input_size, sequence_length=self.sequence_length,
                                              num_filters=self.filters, kernel_size=self.kernel_size)

    @staticmethod
    def max_over_time_pooling_cnn(tensor, input_size, sequence_length, num_filters, kernel_size):
        """
        Return a 1D CNN with max-over-time pooling.
        :param input_size: channel dimensionality
        :param tensor: 4D input tensor: [batch_size, time_steps, sequence_length, num_channels]
        :param sequence_length: number of time steps (3rd dimension of input tensor)
        :param num_filters: number of [kernel_size, num_channels] filters, the dimensionality of output
        :param kernel_size: size of 1D window, length of first dimension of each filter
        :return: 3D tensor [batch_size, time_steps, filters]
        """
        shape = tf.shape(tensor)
        # flatten sequences for input
        tensor = tf.reshape(tensor, shape=[-1, sequence_length, input_size])

        # initializer = None
        limit = math.sqrt(3.0 / num_filters)
        initializer = tf.random_uniform_initializer(-limit, limit)
        tensor = tf.layers.conv1d(tensor, filters=num_filters, kernel_size=kernel_size, activation=tf.nn.relu,
                                  kernel_initializer=initializer)
        tensor = tf.layers.max_pooling1d(tensor, pool_size=sequence_length - kernel_size + 1, strides=1)
        tensor = tf.reshape(tensor, shape=[-1, shape[1], num_filters])
        return tensor
