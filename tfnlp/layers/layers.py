import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper

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
        _cell = tf.nn.rnn_cell.LSTMCell(params.config.state_size, name=name)
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
    if training:
        if feature.embedding is not None:
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
