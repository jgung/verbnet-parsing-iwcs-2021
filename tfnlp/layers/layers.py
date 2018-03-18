import tensorflow as tf


def input_layer(features, feature_configs, training):
    """
    Declare an input layers that compose multiple features into a single tensor across multiple time steps.
    :param features: input dictionary from feature names to 3D/4D Tensors
    :param feature_configs: feature configurations/metadata
    :param training: true if training
    :return: 3D tensor ([batch_size, time_steps, input_dim])
    """
    inputs = []
    for feature_config in feature_configs.values():
        if feature_config.has_vocab():
            inputs.append(_get_input(features[feature_config.name], feature_config, training))
    return tf.concat(inputs, -1, name="inputs")


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

    return result
