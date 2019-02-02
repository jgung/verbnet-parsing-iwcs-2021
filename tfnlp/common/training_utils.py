import tensorflow as tf


def smoothed_labels(label_smoothing, dtype, onehot_labels):
    """
    Implements label smoothing, workaround for https://github.com/tensorflow/tensorflow/issues/24397.
    :param label_smoothing: epsilon for label smoothing, typical value would be 0.1
    :param dtype: type of logits to cast onehot_labels to
    :param onehot_labels: one-hot encoding of labels
    :return: smoothed labels
    """
    num_classes = tf.cast(tf.shape(onehot_labels)[-1], dtype)
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = label_smoothing / num_classes
    onehot_labels = onehot_labels * smooth_positives + smooth_negatives
    return onehot_labels


def assign_ema_weights(ema):
    moving_avg_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
    return tf.group(*[tf.assign(x, ema.average(x)) for x in moving_avg_variables])
