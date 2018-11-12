import tensorflow as tf

from tfnlp.common.constants import F1_METRIC_KEY, PRECISION_METRIC_KEY, RECALL_METRIC_KEY


def calculate_precision(predictions, labels):
    mask = tf.cast(tf.not_equal(predictions, 0), tf.float32)
    matched = tf.cast(tf.equal(predictions, labels), tf.float32)
    total = tf.reduce_sum(mask)
    return tf.cond(total > 0, lambda: tf.reduce_sum(matched * mask) / total, lambda: total)


def calculate_recall(predictions, labels):
    mask = tf.cast(tf.not_equal(predictions, 0), tf.float32)
    matched = tf.cast(tf.equal(predictions, labels), tf.float32)
    non_zero = tf.cast(tf.not_equal(labels, 0), tf.float32)
    total = tf.reduce_sum(non_zero)
    return tf.cond(total > 0, lambda: tf.reduce_sum(matched * mask) / total, lambda: total)


def precision_metric_fn(predictions, labels):
    return tf.metrics.mean(calculate_precision(predictions, labels))


def recall_metric_fn(predictions, labels):
    return tf.metrics.mean(calculate_recall(predictions, labels))


def f_metric_fn(predictions, labels):
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    denom = precision + recall
    f1 = tf.cond(denom > 0, lambda: 2 * precision * recall / denom, lambda: denom)
    return tf.metrics.mean(f1)


def tagger_metrics(predictions, labels, ns=None):
    return {PRECISION_METRIC_KEY if not ns else '%s-%s' % (PRECISION_METRIC_KEY, ns): precision_metric_fn(predictions, labels),
            RECALL_METRIC_KEY if not ns else '%s-%s' % (RECALL_METRIC_KEY, ns): recall_metric_fn(predictions, labels),
            F1_METRIC_KEY if not ns else '%s-%s' % (F1_METRIC_KEY, ns): f_metric_fn(predictions, labels)}
