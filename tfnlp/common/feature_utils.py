import tensorflow as tf


def sequence_example(feature, feature_list):
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    features = tf.train.Features(feature=feature)
    return tf.train.SequenceExample(context=features, feature_lists=feature_lists)


def str_feature_list(vals):
    return tf.train.FeatureList(feature=[str_feature(val) for val in vals])


def int64_feature_list(vals):
    return tf.train.FeatureList(feature=[int64_feature(val) for val in vals])


def str_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(val, encoding='utf-8')]))


def int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def int64_feature_list_fill(length, value):
    return tf.train.FeatureList(feature=[int64_feature(value) for _ in range(length)])
