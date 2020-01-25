from typing import List, Iterable, Union

import tensorflow as tf
from tensorflow.python.data.experimental import bucket_by_sequence_length, AUTOTUNE
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

from tfnlp.common.constants import LENGTH_KEY


def _add_uniform_noise(value, amount):
    value = tf.cast(value, dtype=tf.float32)
    noise_value = value * tf.constant(amount, dtype=tf.float32)
    noise = tf.random.uniform(shape=[], minval=-noise_value, maxval=noise_value)
    return tf.cast(value + noise, dtype=tf.int32)


def make_dataset(extractor,
                 paths: Union[str, Iterable],
                 batch_size: int = 16,
                 bucket_sizes: List[int] = None,
                 evaluate: bool = False,
                 num_parallel_calls: int = 8,
                 num_parallel_reads: int = 1,
                 max_epochs: int = -1,
                 length_noise_stdev: int = 0.1,
                 buffer_size: int = 100000,
                 batch_buffer_size: int = 512,
                 caching=True):
    if bucket_sizes is None:
        bucket_sizes = [5, 10, 25, 50, 100]
    if not isinstance(paths, Iterable):
        paths = [paths]

    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=num_parallel_reads)

    if caching:
        dataset = dataset.cache()

    if not evaluate:
        # shuffle TF records
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat(max_epochs)

    # parse serialized TF records into dictionaries of Tensors for each feature
    dataset = dataset.map(extractor.parse, num_parallel_calls=num_parallel_calls)

    # bucket dataset by sequence length, applying random noise to sequences so we don't repeat the same buckets across epochs
    dataset = dataset.apply(bucket_by_sequence_length(element_length_func=lambda elem: _add_uniform_noise(elem[LENGTH_KEY],
                                                                                                          length_noise_stdev),
                                                      bucket_boundaries=bucket_sizes,
                                                      bucket_batch_sizes=(len(bucket_sizes) + 1) * [batch_size],
                                                      padded_shapes=extractor.get_shapes(),
                                                      padding_values=extractor.get_padding()))
    if not evaluate:
        # now sort bucketed batches -- maybe not efficient, but let's ensure our training set order is really random
        dataset = dataset.shuffle(batch_buffer_size)

    # seems to be generally set to 1 or 2
    dataset = dataset.prefetch(AUTOTUNE)

    return DatasetV1Adapter(dataset)


def padded_batch(extractor, placeholder, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(placeholder)
    dataset = dataset.map(lambda x: extractor.parse(x, train=False))
    dataset = dataset.padded_batch(batch_size, extractor.get_shapes(train=False), extractor.get_padding(train=False))
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    with tf.control_dependencies([iterator.initializer]):
        return iterator.get_next()
