import tensorflow as tf
from tensorflow.contrib.data import bucket_by_sequence_length, shuffle_and_repeat

from tfnlp.common.constants import LENGTH_KEY


def make_dataset(extractor, paths, batch_size=16, bucket_sizes=None, evaluate=False, num_parallel_calls=4,
                 num_parallel_reads=1, prefetch_batches=1, max_epochs=999):
    if bucket_sizes is None:
        bucket_sizes = [5, 10, 25, 50, 100]
    if type(paths) not in [list, tuple]:
        paths = [paths]

    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=num_parallel_reads)

    if not evaluate:
        dataset = shuffle_and_repeat(buffer_size=100000, count=max_epochs)(dataset)

    dataset = dataset.map(extractor.parse, num_parallel_calls=num_parallel_calls)

    dataset = dataset.apply(bucket_by_sequence_length(element_length_func=lambda elem: tf.cast(elem[LENGTH_KEY], dtype=tf.int32),
                                                      bucket_boundaries=bucket_sizes,
                                                      bucket_batch_sizes=(len(bucket_sizes) + 1) * [batch_size],
                                                      padded_shapes=extractor.get_shapes(),
                                                      padding_values=extractor.get_padding()))

    dataset = dataset.prefetch(prefetch_batches)

    return dataset
