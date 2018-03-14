import tensorflow as tf

from tfnlp.common.constants import LENGTH_KEY
from tfnlp.layers.grouping import bucket_by_sequence_length


def make_dataset(extractor, paths, batch_size=16, bucket_sizes=None, evaluate=False):
    if bucket_sizes is None:
        bucket_sizes = [5, 10, 25, 50, 100]
    if type(paths) not in [list, tuple]:
        paths = [paths]

    dataset = tf.data.TFRecordDataset(paths) \
        .map(extractor.parse)

    if not evaluate:
        dataset = dataset.shuffle(buffer_size=100000)

    dataset = bucket_by_sequence_length(element_length_func=lambda elem: tf.cast(elem[LENGTH_KEY], dtype=tf.int32),
                                        bucket_boundaries=bucket_sizes,
                                        bucket_batch_sizes=(len(bucket_sizes) + 1) * [batch_size],
                                        padded_shapes=extractor.get_shapes(),
                                        padding_values=extractor.get_padding())(dataset)
    if not evaluate:
        dataset = dataset.repeat()

    return dataset


def prepare_dataset_iterator(extractor, paths, batch_size=16):
    train_ds = make_dataset(extractor, paths, batch_size=batch_size)
    return tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
