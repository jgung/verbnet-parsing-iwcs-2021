import tensorflow as tf


def make_dataset(extractor, paths, batch_size=128):
    if type(paths) not in [list, tuple]:
        paths = [paths]

    dataset = tf.data.TFRecordDataset(paths) \
        .map(extractor.parse) \
        .shuffle(buffer_size=100000) \
        .padded_batch(batch_size,
                      padded_shapes=extractor.get_shapes(),
                      padding_values=extractor.get_padding())
    return dataset


def prepare_dataset_iterator(extractor, paths, batch_size=128):
    train_ds = make_dataset(extractor, paths, batch_size=batch_size)
    return tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
