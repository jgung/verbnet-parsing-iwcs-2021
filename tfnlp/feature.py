import os
from itertools import chain

import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.lib.io import file_io

from tfnlp.common.constants import END_WORD, INITIALIZER, LENGTH_KEY, PAD_WORD, SENTENCE_INDEX, START_WORD, UNKNOWN_WORD
from tfnlp.common.embedding import initialize_embedding_from_dict, read_vectors
from tfnlp.common.utils import Params, deserialize, serialize


def write_features(examples, out_path):
    """
    Write a list of feature instances (SequenceExamples) to a given output file as TFRecords.
    :param examples:  list of SequenceExample
    :param out_path: output path
    """
    with file_io.FileIO(out_path, 'w') as file:
        writer = tf.python_io.TFRecordWriter(file.name)
        for example in examples:
            writer.write(example.SerializeToString())


class Extractor(object):
    """
    This class encompasses features that do not use vocabularies. This includes non-categorical features as well as
    metadata such as sequence lengths and identifiers.
    """

    def __init__(self, name, key, config=None, mapping_funcs=None, **kwargs):
        """
        Initialize an extractor.
        :param name: unique identifier for this feature
        :param key: key used for extracting values from input instance dictionary
        :param config: extra parameters used during training
        :param mapping_funcs: list of functions mapping features to new values
        """
        self.name = name
        self.key = key
        self.rank = 1
        self.config = config if config else Params()
        self.mapping_funcs = mapping_funcs if mapping_funcs else []
        self.dtype = tf.int64

    def extract(self, instance):
        """
        Extracts a feature for a given instance.
        :param instance: feature extraction input
        :return: resulting extracted feature
        """
        value = self.get_values(instance)
        value = self.map(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def map(self, value):
        """
        Function applied to each token in a sequence.
        :param value: input token
        :return: transformed value
        """
        for func in self.mapping_funcs:
            value = func(value)
        return value

    def get_values(self, sequence):
        """
        :param sequence: dictionary of sequences for feature extraction
        :return: target(s) for feature extraction
        """
        return sequence[self.key]

    def has_vocab(self):
        return False


class Feature(Extractor):
    def __init__(self, name, key, config=None, train=False, indices=None, unknown_word=UNKNOWN_WORD,
                 mapping_funcs=None, **kwargs):
        """
        This class serves as a single feature extractor and manages the associated feature vocabulary.
        :param name: unique identifier for this feature
        :param key: key used for extracting values from input instance dictionary
        :param train: if `True`, then update vocabulary upon encountering new words--otherwise, leave it unmodified
        :param indices: (optional) initial indices
        :param unknown_word: (optional) unknown word form
        :param mapping_funcs: list of functions mapping features to new values
        """
        super(Feature, self).__init__(name=name, key=key, config=config, mapping_funcs=mapping_funcs, **kwargs)
        self.train = train
        self.indices = indices  # feat_to_index dict
        self.reversed = None  # index_to_feat dict
        if self.indices is None:
            self.indices = {PAD_WORD: 0, unknown_word: 1, START_WORD: 2, END_WORD: 3}

        if unknown_word not in self.indices:
            self.indices[unknown_word] = len(self.indices)
        self.unknown_index = self.indices[unknown_word]
        self.embedding = None

    def extract(self, instance):
        value = self.get_values(instance)
        value = self.map(value)
        index = self.feat_to_index(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))

    def map(self, value):
        return super(Feature, self).map(value)

    def feat_to_index(self, feat):
        """
        Extract the index of a single feature, updating the index dictionary if training.
        :param feat: feature value
        :return: feature index
        """
        index = self.indices.get(feat)
        if index is None:
            if self.train:
                index = len(self.indices)
                self.indices[feat] = index
            else:
                index = self.unknown_index
        return index

    def index_to_feat(self, index):
        """
        Returns the feature for a corresponding index. Errors out if feature is not in vocabulary.
        :param index: index in feature vocab
        :return: corresponding feature
        """
        if not self.reversed:
            self.reversed = self._reverse()
        return self.reversed[index]

    def write_vocab(self, path, overwrite=False):
        """
        Write vocabulary as a file with a single line per entry and index 0 corresponding to the first line.
        :param path: path to file to save vocabulary
        :param overwrite: overwrite previously saved vocabularies--if `False`, raises an error if there is a pre-existing file
        """
        if not overwrite and os.path.exists(path):
            raise AssertionError("Pre-existing vocabulary file at %s. Set `overwrite` to `True` to ignore." % path)
        with file_io.FileIO(path, mode='w') as vocab:
            for i in range(len(self.indices)):
                vocab.write('{}\n'.format(self.index_to_feat(i)))

    def read_vocab(self, path):
        """
        Read vocabulary from file at given path, with a single line per entry and index 0 corresponding to the first line.
        :param path: vocabulary file
        :return: `True` if vocabulary successfully read
        """
        self.indices = {}
        self.reversed = None
        try:
            with file_io.FileIO(path, mode='r') as vocab:
                for line in vocab:
                    line = line.strip()
                    if line:
                        self.indices[line] = len(self.indices)
        except NotFoundError:
            return False
        return True

    def vocab_size(self):
        """
        Return the number of entries in the feature vocabulary, including OOV/padding features.
        """
        return len(self.indices)

    def has_vocab(self):
        return True

    def _reverse(self):
        return {i: key for (key, i) in self.indices.items()}


class SequenceExtractor(Extractor):
    def __init__(self, name, key, config=None, mapping_funcs=None, **kwargs):
        super().__init__(name, key, config, mapping_funcs, **kwargs)
        self.rank = 2

    def extract(self, sequence):
        input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[self.map(result)]))
                          for result in self.get_values(sequence)]
        return tf.train.FeatureList(feature=input_features)


class TextExtractor(Extractor):
    def __init__(self, name, key, config=None, mapping_funcs=None, **kwargs):
        super().__init__(name, key, config, mapping_funcs, **kwargs)
        self.rank = 2
        self.dtype = tf.string

    def extract(self, sequence):
        input_features = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(self.map(result), encoding='utf-8')]))
                          for result in self.get_values(sequence)]
        return tf.train.FeatureList(feature=input_features)

    def map(self, value):
        return value


class SequenceFeature(Feature):

    def __init__(self, name, key, config=None, train=False, indices=None, unknown_word=UNKNOWN_WORD, **kwargs):
        super().__init__(name, key, config=config, train=train, indices=indices, unknown_word=unknown_word, **kwargs)
        self.rank = 2

    def extract(self, sequence):
        input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[self.feat_to_index(self.map(result))]))
                          for result in self.get_values(sequence)]
        return tf.train.FeatureList(feature=input_features)

    def map(self, value):
        return super(SequenceFeature, self).map(value)


class SequenceListFeature(SequenceFeature):

    def __init__(self, name, key, config=None,
                 max_len=20, train=False, indices=None, unknown_word=UNKNOWN_WORD, pad_word=PAD_WORD,
                 left_padding=0,
                 right_padding=0,
                 left_pad_word=START_WORD,
                 right_pad_word=END_WORD,
                 **kwargs):
        super().__init__(name, key, config=config, train=train, indices=indices, unknown_word=unknown_word, **kwargs)
        self.rank = 3
        self.max_len = max_len

        if pad_word not in self.indices:
            self.indices[pad_word] = len(self.indices)
        if left_pad_word not in self.indices:
            self.indices[left_pad_word] = len(self.indices)
        if right_pad_word not in self.indices:
            self.indices[right_pad_word] = len(self.indices)

        self.pad_index = self.indices[pad_word]
        self.start_index = self.indices[left_pad_word]
        self.end_index = self.indices[right_pad_word]
        self.left_padding = left_padding
        self.right_padding = right_padding

    def extract(self, sequence):
        input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=self.feat_to_index(self.map(result))))
                          for result in self.get_values(sequence)]
        return tf.train.FeatureList(feature=input_features)

    def feat_to_index(self, features):
        result = [super(SequenceListFeature, self).feat_to_index(feat) for feat in features]
        result = self.left_padding * [self.start_index] + result + self.right_padding * [self.end_index]
        if len(result) < self.max_len:
            result += (self.max_len - len(result)) * [self.pad_index]
        else:
            result = result[:self.max_len]
        return result

    def map(self, value):
        value = super(SequenceListFeature, self).map(value)
        return list(value)

    def get_values(self, sequence):
        return super(SequenceListFeature, self).get_values(sequence)


class LengthFeature(Extractor):
    def __init__(self, key, name=LENGTH_KEY):
        super().__init__(name=name, key=key)

    def map(self, value):
        value = super(LengthFeature, self).map(value)
        return len(value)


def index_feature():
    return Extractor(name=SENTENCE_INDEX, key=SENTENCE_INDEX)


class FeatureExtractor(object):
    def __init__(self, features, targets=None):
        """
        This class encompasses multiple Feature objects, creating TFRecord-formatted instances.
        :param features: list of Features
        :param targets: list of targets
        """
        super().__init__()
        self.features = {feature.name: feature for feature in features}
        self.targets = {target.name: target for target in targets} if targets else {}

    def extractors(self, train=True):
        if train:
            return chain(self.features.values(), self.targets.values())
        return self.features.values()

    def feature(self, name):
        return self.features[name]

    def target(self, name):
        return self.targets[name]

    def extract(self, instance, train=True):
        """
        Extract features for a single instance as a SequenceExample.
        :param instance: input instance dictionary
        :param train: extract targets if `True`
        :return: TF SequenceExample
        """
        feature_list = {}
        features = {}
        for feature in self.extractors(train):
            feat = feature.extract(instance)
            if isinstance(feat, tf.train.FeatureList):
                feature_list[feature.name] = feat
            else:
                features[feature.name] = feat

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        features = tf.train.Features(feature=features)
        return tf.train.SequenceExample(context=features, feature_lists=feature_lists)

    def parse(self, example, train=True):
        """
        Parse a single TFRecord example into a dictionary from feature names onto corresponding Tensors.
        :param example: serialized TFRecord example
        :param train: parse targets if `True`
        :return: dictionary of Tensors
        """
        context_features = {}
        sequence_features = {}
        for feature in self.extractors(train):
            if feature.rank == 1:
                context_features[feature.name] = tf.FixedLenFeature([], dtype=feature.dtype)
            elif feature.rank == 2:
                sequence_features[feature.name] = tf.FixedLenSequenceFeature([], dtype=feature.dtype)
            elif feature.rank == 3:
                sequence_features[feature.name] = tf.FixedLenSequenceFeature([feature.max_len], dtype=feature.dtype)
            else:
                raise AssertionError("Unexpected feature rank value: {}".format(feature.rank))

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return {**sequence_parsed, **context_parsed}

    def get_shapes(self, train=True):
        """
        Create a dictionary of TensorShapes corresponding to features. Used primarily for TF Dataset API.
        :param train: include target shapes if `True`
        :return: dict from feature names to TensorShapes
        """
        shapes = {}
        for feature in self.extractors(train):
            if feature.rank == 3:
                shapes[feature.name] = tf.TensorShape([None, feature.max_len])
            elif feature.rank == 2:
                shapes[feature.name] = tf.TensorShape([None])
            elif feature.rank == 1:
                shapes[feature.name] = tf.TensorShape([])
            else:
                raise AssertionError("Unexpected feature rank value: {}".format(feature.rank))

        return shapes

    def get_padding(self, train=True):
        """
        Create a dictionary of default padding values for each feature. Used primarily for TF Dataset API.
                :param train: include target padding if `True`
        :return: dict from feature names to padding Tensors
        """
        padding = {}
        for feature in self.extractors(train):
            index = 0
            if feature.has_vocab():
                if hasattr(feature, 'pad_index'):
                    index = feature.pad_index
                elif PAD_WORD in feature.indices:
                    index = feature.indices[PAD_WORD]
                padding[feature.name] = tf.constant(index, dtype=tf.int64)
            elif feature.dtype == tf.string:
                padding[feature.name] = tf.constant(b"<S>", dtype=tf.string)
            else:
                padding[feature.name] = tf.constant(0, dtype=tf.int64)
        return padding

    def train(self, train=True):
        """
        When called, sets all features to the specified training mode.
        :param train: `True` by default
        """
        for feature in self.extractors():
            feature.train = train

    def test(self):
        """
        Set all features to test mode (do not update feature dictionaries).
        """
        self.train(False)

    def initialize(self, resources=''):
        """
        Initialize feature vocabularies from pre-trained vectors if available.
        """
        self.train()
        for feature in self.extractors():
            initializer = feature.config.get(INITIALIZER)
            if not initializer:
                continue
            num_vectors_to_read = initializer.include_in_vocab
            if not num_vectors_to_read or num_vectors_to_read <= 0:
                continue

            vectors, dim = read_vectors(resources + initializer.embedding, max_vecs=num_vectors_to_read)
            tf.logging.info("Read %d vectors of length %d from %s", len(vectors), dim, resources + initializer.embedding)
            for key in vectors:
                feature.feat_to_index(key)

    def write_vocab(self, base_path, overwrite=False, resources=''):
        """
        Write vocabulary files to directory given by `base_path`. Creates base_path if it doesn't exist.
        Creates pickled embeddings if explicit initializers are provided.
        :param base_path: base directory for vocabulary files
        :param overwrite: overwrite pre-existing vocabulary files--if `False`, raises an error when already existing
        :param resources: optional base path for embedding resources
        """
        for feature in self.extractors():
            if not feature.has_vocab():
                continue
            path = os.path.join(base_path, feature.name)
            parent_path = os.path.abspath(os.path.join(path, os.path.pardir))
            try:
                os.makedirs(parent_path)
            except OSError:
                if not os.path.isdir(parent_path):
                    raise
            feature.write_vocab(path, overwrite=overwrite)

            initializer = feature.config.get(INITIALIZER)

            if initializer:
                num_vectors_to_read = initializer.include_in_vocab
                if not num_vectors_to_read or num_vectors_to_read <= 0:
                    continue
                vectors, dim = read_vectors(resources + initializer.embedding, max_vecs=num_vectors_to_read)
                tf.logging.info("Read %d vectors of length %d from %s", len(vectors), dim, resources + initializer.embedding)
                feature.embedding = initialize_embedding_from_dict(vectors, dim, feature.indices)
                tf.logging.info("Saving %d vectors as embedding", feature.embedding.shape[0])
                serialize(feature.embedding, out_path=base_path, out_name=initializer.pkl_path, overwrite=overwrite)

    def read_vocab(self, base_path):
        """
        Read vocabulary from vocabulary files in directory given by `base_path`. Loads any pickled embeddings.
        :param base_path: base directory for vocabulary files
        :return: `True` if vocabulary was successfully read
        """
        for feature in self.extractors():
            if not feature.has_vocab():
                continue
            path = os.path.join(base_path, feature.name)
            success = feature.read_vocab(path)
            if not success:
                return False
            initializer = feature.config.get(INITIALIZER)
            try:
                if initializer:
                    feature.embedding = deserialize(in_path=base_path, in_name=initializer.pkl_path)
            except NotFoundError:
                return False
        return True
