import os

import tensorflow as tf

from tfnlp.common.constants import END_WORD, PAD_WORD, START_WORD, UNKNOWN_WORD


def write_features(examples, out_path):
    """
    Write a list of feature instances (SequenceExamples) to a given output file as TFRecords.
    :param examples:  list of SequenceExample
    :param out_path: output path
    """
    with open(out_path, 'w') as file:
        writer = tf.python_io.TFRecordWriter(file.name)
        for example in examples:
            writer.write(example.SerializeToString())


class Extractor(object):
    """
    This class encompasses features that do not use vocabularies. This includes non-categorical features as well as
    metadata such as sequence lengths and identifiers.
    """

    def __init__(self, name, key):
        self.sequential = False
        self.name = name
        self.key = key

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
        return value

    def get_values(self, sequence):
        """
        :param sequence: dictionary of sequences for feature extraction
        :return: target(s) for feature extraction
        """
        return sequence[self.key]

    def trainable(self):
        return False


class Feature(Extractor):
    def __init__(self, name, key, train=False, indices=None, unknown_word=UNKNOWN_WORD):
        """
        This class serves as a single feature extractor and manages the associated feature vocabulary.
        :param name: unique identifier for this feature
        :param key: key used for extracting values from input instance dictionary
        :param train: if `True`, then update vocabulary upon encountering new words--otherwise, leave it unmodified
        :param indices: (optional) initial indices
        :param unknown_word: (optional) unknown word form
        """
        super(Feature, self).__init__(name=name, key=key)
        self.train = train
        self.indices = indices  # feat_to_index dict
        self.reversed = None  # index_to_feat dict
        if self.indices is None:
            self.indices = {PAD_WORD: 0, unknown_word: 1, START_WORD: 2, END_WORD: 3}

        if unknown_word not in self.indices:
            # noinspection PyTypeChecker
            self.indices[unknown_word] = len(self.indices)
        self.unknown_index = self.indices[unknown_word]

    def extract(self, instance):
        value = self.get_values(instance)
        value = self.map(value)
        index = self.feat_to_index(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))

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

    def write_vocab(self, path):
        """
        Write vocabulary as a file with a single line per entry and index 0 corresponding to the first line.
        :param path: path to file to save vocabulary
        """
        with open(path, mode='w') as vocab:
            for i in range(len(self.indices)):
                vocab.write('{}\n'.format(self.index_to_feat(i)))

    def read_vocab(self, path):
        """
        Read vocabulary from file at given path, with a single line per entry and index 0 corresponding to the first line.
        :param path: vocabulary file
        """
        self.indices = {}
        self.reversed = None
        with open(path, mode='r') as vocab:
            for line in vocab:
                line = line.strip()
                if line:
                    self.indices[line] = len(self.indices)

    def vocab_size(self):
        """
        Return the number of entries in the feature vocabulary, including OOV/padding features.
        """
        return len(self.indices)

    def _reverse(self):
        return {i: key for (key, i) in self.indices.items()}

    def trainable(self):
        return True


class SequenceFeature(Feature):

    def __init__(self, name, key, train=False, indices=None, unknown_word=UNKNOWN_WORD):
        super().__init__(name, key, train, indices, unknown_word)
        self.sequential = True

    def extract(self, sequence):
        input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[self.feat_to_index(self.map(result))]))
                          for result in self.get_values(sequence)]
        return tf.train.FeatureList(feature=input_features)


class SequenceListFeature(SequenceFeature):

    def __init__(self, name, key, max_len=20, train=False, indices=None, unknown_word=UNKNOWN_WORD, pad_word=PAD_WORD):
        super().__init__(name, key, train, indices, unknown_word)
        self.max_len = max_len

        if pad_word not in self.indices:
            # noinspection PyTypeChecker
            self.indices[pad_word] = len(self.indices)
        self.pad_index = self.indices[pad_word]

    def extract(self, sequence):
        input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=self.feat_to_index(self.map(result))))
                          for result in self.get_values(sequence)]
        return tf.train.FeatureList(feature=input_features)

    def feat_to_index(self, features):
        result = [super(SequenceListFeature, self).feat_to_index(feat) for feat in features]
        if len(result) < self.max_len:
            result += (self.max_len - len(result)) * [self.pad_index]
        else:
            result = result[:self.max_len]
        return result

    def map(self, value):
        return list(super(SequenceListFeature, self).map(value))

    def get_values(self, sequence):
        return super(SequenceListFeature, self).get_values(sequence)


class LengthFeature(Extractor):
    def map(self, value):
        return len(value)


class FeatureExtractor(object):
    def __init__(self, features):
        """
        This class encompasses multiple Feature objects, creating TFRecord-formatted instances.
        :param features: list of Features
        """
        super().__init__()
        self.features = {feature.name: feature for feature in features}

    def feature(self, name):
        return self.features[name]

    def extract(self, instance):
        """
        Extract features for a single instance as a SequenceExample.
        :param instance: input instance dictionary
        :return: TF SequenceExample
        """
        feature_list = {}
        features = {}
        for feature in self.features.values():
            feat = feature.extract(instance)
            if isinstance(feat, tf.train.FeatureList):
                feature_list[feature.name] = feat
            else:
                features[feature.name] = feat

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        features = tf.train.Features(feature=features)
        return tf.train.SequenceExample(context=features, feature_lists=feature_lists)

    def parse(self, example):
        """
        Parse a single TFRecord example into a dictionary from feature names onto corresponding Tensors.
        :param example: serialized TFRecord example
        :return: dictionary of Tensors
        """
        context_features = {}
        sequence_features = {}
        for feature in self.features.values():
            if isinstance(feature, SequenceFeature):
                if isinstance(feature, SequenceListFeature):
                    sequence_features[feature.name] = tf.FixedLenSequenceFeature([feature.max_len], dtype=tf.int64)
                else:
                    sequence_features[feature.name] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
            else:
                context_features[feature.name] = tf.FixedLenFeature([], dtype=tf.int64)

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return {**sequence_parsed, **context_parsed}

    def get_shapes(self):
        """
        Create a dictionary of TensorShapes corresponding to features. Used primarily for TF Dataset API.
        :return: dict from feature names to TensorShapes
        """
        shapes = {}
        for feature in self.features.values():
            if isinstance(feature, SequenceFeature):
                if isinstance(feature, SequenceListFeature):
                    shapes[feature.name] = tf.TensorShape([None, feature.max_len])
                else:
                    shapes[feature.name] = tf.TensorShape([None])
            else:
                shapes[feature.name] = tf.TensorShape([])
        return shapes

    def get_padding(self):
        """
        Create a dictionary of default padding values for each feature. Used primarily for TF Dataset API.
        :return: dict from feature names to padding Tensors
        """
        padding = {}
        for feature in self.features.values():
            index = 0
            if feature.trainable():
                if hasattr(feature, 'pad_index'):
                    index = feature.pad_index
                elif PAD_WORD in feature.indices:
                    index = feature.indices[PAD_WORD]
            padding[feature.name] = tf.constant(index, dtype=tf.int64)
        return padding

    def train(self, train=True):
        """
        When called, sets all features to the specified training mode.
        :param train: `True` by default
        """
        for feature in self.features.values():
            feature.train = train

    def test(self):
        """
        Set all features to test mode (do not update feature dictionaries).
        """
        self.train(False)

    def write_vocab(self, base_path):
        """
        Write vocabulary files to directory given by `base_path`. Creates base_path if it doesn't exist.
        :param base_path: base directory for vocabulary files
        """
        for key, feature in self.features.items():
            if not feature.trainable():
                continue
            path = os.path.join(base_path, key)
            parent_path = os.path.abspath(os.path.join(path, os.pardir))
            try:
                os.makedirs(parent_path)
            except OSError:
                if not os.path.isdir(parent_path):
                    raise
            feature.write_vocab(path)

    def read_vocab(self, base_path):
        """
        Read vocabulary from vocabulary files in directory given by `base_path`.
        :param base_path: base directory for vocabulary files
        """
        for key, feature in self.features.items():
            if not feature.trainable():
                continue
            path = os.path.join(base_path, key)
            feature.read_vocab(path)
