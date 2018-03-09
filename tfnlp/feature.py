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


class Feature(object):
    def __init__(self, name, key, train=False, indices=None, unknown_word=UNKNOWN_WORD):
        """
        This class serves as a single feature extractor and manages the associated feature vocabulary.
        :param name: unique identifier for this feature
        :param key: key used for extracting values from input instance dictionary
        :param train: if `True`, then update vocabulary upon encountering new words--otherwise, leave it unmodified
        :param indices: (optional) initial indices
        :param unknown_word: (optional) unknown word form
        """
        super(Feature, self).__init__()
        self.name = name
        self.key = key
        self.train = train
        self.indices = indices
        if self.indices is None:
            self.indices = {PAD_WORD: 0, unknown_word: 1, START_WORD: 2, END_WORD: 3}

        if unknown_word not in self.indices:
            # noinspection PyTypeChecker
            self.indices[unknown_word] = len(self.indices)

        self.unknown_index = self.indices[unknown_word]

        self.sequential = False

    def extract(self, instance):
        """
        Extracts a feature for a given instance.
        :param instance: feature extraction input
        :return: resulting extracted feature
        """
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


class FeatureExtractor(object):
    def __init__(self, features):
        """
        This class encompasses multiple Feature objects, creating TFRecord-formatted instances.
        :param features: list of Features
        """
        super().__init__()
        self.features = {feature.name: feature for feature in features}

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
        Create a dictionary of TensorShapes corresponding to features.
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
        Create a dictionary of default padding values for each feature.
        :return: dict from feature names to padding Tensors
        """
        padding = {}
        for feature in self.features.values():
            index = 0
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
