import os
import re
from itertools import chain

import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.lib.io import file_io

from tfnlp.common.constants import ELMO_KEY, END_WORD, INITIALIZER, LENGTH_KEY, PAD_WORD, SENTENCE_INDEX, START_WORD, UNKNOWN_WORD
from tfnlp.common.embedding import initialize_embedding_from_dict, read_vectors
from tfnlp.common.utils import Params, deserialize, serialize
from tfnlp.layers.reduce import ConvNet

LOWER = "lower"
NORMALIZE_DIGITS = "digit_norm"
CHARACTERS = "chars"
PREDICATE = "predicate"
PADDING = "pad"
CONV_PADDING = "conv"


def _get_reduce_function(config, dim, length):
    """
    Return a neural sequence reduction op given a configuration, input dimensionality, and max length.
    :param config: reduction op configuration
    :param dim: input dimensionality
    :param length: input max length
    :return: neural sequence reduction operation
    """
    if config.name == "ConvNet":
        return ConvNet(input_size=dim, kernel_size=config.kernel_size, num_filters=config.num_filters, max_length=length)
    else:
        raise AssertionError("Unexpected feature function: {}".format(config.name))


def lower(raw):
    if isinstance(raw, str):
        return raw.lower()
    return [word.lower() for word in raw]


def normalize_digits(raw):
    if isinstance(raw, str):
        return re.sub("\d", "#", raw)
    return [re.sub("\d", "#", word) for word in raw]


def characters(value):
    return list(value)


def is_predicate(value):
    return '0' if value == '-' else '1'


def _get_mapping_function(func):
    if func == LOWER:
        return lower
    elif func == NORMALIZE_DIGITS:
        return normalize_digits
    elif func == CHARACTERS:
        return characters
    elif func == PREDICATE:
        return is_predicate
    else:
        raise AssertionError("Unexpected function name: {}".format(func))


def _get_padding_function(func):
    func_type = func['type']
    if func_type == PADDING:
        val = func.get('value', PAD_WORD)
        count = func.get('count', 1)

        def padding(value):
            return count * [val] + value

        return padding
    elif func_type == CONV_PADDING:
        left = func.get('left', START_WORD)
        right = func.get('right', END_WORD)
        count = func.get('count', 1)

        def padding(value):
            return (count * [left]) + value + (count * [right])

        return padding
    raise AssertionError("Unexpected function type: {}".format(func_type))


class FeatureInitializer(Params):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if not config:
            config = {}
        # number of entries (words) in initializer (embedding) to include in vocabulary
        self.include_in_vocab = config.get('include_in_vocab', 0)
        # initialize to zero if no pre-trained embedding is provided (if False, use Gaussian normal initialization)
        self.zero_init = config.get('zero_init', False)
        # path to raw embedding file
        self.embedding = config.get('embedding')
        # name of serialized initializer after vocabulary training
        self.pkl_path = config.get('pkl_path')
        if self.embedding and not self.pkl_path:
            raise AssertionError("Missing 'pkl_path' parameter, which provides the path to the resulting serialized initializer")
        # if `True`, restrict vocabulary to entries with corresponding embeddings
        self.restrict_vocab = config.get('restrict_vocab', False)


class FeatureHyperparameters(Params):
    def __init__(self, config, feature, **kwargs):
        super().__init__(**kwargs)
        # dropout specific to this feature
        self.dropout = config.get('dropout', 0)
        # word-level dropout for this feature
        self.word_dropout = config.get('word_dropout', 0)
        # indicates whether variables for this feature should be trainable (`True`) or fixed (`False`)
        self.trainable = config.get('trainable', True)
        # dimensionality for this feature if an initializer is not provided
        self.dim = config.get('dim', 0)
        # variable initializer used to initialize lookup table for this feature (such as word embeddings)
        self.initializer = FeatureInitializer(config.get(INITIALIZER))
        reduce_func = config.get('function')
        if reduce_func:
            self.func = _get_reduce_function(reduce_func, self.dim, feature.max_len)
        self.add_group = config.get('add_group')


class FeatureConfig(Params):
    def __init__(self, feature, **kwargs):
        # name used to instantiate this feature
        super().__init__(**kwargs)
        self.name = feature.get('name')
        # key used for lookup during feature extraction
        self.key = feature.get('key')
        # count threshold for features
        self.threshold = feature.get('threshold', 0)
        # number of tokens to use for left padding
        self.left_padding = feature.get('left_padding', 0)
        # word used for left padding
        self.left_pad_word = feature.get('left_pad_word', START_WORD)
        # number of tokens to use for right padding
        self.right_padding = feature.get('right_padding', 0)
        # word used for right padding
        self.right_pad_word = feature.get('right_pad_word', END_WORD)
        # 2 most common feature rank for our NLP applications (word/token-level features)
        self.rank = feature.get('rank', 2)
        # string mapping functions applied during extraction
        self.mapping_funcs = [_get_mapping_function(mapping_func) for mapping_func in feature.get('mapping_funcs', [])]
        # padding functions applied during extraction
        self.padding_funcs = [_get_padding_function(padding_func) for padding_func in feature.get('padding_funcs', [])]
        # maximum sequence length of feature
        self.max_len = feature.get('max_len')
        # word used to replace OOV tokens
        self.unknown_word = feature.get('unknown_word', UNKNOWN_WORD)
        # padding token
        self.pad_word = feature.get('pad_word', PAD_WORD)
        # pre-initialized vocabulary
        self.indices = feature.get('indices')
        self.numeric = feature.get('numeric')
        # hyperparameters related to this feature
        config = feature.get('config', Params())
        self.config = FeatureHyperparameters(config, self)


class FeaturesConfig(object):
    def __init__(self, features):
        self.seq_feat = features.get('seq_feat', 'word')

        targets = features.get('targets')
        if not targets:
            tf.logging.warn("No 'targets' parameter provided in feature configuration--this could be an error if training.")
            self.targets = []
        else:
            self.targets = [_get_feature(target) for target in targets]

        feats = features.get('inputs')
        if not feats:
            raise AssertionError("No 'inputs' parameter provided in feature configuration, requires at least one input")
        else:
            self.inputs = [_get_feature(feature) for feature in feats]


def _get_feature(feature):
    """
    Create an individual feature from an input feature configuration.
    :param feature: feature configuration
    :return: feature
    """
    feature = FeatureConfig(feature)
    if feature.name == ELMO_KEY:
        feat = TextExtractor
    elif feature.rank == 3:
        feat = SequenceListFeature
    elif feature.rank == 2:
        feat = SequenceExtractor if feature.numeric else SequenceFeature
    elif feature.rank == 1:
        feat = Extractor if feature.numeric else Feature
    else:
        raise AssertionError("Unexpected feature rank: {}".format(feature.rank))
    return feat(**feature)


def get_feature_extractor(config):
    """
    Create a `FeatureExtractor` from a given feature configuration.
    :param config: feature configuration
    :return: feature extractor
    """
    # load default configurations
    config = FeaturesConfig(config)

    # use this feature to keep track of instance indices for error analysis
    config.inputs.append(index_feature())
    # used to establish sequence length for bucketed batching
    seq_feat = next((feat for feat in config.targets if feat.name == config.seq_feat),
                    next((feat for feat in config.inputs if feat.name == config.seq_feat), None))
    if not seq_feat:
        raise AssertionError("No sequence length feature provided with name: " + config.seq_feat)
    config.inputs.append(LengthFeature(seq_feat))

    return FeatureExtractor(features=config.inputs, targets=config.targets)


class Extractor(object):
    """
    This class encompasses features that do not use vocabularies. This includes non-categorical features as well as
    metadata such as sequence lengths and identifiers.
    """

    def __init__(self, name, key, config=None, mapping_funcs=None, padding_funcs=None, default_val=None, **kwargs):
        """
        Initialize an extractor.
        :param name: unique identifier for this feature
        :param key: key used for extracting values from input instance dictionary. Distinct from `name`, as we
        may perform different transformations to the same input to create different features
        :param config: extra parameters used during training
        :param mapping_funcs: list of functions mapping features to new values (e.g. converting to lowercase, normalizing digits)
        :param padding_funcs: list of padding functions
        """
        self.name = name
        self.key = key
        self.rank = 1
        self.config = config if config else FeatureHyperparameters(Params(), None)
        self.mapping_funcs = mapping_funcs if mapping_funcs else []
        self.padding_funcs = padding_funcs if padding_funcs else []
        self.default_val = default_val
        self.dtype = tf.int64

    def _extract_raw(self, instance):
        value = self.get_values(instance)
        value = self.map(value)
        return value

    def train(self, instance):
        """
        Update feature extractors for a given instance. Only applies to extractors with vocabularies.
        """
        pass

    def extract(self, instance):
        """
        Extracts a feature for a given instance.
        :param instance: feature extraction input
        :return: resulting extracted feature
        """
        value = self._extract_raw(instance)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def map(self, value):
        """
        Function applied to each token in a sequence, such as converting to lowercase or normalizing digits.
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
        val = sequence.get(self.key, self.default_val) if self.default_val is not None else sequence[self.key]
        for func in self.padding_funcs:
            val = func(val)
        return val

    def has_vocab(self):
        return False


class Feature(Extractor):
    def __init__(self, name, key, indices=None, pad_word=PAD_WORD, unknown_word=UNKNOWN_WORD,
                 threshold=0, **kwargs):
        """
        This class serves as a single feature extractor and manages the associated feature vocabulary.
        :param name: unique identifier for this feature
        :param key: key used for extracting values from input instance dictionary. Distinct from `name`, as we
        may perform different transformations to the same input to create different features
        :param indices: initial indices used to initialize the vocabulary
        :param pad_word: pad word form
        :param unknown_word: unknown word form
        :param threshold: minimum count of this feature to be saved in vocabulary
        """
        super(Feature, self).__init__(name=name, key=key, **kwargs)
        self._fixed_indices = indices  # indices provided in configuration file, should be guaranteed
        self.indices = indices  # feat_to_index dict
        self.reversed = None  # index_to_feat dict
        self.counts = {}
        self.unknown_word = unknown_word
        self.pad_word = pad_word
        self.reserved_words = {unknown_word, pad_word}
        self.threshold = threshold
        self.embedding = None
        self.frozen = False
        self.dtype = tf.string

    def initialize(self, indices=None, train=True):
        self.indices = indices if indices is not None else {}
        for reserved_word in self.reserved_words:
            if reserved_word in self.indices:
                continue
            if train:
                self.indices[reserved_word] = len(self.indices)
            else:
                raise AssertionError('Missing reserved word "{}" in vocabulary'.format(reserved_word))
        self.reversed = None

    def train(self, instance):
        value = self._extract_raw(instance)
        self.feat2index(value)

    def extract(self, instance):
        value = self._extract_raw(instance)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, encoding='utf-8')]))

    def feat2index(self, feat, train=True, count=True):
        """
        Extract the index of a single feature, updating the index dictionary if training.
        :param feat: feature value
        :param train: if true, update vocabulary
        :param count: include in counts
        :return: feature index
        """
        index = self.indices.get(feat)
        if index is None:
            # when the vocabulary is frozen, we continue keeping track of counts, but do not add new entries
            # this facilitates achieving a vocabulary that is an intersection between a pre-trained vocabulary (such as from word
            # embeddings) and the training data
            if train and not self.frozen:
                index = len(self.indices)
                self.indices[feat] = index
            else:
                index = self.indices[self.unknown_word]
        if train:
            self.counts[feat] = self.counts.get(feat, 0) + (1 if count else 0)
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

    def write_vocab(self, path, overwrite=False, prune=False):
        """
        Write vocabulary as a file with a single line per entry and index 0 corresponding to the first line.
        :param path: path to file to save vocabulary
        :param overwrite: overwrite previously saved vocabularies--if `False`, raises an error if there is a pre-existing file
        :param prune: if `True`, prune feature vocabulary based on counts
        """
        if not overwrite and os.path.exists(path):
            raise AssertionError("Pre-existing vocabulary file at %s. Set `overwrite` to `True` to ignore." % path)
        if prune:
            self.prune_vocab()
        with file_io.FileIO(path, mode='w') as vocab:
            for feat in self.ordered_feats():
                vocab.write('{}\n'.format(feat))

    def ordered_feats(self):
        for i in range(len(self.indices)):
            yield self.index_to_feat(i)

    def unk_index(self):
        return self.indices[self.unknown_word]

    def prune_vocab(self):
        """
        Filter vocabulary by count threshold (remove all vocabulary entries occurring less than `threshold` times, other than
        reserved words. Reinitialize this feature with the pruned vocabulary. Note, this does not preserve original indices, and
        should be performed prior to feature extraction.
        """
        # initial vocabulary with reserved words
        vocab = {} if self._fixed_indices is None else self._fixed_indices
        for reserved_word in self.reserved_words:
            if reserved_word not in vocab:
                vocab[reserved_word] = len(vocab)

        # after adding reserved words to vocab, sort vocab by counts and add to pruned vocab
        counts = [(feat, self.counts[feat]) for feat in sorted(self.counts, key=self.counts.get, reverse=True)
                  if feat not in vocab and feat not in self.reserved_words]
        for feat, count in counts:
            if feat not in self.indices:
                continue
            if count < self.threshold:
                break
            vocab[feat] = len(vocab)

        # re-initialize with pruned vocabulary
        self.initialize(vocab)

    def read_vocab(self, path):
        """
        Read vocabulary from file at given path, with a single line per entry and index 0 corresponding to the first line.
        If vocabulary is not found at the path, returns `False` in lieu of raising an exception.
        :param path: vocabulary file
        :return: `True` if vocabulary successfully read
        """
        if not file_io.file_exists(path):
            return False
        indices = {}
        with file_io.FileIO(path, mode='r') as vocab:
            for line in vocab:
                line = line.strip()
                if line:
                    if line in indices:
                        raise AssertionError('Duplicate entry in vocabulary given at {}: {}'.format(path, line))
                    indices[line] = len(indices)
        # re-initialize with vocabulary read from file
        self.initialize(indices, train=False)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 2

    def _extract_raw(self, sequence):
        return [self.map(result) for result in self.get_values(sequence)]

    def extract(self, sequence):
        raw = self._extract_raw(sequence)
        input_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[result])) for result in raw]
        return tf.train.FeatureList(feature=input_features)


class TextExtractor(Extractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 2
        self.dtype = tf.string

    def _extract_raw(self, sequence):
        return [self.map(result) for result in self.get_values(sequence)]

    def extract(self, sequence):
        raw = self._extract_raw(sequence)
        input_features = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(result, encoding='utf-8')]))
                          for result in raw]
        return tf.train.FeatureList(feature=input_features)


class SequenceFeature(Feature):
    def __init__(self, name, key, left_pad_word=START_WORD, right_pad_word=END_WORD, left_padding=0, right_padding=0, **kwargs):
        super().__init__(name, key, **kwargs)
        self.rank = 2

        self.left_pad_word, self.right_pad_word = left_pad_word, right_pad_word
        self.left_padding, self.right_padding = left_padding, right_padding

        if self.left_padding > 0:
            self.reserved_words = self.reserved_words | {left_pad_word}
        if self.right_padding > 0:
            self.reserved_words = self.reserved_words | {right_pad_word}

    def _extract_raw(self, sequence):
        return [self.map(result) for result in self.get_values(sequence)]

    def train(self, sequence):
        for value in self._extract_raw(sequence):
            self.feat2index(value)

    def extract(self, sequence):
        values = self._extract_raw(sequence)
        input_features = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(result, encoding='utf-8')]))
                          for result in values]
        return tf.train.FeatureList(feature=input_features)

    def map(self, value):
        return super(SequenceFeature, self).map(value)


class SequenceListFeature(SequenceFeature):

    def __init__(self, name, key, max_len=20, **kwargs):
        super().__init__(name, key, **kwargs)
        self.rank = 3

        self.max_len = max_len
        if not max_len:
            raise AssertionError("Sequence list features require \"max_len\" to be specified")

    def _extract_raw(self, sequence):
        values = super(SequenceListFeature, self)._extract_raw(sequence)

        results = []
        for vals in values:
            left_padding = self.left_padding * [self.left_pad_word]
            right_padding = self.right_padding * [self.right_pad_word]
            result = left_padding + vals + right_padding

            if len(result) < self.max_len:
                result += (self.max_len - len(result)) * [self.pad_word]
            else:
                result = result[:self.max_len]

            results.append(result)
        return results

    def train(self, sequence):
        values = self._extract_raw(sequence)
        for vals in values:
            for val in vals:
                self.feat2index(val)

    def extract(self, sequence):
        values = self._extract_raw(sequence)
        input_features = [tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[bytes(val, encoding='utf-8') for val in vals])) for vals in values
        ]
        return tf.train.FeatureList(feature=input_features)

    def map(self, value):
        mapped = super(SequenceListFeature, self).map(value)
        return mapped

    def get_values(self, sequence):
        values = super(SequenceListFeature, self).get_values(sequence)
        return values


class ConcatenatingListFeatureExtractor(SequenceListFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_values(self, sequence):
        lists = super(ConcatenatingListFeatureExtractor, self).get_values(sequence)
        result = []
        for item in lists:
            for sublist in item:
                result.append(sublist)
        return lists


class LengthFeature(Extractor):
    def __init__(self, seq_feat, name=LENGTH_KEY):
        super().__init__(name=name, key=seq_feat.key)
        self.seq_feat = seq_feat
        self.counts = {}

    def map(self, value):
        value = self.seq_feat.map(value)
        length = len(value)
        self.counts[length] = self.counts.get(length, 0) + 1
        return length

    def get_values(self, sequence):
        return self.seq_feat.get_values(sequence)


def index_feature():
    return Extractor(name=SENTENCE_INDEX, key=SENTENCE_INDEX, default_val=0)


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

    def train(self, instances):
        for instance in instances:
            for feature in self.extractors(train=True):
                feature.train(instance)

    def extract_all(self, instances, train=True):
        return [self.extract(instance, train) for instance in instances]

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
            if feature.dtype == tf.string:
                if feature.has_vocab():
                    padding[feature.name] = tf.constant(feature.pad_word, dtype=tf.string)
                else:
                    padding[feature.name] = tf.constant(b"<S>", dtype=tf.string)
            else:
                padding[feature.name] = tf.constant(0, dtype=tf.int64)
        return padding

    def initialize(self, resources=''):
        """
        Initialize feature vocabularies from pre-trained vectors if available.
        """
        for feature in self.extractors():
            if not feature.has_vocab():
                continue
            feature.initialize()

            initializer = feature.config.initializer
            if not initializer.embedding:
                continue
            num_vectors_to_read = initializer.include_in_vocab
            if num_vectors_to_read <= 0:
                continue

            vectors_path = os.path.join(resources, initializer.embedding)
            vectors, dim = read_vectors(vectors_path, max_vecs=num_vectors_to_read)
            tf.logging.info("Read %d vectors of length %d from %s", len(vectors), dim, vectors_path)
            for key in vectors:
                feature.feat2index(feature.map(key), count=False)
            if initializer.restrict_vocab:
                feature.frozen = True

    def write_vocab(self, base_path, overwrite=False, resources='', prune=False):
        """
        Write vocabulary files to directory given by `base_path`. Creates base_path if it doesn't exist.
        Creates pickled embeddings if explicit initializers are provided.
        :param base_path: base directory for vocabulary files
        :param overwrite: overwrite pre-existing vocabulary files--if `False`, raises an error when already existing
        :param resources: optional base path for embedding resources
        :param prune: if `True`, prune feature vocabularies based on counts
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

            feature.write_vocab(path, overwrite=overwrite, prune=prune)

            initializer = feature.config.initializer
            if not initializer.embedding:
                continue
            num_vectors_to_read = initializer.include_in_vocab
            if num_vectors_to_read <= 0:
                continue

            vectors_path = os.path.join(resources, initializer.embedding)
            _vectors, dim = read_vectors(vectors_path, max_vecs=num_vectors_to_read)
            vectors = {}
            for key, vector in _vectors.items():
                key = feature.map(key)
                if key not in vectors:
                    vectors[key] = vector

            tf.logging.info("Read %d vectors of length %d from %s", len(vectors), dim, vectors_path)
            feature.embedding = initialize_embedding_from_dict(vectors, dim, feature.indices, initializer.zero_init)
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
            initializer = feature.config.initializer
            try:
                if initializer.embedding:
                    feature.embedding = deserialize(in_path=base_path, in_name=initializer.pkl_path)
            except NotFoundError:
                return False
        return True


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


def get_default_buckets(lengths, min_count, max_length=None):
    """
    Apply simple heuristic to generate reasonable bucket sizes, given a dict of counts of each sequence length.
    :param lengths: sequence length count dict
    :param min_count: minimum count needed to create a bucket for a length (e.g. batch size, or some multiple of this)
    :param max_length: max-sized bucket to consider
    :return: list of bucket sizes
    """
    total = 0
    buckets = []
    for length, count in lengths.items():
        if max_length and length > max_length:
            continue
        if count + total >= min_count:  # if there are enough seqs w/ this length, create a bucket
            buckets.append(length)
            total = 0
        else:
            total += count
    if total > 0:
        buckets.append(max_length if max_length else max(lengths.keys()))
    return buckets
