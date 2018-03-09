import tempfile
import unittest

import pkg_resources
from tensorflow import Session

from tfnlp.feature import Feature, FeatureExtractor, SequenceFeature, SequenceListFeature

WORD_KEY = "word"
CHAR_KEY = "char"
NUM_KEY = "num"


class TestFeature(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.sentence = {NUM_KEY: '0', WORD_KEY: "the cat sat on the mat".split()}
        self.other_sentence = {NUM_KEY: '1', WORD_KEY: "the foo".split()}
        self.num_feature = Feature(NUM_KEY, NUM_KEY)
        self.word_feature = SequenceFeature(WORD_KEY, WORD_KEY)
        self.char_feature = SequenceListFeature(CHAR_KEY, WORD_KEY)
        self.extractor = FeatureExtractor([self.word_feature, self.char_feature, self.num_feature])
        self.extractor.train()

    def test_scalar(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(4, feats.context.feature[NUM_KEY].int64_list.value[0])

    def test_sequence(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual([[4], [5], [6], [7], [4], [8]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])

    def test_sequence_list(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(6, len(feats.feature_lists.feature_list[CHAR_KEY].feature))
        padding = (self.char_feature.max_len - 3) * [self.char_feature.pad_index]
        self.assertEqual([4, 5, 6] + padding, feats.feature_lists.feature_list[CHAR_KEY].feature[0].int64_list.value)
        self.assertEqual([12, 8, 4] + padding, feats.feature_lists.feature_list[CHAR_KEY].feature[5].int64_list.value)

    def test_not_train(self):
        self.extractor.extract(self.sentence)
        self.extractor.test()
        feats = self.extractor.extract(self.other_sentence)
        self.assertEqual([[4], [self.word_feature.unknown_index]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])

    def test_parse(self):
        example = self.extractor.extract(self.sentence)
        result = self.extractor.parse(example.SerializeToString())
        self.assertEqual(3, len(result))
        self.assertEqual(self.char_feature.max_len, result[CHAR_KEY].shape.dims[1].value)
        with Session():
            result[CHAR_KEY].eval()
            result[WORD_KEY].eval()
            result[NUM_KEY].eval()

    def test_read_vocab(self):
        dirpath = pkg_resources.resource_filename(__name__, "resources/vocab/word.txt")
        self.word_feature.read_vocab(dirpath)
        self.assertEqual(5, len(self.word_feature.indices))
        self.assertEqual("the", self.word_feature.index_to_feat(0))
        self.assertEqual("mat", self.word_feature.index_to_feat(4))

    def test_write_and_read_vocab(self):
        self.extractor.extract(self.sentence)
        file = tempfile.NamedTemporaryFile()
        self.word_feature.write_vocab(file.name)
        self.word_feature.read_vocab(file.name)
        self.assertEqual(9, len(self.word_feature.indices))
        self.assertEqual("mat", self.word_feature.index_to_feat(8))
