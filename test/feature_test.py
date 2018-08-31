import tempfile
import unittest

import pkg_resources
from tensorflow import Session

from tfnlp.common.config import get_feature_extractor
from tfnlp.common.constants import CHAR_KEY, LABEL_KEY, LENGTH_KEY, WORD_KEY
from tfnlp.common.utils import read_json
from tfnlp.feature import Feature, FeatureExtractor, LengthFeature, SequenceFeature, SequenceListFeature


def test_extractor():
    num_feature = Feature(LABEL_KEY, LABEL_KEY)
    len_feature = LengthFeature(WORD_KEY)
    word_feature = SequenceFeature(WORD_KEY, WORD_KEY)
    char_feature = SequenceListFeature(CHAR_KEY, WORD_KEY)
    extractor = FeatureExtractor([word_feature, char_feature, len_feature], [num_feature])
    extractor.train()
    return extractor


class TestFeature(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.extractor = test_extractor()
        self.sentence = {LABEL_KEY: '0', WORD_KEY: "the cat sat on the mat".split()}
        self.other_sentence = {LABEL_KEY: '1', WORD_KEY: "the foo".split()}

    def test_scalar(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(4, feats.context.feature[LABEL_KEY].int64_list.value[0])

    def test_length(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(6, feats.context.feature[LENGTH_KEY].int64_list.value[0])

    def test_sequence(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual([[4], [5], [6], [7], [4], [8]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])

    def test_sequence_list(self):
        feats = self.extractor.extract(self.sentence)
        char_feature = self.extractor.feature(CHAR_KEY)
        padding = (char_feature.max_len - 3) * [char_feature.pad_index]

        self.assertEqual(6, len(feats.feature_lists.feature_list[CHAR_KEY].feature))
        self.assertEqual([4, 5, 6] + padding, feats.feature_lists.feature_list[CHAR_KEY].feature[0].int64_list.value)
        self.assertEqual([12, 8, 4] + padding, feats.feature_lists.feature_list[CHAR_KEY].feature[5].int64_list.value)

    def test_not_train(self):
        self.extractor.extract(self.sentence)
        self.extractor.test()
        feats = self.extractor.extract(self.other_sentence)
        word_feature = self.extractor.feature(WORD_KEY)
        self.assertEqual([[4], [word_feature.unknown_index]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])

    def test_parse(self):
        example = self.extractor.extract(self.sentence)
        result = self.extractor.parse(example.SerializeToString())

        char_feature = self.extractor.feature(CHAR_KEY)
        self.assertEqual(4, len(result))
        self.assertEqual(char_feature.max_len, result[CHAR_KEY].shape.dims[1].value)
        with Session():
            result[CHAR_KEY].eval()
            result[WORD_KEY].eval()
            result[LABEL_KEY].eval()
            result[LENGTH_KEY].eval()

    def test_read_vocab(self):
        dirpath = pkg_resources.resource_filename(__name__, "resources/vocab/word.txt")
        word_feature = self.extractor.feature(WORD_KEY)
        word_feature.read_vocab(dirpath)
        self.assertEqual(5, len(word_feature.indices))
        self.assertEqual("the", word_feature.index_to_feat(0))
        self.assertEqual("mat", word_feature.index_to_feat(4))

    def test_write_and_read_vocab(self):
        self.extractor.extract(self.sentence)
        file = tempfile.NamedTemporaryFile()
        word_feature = self.extractor.feature(WORD_KEY)
        word_feature.write_vocab(file.name, overwrite=True)
        word_feature.read_vocab(file.name)
        self.assertEqual(9, len(word_feature.indices))
        self.assertEqual("mat", word_feature.index_to_feat(8))

    def test_read_config(self):
        configpath = pkg_resources.resource_filename(__name__, "resources/feats.json")
        config = read_json(configpath)
        extractor = get_feature_extractor(config.features)
        extractor.train()
        feats = extractor.extract(self.sentence)
        self.assertEqual(6, len(feats.feature_lists.feature_list[CHAR_KEY].feature))
        self.assertEqual([[4], [5], [6], [7], [4], [8]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])
        self.assertEqual([[1]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[LABEL_KEY].feature])
