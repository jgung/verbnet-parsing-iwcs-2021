import unittest

from tensorflow import Session

from tfnlp.feature import Feature, FeatureExtractor, SequenceFeature, SequenceListFeature


class TestFeature(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.word_key = "word"
        self.char_key = "char"
        self.num_key = "num"
        self.sentence = {self.num_key: '0', self.word_key: "the cat sat on the mat".split()}
        self.other_sentence = {self.num_key: '1', self.word_key: "the foo".split()}
        self.num_feature = Feature(self.num_key, self.num_key)
        self.word_feature = SequenceFeature(self.word_key, self.word_key)
        self.char_feature = SequenceListFeature(self.char_key, self.word_key)
        self.extractor = FeatureExtractor([self.word_feature, self.char_feature, self.num_feature])
        self.extractor.train()

    def test_scalar(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(4, feats.context.feature[self.num_key].int64_list.value[0])

    def test_sequence(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual([[4], [5], [6], [7], [4], [8]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[self.word_key].feature])

    def test_sequence_list(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(6, len(feats.feature_lists.feature_list[self.char_key].feature))
        padding = (self.char_feature.max_len - 3) * [self.char_feature.pad_index]
        self.assertEqual([4, 5, 6] + padding, feats.feature_lists.feature_list[self.char_key].feature[0].int64_list.value)
        self.assertEqual([12, 8, 4] + padding, feats.feature_lists.feature_list[self.char_key].feature[5].int64_list.value)

    def test_not_train(self):
        self.extractor.extract(self.sentence)
        self.extractor.test()
        feats = self.extractor.extract(self.other_sentence)
        self.assertEqual([[4], [self.word_feature.unknown_index]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[self.word_key].feature])

    def test_parse(self):
        example = self.extractor.extract(self.sentence)
        result = self.extractor.parse(example.SerializeToString())
        self.assertEqual(3, len(result))
        self.assertEqual(self.char_feature.max_len, result[self.char_key].shape.dims[1].value)
        with Session():
            result[self.char_key].eval()
            result[self.word_key].eval()
            result[self.num_key].eval()
