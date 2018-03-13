import unittest

import pkg_resources

from test.feature_test import test_extractor
from tfnlp.common.constants import LABEL_KEY, WORD_KEY
from tfnlp.common.embedding import read_vectors, initialize_embedding_from_dict


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.extractor = test_extractor()

    @staticmethod
    def _read_vectors():
        vectorspath = pkg_resources.resource_filename(__name__, "resources/fake-vectors.txt")
        return read_vectors(vectorspath)

    def test_read_vectors(self):
        result, dim = TestEmbedding._read_vectors()
        self.assertEqual(3, dim)
        self.assertEqual(1, result["the"][0])
        self.assertEqual(-1, result["the"][1])
        self.assertEqual(2, result["the"][2])
        self.assertEqual(3, result["sat"][0])

    def test_initialize_embedding_from_dict(self):
        sentence = {LABEL_KEY: '0', WORD_KEY: "the cat sat on the mat".split()}
        self.extractor.extract(sentence)

        result, dim = TestEmbedding._read_vectors()
        emb = initialize_embedding_from_dict(result, dim, self.extractor.features[WORD_KEY].indices)
        self.assertEqual(dim, emb.shape[1])
        self.assertEqual(1, emb[self.extractor.features[WORD_KEY].feat_to_index("the")][0])
