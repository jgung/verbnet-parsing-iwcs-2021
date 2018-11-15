import unittest

import pkg_resources

from tfnlp.common.embedding import read_vectors


class TestEmbedding(unittest.TestCase):

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
