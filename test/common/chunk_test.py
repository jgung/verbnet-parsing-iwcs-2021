import unittest

from tfnlp.common.chunk import chunk, chunk_besio, chunk_conll


class TestChunk(unittest.TestCase):

    def test_iob(self):
        chunked = chunk(["I-A", "O", "I-B", "I-C", "I-B", "O", "I-B", "B-B", "I-B", "I-B"], besio=False)
        self.assertEqual(["B-A", "O", "B-B", "B-C", "B-B", "O", "B-B", "B-B", "I-B", "I-B"], chunked)

    def test_besio(self):
        chunked = chunk_besio(["I-A", "O", "I-B", "I-C", "I-B", "O", "I-B", "B-B", "I-B", "I-B"])
        self.assertEqual(["S-A", "O", "S-B", "S-C", "S-B", "O", "S-B", "B-B", "I-B", "E-B"], chunked)

    def test_conll(self):
        chunked = chunk_conll(["I-A", "O", "I-B", "I-C", "I-B", "O", "I-B", "B-B", "I-B", "I-B"])
        self.assertEqual(["(A*)", "*", "(B*)", "(C*)", "(B*)", "*", "(B*)", "(B*", "*", "*)"], chunked)
