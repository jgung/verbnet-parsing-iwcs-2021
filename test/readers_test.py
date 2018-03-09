import unittest

import pkg_resources

from tfnlp.common.constants import LABEL_KEY
from tfnlp.readers import conll_2003_reader


class TestChunk(unittest.TestCase):

    def test_reader_single(self):
        filepath = pkg_resources.resource_filename(__name__, "resources/conll03-test.txt")
        instances = list(conll_2003_reader().read_file(filepath))
        self.assertEqual(3, len(instances))
        self.assertEqual(["O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O"], instances[0][LABEL_KEY])
        self.assertEqual(["B-LOC", "I-LOC", "O"], instances[1][LABEL_KEY])
