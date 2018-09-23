import unittest

import pkg_resources

from tfnlp.common.constants import LABEL_KEY
from tfnlp.readers import conll_2003_reader, conll_2005_reader, conllx_reader, MultiConllReader, ConllReader


class TestChunk(unittest.TestCase):

    def test_multi_reader(self):
        filepath = pkg_resources.resource_filename(__name__, "resources/conllx.txt")
        multireader = MultiConllReader([conllx_reader(), ConllReader({0: "pred_pos"})], ['.dep', '.pos'])
        instances = list(multireader.read_file(filepath))
        self.assertEqual(2, len(instances))
        self.assertTrue("pred_pos" in instances[0])

    def test_reader_single(self):
        filepath = pkg_resources.resource_filename(__name__, "resources/conll03-test.txt")
        instances = list(conll_2003_reader().read_file(filepath))
        self.assertEqual(3, len(instances))
        self.assertEqual(["O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O"], instances[0][LABEL_KEY])
        self.assertEqual(["B-LOC", "I-LOC", "O"], instances[1][LABEL_KEY])

    def test_srl_reader(self):
        filepath = pkg_resources.resource_filename(__name__, "resources/conll05-test.txt")
        instances = list(conll_2005_reader().read_file(filepath))
        self.assertEqual(6, len(instances))
        self.assertEqual(
            ['B-A1', 'I-A1', 'I-A1', 'I-A1', 'B-AM-MOD', 'O', 'B-V', 'B-A2', 'I-A2', 'I-A2', 'I-A2', 'B-AM-TMP', 'I-AM-TMP', 'O',
             'B-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV',
             'I-AM-ADV', 'O'], instances[0][LABEL_KEY])
        self.assertEqual(
            ['B-A1', 'I-A1', 'I-A1', 'O', 'B-V', 'B-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1',
             'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'O', 'B-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV',
             'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV',
             'I-AM-ADV', 'I-AM-ADV', 'O'], instances[1][LABEL_KEY])
        self.assertEqual(['B-A1', 'I-A1', 'I-A1', 'O', 'O', 'O', 'B-V', 'B-A4', 'I-A4', 'I-A4', 'I-A4', 'I-A4', 'B-A3', 'I-A3',
                          'I-A3', 'I-A3', 'I-A3', 'I-A3', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                          'O', 'O', 'O', 'O'], instances[2][LABEL_KEY])
        self.assertEqual(
            ['B-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'O', 'B-V', 'B-C-A1', 'I-C-A1', 'I-C-A1',
             'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1',
             'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'O'], instances[3][LABEL_KEY])
        self.assertEqual(
            ['B-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'I-A1', 'O', 'O', 'O', 'B-V', 'O', 'B-AM-ADV',
             'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV',
             'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'O'], instances[4][LABEL_KEY])
        self.assertEqual(
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-A1', 'I-A1',
             'I-A1', 'I-A1', 'B-V', 'B-AM-TMP', 'B-AM-LOC', 'I-AM-LOC', 'I-AM-LOC', 'I-AM-LOC', 'I-AM-LOC', 'O'],
            instances[5][LABEL_KEY])

    def test_srl_reader_phrases(self):
        filepath = pkg_resources.resource_filename(__name__, "resources/conll05-test.txt")
        instances = list(conll_2005_reader(phrase=True).read_file(filepath))
        self.assertEqual(6, len(instances))
        self.assertEqual(
            ['B-A1', 'I-A1', 'B-AM-MOD', 'O', 'B-V', 'B-A2', 'I-A2', 'B-AM-TMP', 'O', 'B-AM-ADV', 'I-AM-ADV', 'I-AM-ADV',
             'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'O'], instances[0][LABEL_KEY])
        self.assertEqual(
            ['B-A1', 'O', 'B-V', 'B-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'O', 'B-AM-ADV',
             'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV',
             'I-AM-ADV', 'O'], instances[1][LABEL_KEY])
        self.assertEqual(
            ['B-A1', 'O', 'O', 'O', 'B-V', 'B-A4', 'I-A4', 'B-A3', 'I-A3', 'I-A3', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O'], instances[2][LABEL_KEY])
        self.assertEqual(
            ['B-A1', 'I-A1', 'I-A1', 'I-A1', 'O', 'B-V', 'B-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1',
             'I-C-A1', 'I-C-A1', 'I-C-A1', 'I-C-A1', 'O'], instances[3][LABEL_KEY])
        self.assertEqual(
            ['B-A1', 'I-A1', 'I-A1', 'I-A1', 'O', 'O', 'O', 'B-V', 'O', 'B-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV',
             'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'I-AM-ADV', 'O'], instances[4][LABEL_KEY])
        self.assertEqual(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-A1', 'B-V', 'B-AM-TMP', 'B-AM-LOC', 'I-AM-LOC', 'O'],
                         instances[5][LABEL_KEY])
