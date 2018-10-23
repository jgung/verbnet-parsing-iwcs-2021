import os
import tempfile
import unittest

import tensorflow as tf

from test.test_feature import test_extractor
from tfnlp.common.constants import CHAR_KEY, LABEL_KEY, WORD_KEY
from tfnlp.datasets import make_dataset
from tfnlp.feature import write_features


class TestDatasets(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.sentence = {LABEL_KEY: '0', WORD_KEY: "the cat sat on the mat".split()}
        self.other_sentence = {LABEL_KEY: '0', WORD_KEY: "the foo".split()}
        self.extractor = test_extractor()
        self.extractor.write_vocab(tempfile.NamedTemporaryFile().name, prune=True)
        ex1 = self.extractor.extract(self.sentence)
        ex2 = self.extractor.extract(self.other_sentence)
        self.file = tempfile.NamedTemporaryFile()
        write_features([ex1, ex2], os.path.abspath(self.file.name))

    def test_dataset(self):
        dataset = make_dataset(self.extractor, os.path.abspath(self.file.name), batch_size=128)
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)
        training_init_op = iterator.make_initializer(dataset)
        next_element = iterator.get_next()
        with tf.Session() as sess:
            sess.run(training_init_op)
            next_element = sess.run(next_element)

            char_feature = self.extractor.feature(CHAR_KEY)
            left_padding = char_feature.left_padding * [char_feature.indices[char_feature.left_pad_word]]
            right_padding = char_feature.right_padding * [char_feature.indices[char_feature.right_pad_word]]
            pad_index = char_feature.indices.get(char_feature.pad_word)
            padding = (char_feature.max_len - 3 - len(left_padding) - len(right_padding)) * [pad_index]
            self.assertEqual(left_padding + [4, 5, 6] + right_padding + padding, next_element[CHAR_KEY][0][0].tolist())
            self.assertEqual(2, next_element[WORD_KEY][0][0])
            self.assertEqual(1, next_element[LABEL_KEY][0])
