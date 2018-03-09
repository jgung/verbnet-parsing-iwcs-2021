import os
import tempfile
import unittest

import tensorflow as tf

from test.feature_test import CHAR_KEY, NUM_KEY, WORD_KEY, test_extractor
from tfnlp.datasets import make_dataset
from tfnlp.feature import write_features


class TestDatasets(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.sentence = {NUM_KEY: '0', WORD_KEY: "the cat sat on the mat".split()}
        self.other_sentence = {NUM_KEY: '0', WORD_KEY: "the foo".split()}
        self.extractor = test_extractor()
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
            padding = (char_feature.max_len - 3) * [char_feature.pad_index]
            self.assertEqual([4, 5, 6] + padding, next_element[CHAR_KEY][0][0].tolist())
            self.assertEqual(4, next_element[WORD_KEY][0][0])
            self.assertEqual(4, next_element[NUM_KEY][0])
