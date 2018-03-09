import os
import tempfile
import unittest

import tensorflow as tf

from tfnlp.datasets import make_dataset
from tfnlp.feature import Feature, FeatureExtractor, SequenceFeature, SequenceListFeature, write_features


class TestDatasets(unittest.TestCase):

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

            padding = (self.char_feature.max_len - 3) * [self.char_feature.pad_index]
            self.assertEqual([4, 5, 6] + padding, next_element[self.char_key][0][0].tolist())
            self.assertEqual(4, next_element[self.word_key][0][0])
            self.assertEqual(4, next_element[self.num_key][0])
