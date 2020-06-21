import gzip
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io


def random_normal_initializer(_, dim):
    return np.random.normal(0, 0.01, dim)


def zero_initializer(_, dim):
    return np.zeros(dim)


def read_vectors(path, max_vecs=1000000) -> Tuple[Dict[str, np.array], int]:
    """
    Read word vectors from a specified path as float32 numpy arrays.
    :param path: path to .gz file or text file
    :param max_vecs: limit of vectors to read into memory
    :return: tuple of the vector dictionary and the vector dimensionality
    """
    vectors = OrderedDict()
    dim = 0
    with gzip.open(path, 'rt') if path.endswith('gz') else file_io.FileIO(path, 'r') as lines:
        for line in lines:
            if len(vectors) >= max_vecs:
                break
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            if dim == 0:
                dim = len(fields) - 1
            elif dim != len(fields) - 1:
                tf.logging.warn('Skipping vector with unexpected number of dimensions in line %d: %s', len(vectors), line)
                continue
            vec = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vectors[fields[0]] = vec

    return vectors, dim


def write_vectors(vectors, path):
    with file_io.FileIO(path, 'w') as lines:
        for word, vector in vectors.items():
            lines.write(word + ' ' + ' '.join([str(ele) for ele in vector]))
            lines.write('\n')


def initialize_embedding_from_dict(vector_map, dim, vocabulary, zero_init=False, standardize=False):
    """
    Initialize a numpy matrix from pre-existing vectors with indices corresponding to a given vocabulary. Words in vocabulary
    not in vectors are initialized using a given function.
    :param vector_map: dictionary from words to numpy arrays
    :param dim: dimensionality of vectors
    :param vocabulary: dictionary from words to corresponding indices
    :param zero_init: initialization function taking the word and dimensionality for words not in vector_map
    :param standardize: set word embedding values to have standard deviation of 1
    :return: numpy matrix with rows corresponding to vectors
    """
    initializer = random_normal_initializer
    if zero_init:
        initializer = zero_initializer
    emb = np.zeros([len(vocabulary), dim], dtype=np.float32)
    for word, index in vocabulary.items():
        if word not in vector_map:
            vector_map[word] = initializer(word, dim)
        emb[index] = vector_map[word]
    if standardize:
        tf.logging.info("Normalizing word embeddings")
        emb /= np.std(emb)
    return emb
