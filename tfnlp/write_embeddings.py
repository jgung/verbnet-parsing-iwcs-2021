import numpy as np

from tfnlp.common.embedding import read_vectors, write_vectors
from tfnlp.common.utils import read_json

if __name__ == '__main__':
    vectors, dim = read_vectors('data/vectors/glove.6B.100d.txt')
    print("Read %d %d-length vectors" % (len(vectors), dim))
    vn_members = read_json('data/config/experimental/mappings/vn-members.json')

    result = {}
    for cls, members in vn_members.items():
        emb = np.zeros([dim], dtype=np.float32)
        for member in members:
            if member in vectors:
                emb += vectors[member]
        result[cls] = emb
    print("Writing %d vectors" % len(result))
    write_vectors(result, 'data/vectors/vn.glove.6B.100d.txt')
