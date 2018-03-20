import fnmatch
import os
import re
from collections import defaultdict

from tfnlp.common.chunk import chunk
from tfnlp.common.constants import CHUNK_KEY, LABEL_KEY, NAMED_ENTITY_KEY, POS_KEY, WORD_KEY


class ConllReader(object):
    def __init__(self, index_field_map, line_filter=lambda line: False, label_field=None, chunk_func=lambda x: x):
        """
        Initialize a CoNLL reader with a given map from column indices to field names.
        :param index_field_map: dictionary from column indices to field names
        :param line_filter: function that can be used to filter/ignore specific lines
        """
        super(ConllReader, self).__init__()
        self._index_field_map = index_field_map
        self.line_filter = line_filter
        self.label_field = label_field
        self.chunk_func = chunk_func

    def read_files(self, path, extension="txt"):
        """
        Read instances from a given path--if the provided path is a directory, recursively find all files with provided extension.
        :param path: directory or single file
        :param extension: extension of files in directory
        :return: CoNLL instances
        """
        if extension and os.path.isdir(path):
            for root, dir_names, file_names in os.walk(path):
                for file_name in fnmatch.filter(file_names, '*' + extension):
                    for instance in self.read_file(os.path.join(root, file_name)):
                        yield instance
        for instance in self.read_file(path):
            yield instance

    def read_file(self, path):
        """
        Read instances from a file at a given path.
        :param path: path to single CoNLL-formatted file
        :return: CoNLL instances
        """
        with open(path) as lines:
            current = []
            for line in lines:
                line = line.strip()
                if not line or self.line_filter(line):
                    if current:
                        for instance in self.read_instances([line.split() for line in current]):
                            yield instance
                        current = []
                    continue
                current.append(line)
            if current:  # read last instance if there is no newline at end of file
                for instance in self.read_instances([line.split() for line in current]):
                    yield instance

    def read_instances(self, rows):
        """
        Read a list of instances from the given list of row strings.
        :param rows: CoNLL rows
        :return: list of instances
        """
        instances = [self.read_fields(rows)]
        if self.label_field:
            for instance in instances:
                instance[LABEL_KEY] = self.chunk_func(instance[self.label_field])
        return instances

    def read_fields(self, rows):
        """
        Read a field dictionary from the given list of rows--from each field names to a list of corresponding features.
        :param rows: CoNLL rows
        :return: field dictionary, from keys to lists of tokens
        """
        sentence = defaultdict(list)
        for row in rows:
            for index, val in self._index_field_map.items():
                sentence[val].append(row[index])
        return sentence


def conll_2003_reader(chunk_func=chunk):
    """
    Initialize and return a CoNLL reader for the CoNLL-2003 NER shared task.
    :param chunk_func: chunking function (IOB2 by default)
    :return: CoNLL-2003 reader
    """
    return ConllReader(index_field_map={0: WORD_KEY, 1: POS_KEY, 2: CHUNK_KEY, 3: NAMED_ENTITY_KEY},
                       label_field=NAMED_ENTITY_KEY, chunk_func=chunk_func)


def ptb_pos_reader():
    """
    Initialize and return a CoNLL reader that reads two columns: word and part-of-speech tag.
    :return: POS reader
    """
    return ConllReader(index_field_map={0: WORD_KEY, 1: POS_KEY}, label_field=POS_KEY)


def get_reader(reader_config):
    """
    Return a corpus reader for a given config.
    :param reader_config: reader configuration
    """
    if reader_config == 'conll_2003':
        return conll_2003_reader()
    elif reader_config == 'ptb_pos':
        return ptb_pos_reader()
    else:
        raise ValueError("Unexpected reader type: " + reader_config)


def write_ptb_pos_files(wsj_path, out_dir):
    """
    Utility for reading PTB WSJ files and writing them in standard splits to a given directory.
    :param wsj_path: path to WSJ parse directory in PTB3 release
    :param out_dir: output directory to save train/test/dev files
    """
    from nltk.corpus.reader import CategorizedBracketParseCorpusReader
    reader = CategorizedBracketParseCorpusReader(wsj_path, r'(wsj/\d\d/wsj_\d\d)\d\d.mrg',
                                                 cat_file='allcats.txt', tagset='wsj')

    def write_results(expr, output_path):
        id_list = list(filter(lambda x: re.match(expr, x), reader.fileids()))
        with open(output_path, 'w') as out_file:
            for tagged_sent in reader.tagged_sents(id_list):
                for word, tag in filter(lambda x: x[1] != '-NONE-', tagged_sent):
                    out_file.write(word + '\t' + tag + '\n')
                out_file.write('\n')

    write_results(r'wsj/(0\d|1[0-8])/wsj_\d+\.mrg', out_dir + '/wsj-ptb-pos-train.txt')
    write_results(r'wsj/(19|20|21)/wsj_\d+\.mrg', out_dir + '/wsj-ptb-pos-dev.txt')
    write_results(r'wsj/(22|23|24)/wsj_\d+\.mrg', out_dir + '/wsj-ptb-pos-test.txt')
