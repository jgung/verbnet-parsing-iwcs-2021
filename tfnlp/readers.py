import fnmatch
import os
import re
from collections import defaultdict

from tensorflow.python.lib.io import file_io

from tfnlp.common.chunk import chunk, convert_conll_to_bio
from tfnlp.common.constants import CHUNK_KEY, DEPREL_KEY, FEAT_KEY, HEAD_KEY, ID_KEY, INSTANCE_INDEX, LABEL_KEY, LEMMA_KEY, \
    MARKER_KEY, NAMED_ENTITY_KEY, PARSE_KEY, PDEPREL_KEY, PFEAT_KEY, PHEAD_KEY, PLEMMA_KEY, POS_KEY, PPOS_KEY, PREDICATE_KEY, \
    ROLESET_KEY, SENTENCE_INDEX, WORD_KEY


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
        self._sentence_count = 0

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
        with file_io.FileIO(path, 'r') as lines:
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
        # noinspection PyTypeChecker
        sentence[SENTENCE_INDEX] = self._sentence_count
        self._sentence_count += 1
        return sentence


class ConllDepReader(ConllReader):
    def __init__(self, index_field_map, line_filter=lambda line: False, label_field=None):
        super().__init__(index_field_map=index_field_map, line_filter=line_filter, label_field=label_field,
                         chunk_func=lambda x: x)

    def read_instances(self, rows):
        instances = [self.read_fields(rows)]
        for instance in instances:
            instance[HEAD_KEY] = [int(x) for x in instance[HEAD_KEY]]
            if self.label_field is not None:
                instance[LABEL_KEY] = instance[self.label_field][:]
            # add root
            for key, val in instance.items():
                if key == SENTENCE_INDEX:
                    continue
                if key == HEAD_KEY:
                    val.insert(0, 0)
                else:
                    val.insert(0, '<ROOT>')

        return instances


class ConllSrlReader(ConllReader):
    def __init__(self,
                 index_field_map,
                 pred_start,
                 pred_end=0,
                 pred_key="predicate",
                 chunk_func=lambda x: x,
                 line_filter=lambda line: False):
        """
        Construct an CoNLL reader for SRL.
        :param index_field_map: map from indices to corresponding fields
        :param pred_start: first column index containing semantic roles
        :param pred_end: number of columns following semantic role columns (e.g. 1, if there is only a single extra column)
        :param pred_key: prediction key
        :param chunk_func: function applied to IOB labeling to get final chunking
        :param line_filter: predicate used to identify lines in input which should not be processed
        """
        super(ConllSrlReader, self).__init__(index_field_map,
                                             line_filter=line_filter,
                                             label_field=None,
                                             chunk_func=chunk_func)
        self._pred_start = pred_start
        self._pred_end = pred_end
        self._pred_index = [key for key, val in self._index_field_map.items() if val == pred_key][0]
        self.is_predicate = lambda x: x[self._pred_index] is not '-'
        self.prop_count = 0

    def read_instances(self, rows):
        instances = []
        fields = self.read_fields(rows)
        for key, labels in self.read_predicates(rows).items():
            instance = dict(fields)  # copy instance dictionary and add labels
            instance[LABEL_KEY] = labels
            instance[MARKER_KEY] = [index == key and '1' or '0' for index in range(0, len(labels))]
            instance[INSTANCE_INDEX] = self.prop_count
            instances.append(instance)
            self.prop_count += 1
        return instances

    def read_predicates(self, rows):
        pred_indices = []
        pred_cols = defaultdict(list)
        for token_idx, row_fields in enumerate(rows):
            if self.is_predicate(row_fields):
                pred_indices.append(token_idx)
            for index in range(self._pred_start, len(row_fields) - self._pred_end):
                pred_cols[index - self._pred_start].append(row_fields[index])
        # convert from CoNLL05 labels to IOB labels
        for key, val in pred_cols.items():
            pred_cols[key] = convert_conll_to_bio(val)

        assert len(pred_indices) <= len(pred_cols), (
                'Unexpected number of predicate columns: %d instead of %d'
                ', check that predicate start and end indices are correct: %s' % (len(pred_cols), len(pred_indices), rows))
        # create predicate dictionary with keys as predicate word indices and values as corr. lists of labels (1 for each token)
        predicates = {i: pred_cols[index] for index, i in enumerate(pred_indices)}
        return predicates


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


def conll_2009_reader():
    return ConllDepReader(index_field_map={0: ID_KEY, 1: WORD_KEY, 2: LEMMA_KEY, 3: PLEMMA_KEY, 4: POS_KEY, 5: PPOS_KEY,
                                           6: FEAT_KEY, 7: PFEAT_KEY, 8: HEAD_KEY, 9: PHEAD_KEY, 10: DEPREL_KEY, 11: PDEPREL_KEY},
                          label_field=DEPREL_KEY)


def conll_2005_reader():
    return ConllSrlReader({0: WORD_KEY, 1: POS_KEY, 2: PARSE_KEY, 3: NAMED_ENTITY_KEY, 4: ROLESET_KEY, 5: PREDICATE_KEY},
                          pred_start=6)


def conll_2012_reader():
    reader = ConllSrlReader({3: WORD_KEY, 4: POS_KEY, 5: PARSE_KEY, 6: PREDICATE_KEY, 7: ROLESET_KEY},
                            pred_start=11, pred_end=1)
    reader.is_predicate = lambda line: line[6] is not '-' and line[7] is not '-'
    reader.skip_line = lambda line: line.startswith("#")  # skip comments
    return reader


def get_reader(reader_config):
    """
    Return a corpus reader for a given config.
    :param reader_config: reader configuration
    """
    if reader_config == 'conll_2003':
        return conll_2003_reader()
    elif reader_config == 'conll_2009':
        return conll_2009_reader()
    elif reader_config == 'conll_2005':
        return conll_2005_reader()
    elif reader_config == 'conll_2012':
        return conll_2012_reader()
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
        with file_io.FileIO(output_path, 'w') as out_file:
            for tagged_sent in reader.tagged_sents(id_list):
                for word, tag in filter(lambda x: x[1] != '-NONE-', tagged_sent):
                    out_file.write(word + '\t' + tag + '\n')
                out_file.write('\n')

    write_results(r'wsj/(0\d|1[0-8])/wsj_\d+\.mrg', out_dir + '/wsj-ptb-pos-train.txt')
    write_results(r'wsj/(19|20|21)/wsj_\d+\.mrg', out_dir + '/wsj-ptb-pos-dev.txt')
    write_results(r'wsj/(22|23|24)/wsj_\d+\.mrg', out_dir + '/wsj-ptb-pos-test.txt')
