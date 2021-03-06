import fnmatch
import os
import re
from collections import defaultdict
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tfnlp.common import constants
from tfnlp.common.chunk import chunk, convert_conll_to_bio, end_of_chunk, start_of_chunk
from tfnlp.common.constants import CHUNK_KEY, DEPREL_KEY, ENHANCED_DEPS_KEY, FEAT_KEY, HEAD_KEY, ID_KEY, INSTANCE_INDEX, \
    LABEL_KEY, LEMMA_KEY, MARKER_KEY, MISC_KEY, NAMED_ENTITY_KEY, PARSE_KEY, PDEPREL_KEY, PFEAT_KEY, PHEAD_KEY, PLEMMA_KEY, \
    POS_KEY, PPOS_KEY, PREDICATE_KEY, SENSE_KEY, SENTENCE_INDEX, TOKEN_INDEX_KEY, WORD_KEY, XPOS_KEY, PREDICATE_INDEX_KEY
from tfnlp.common.utils import Params


class TsvReader(object):

    def __init__(self, line_filter=lambda line: False):
        super(TsvReader, self).__init__()
        self.line_filter = line_filter

    def read_file(self, path):
        """
        Read instances from a file at a given path.
        :param path: path to single TSV file with labels in the first column and sentences in the second
        :return: labeled sentences
        """
        with file_io.FileIO(path, 'r') as lines:
            for line in lines:
                line = line.strip()
                if line or not self.line_filter(line):
                    line = line.split('\t')
                    yield self._process_fields(line)

    def _process_fields(self, fields):
        if len(fields) != 2:
            raise AssertionError('Incorrect number of fields (was expecting 2) in line: %s' % '\t'.join(fields))

        return {LABEL_KEY: fields[0], WORD_KEY: word_tokenize(fields[1])}


class SemlinkReader(TsvReader):
    """
    Read SemLink instances from a file at a given path.

    Should be of the following tab-separated format:

    ```
    nw/wsj/23/wsj_2303.parse 5 5 build 26.1	Hooker 's philosophy was to build and sell .
    nw/wsj/23/wsj_2303.parse 5 7 sell 13.1	Hooker 's philosophy was to build and sell .
    ```
    file<TAB>sentence_idx<TAB>token_idx<TAB>lemma<TAB>label<TAB>Space-separated inline tokens.
    """

    def _process_fields(self, fields):
        tokens = fields[1].split()
        fields = fields[0].split()
        return {LABEL_KEY: fields[4], TOKEN_INDEX_KEY: int(fields[2]), WORD_KEY: tokens}


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

    def read_files(self, path, extension=".txt"):
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
                value = row[index]
                if WORD_KEY == val:
                    value = unescape_ptb(value)
                sentence[val].append(value)
        # noinspection PyTypeChecker
        sentence[SENTENCE_INDEX] = self._sentence_count
        self._sentence_count += 1
        return sentence


class MultiConllReader(object):
    def __init__(self, readers, suffixes):
        """
        Initialize a CoNLL reader with a given map from column indices to field names.
        :param readers: individual readers
        :param suffixes: target suffixes for each reader
        """
        super(MultiConllReader, self).__init__()
        self.readers = readers
        self.suffixes = suffixes

    def read_file(self, path):
        """
        Read instances from files at a given path.
        :param path: base path to CoNLL-formatted files
        :return: CoNLL instances
        """
        files = []
        try:
            files = [file_io.FileIO(path + suffix, 'r') for suffix in self.suffixes]
            current = []
            for views in zip(*files):
                views = [line.strip() for line in views]
                blank_line = any(not line or reader.line_filter(line) for line, reader in zip(views, self.readers))
                if blank_line:
                    if current:
                        for instance in self.read_instances(current):
                            yield instance
                        current = []
                    continue
                current.append(views)
            if current:  # read last instance if there is no newline at end of file
                for instance in self.read_instances(current):
                    yield instance
        finally:
            for file in files:
                file.close()

    def read_instances(self, current):
        instances = {}
        for reader, suffix, part in zip(self.readers, self.suffixes, zip(*current)):
            instances[suffix] = reader.read_instances([line.split() for line in part])
        for instance_fields in zip(*(instances[key] for key in self.suffixes)):
            instance = defaultdict(list)
            for field in instance_fields:
                instance.update(field)
            yield instance


class ConllDepReader(ConllReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extract_invalid = False

    def read_instances(self, rows):
        instances = [self.read_fields(rows)]
        for instance in instances:
            try:
                instance[HEAD_KEY] = [int(x) for x in instance[HEAD_KEY]]
            except ValueError:
                tf.logging.warning('Unable to read instance: %s', ' '.join(instance[WORD_KEY]))
                if self.extract_invalid:
                    return [None]
                return []
            if self.label_field is not None:
                instance[LABEL_KEY] = instance[self.label_field][:]

        return instances


class ConllSrlReader(ConllReader):
    def __init__(self,
                 index_field_map,
                 pred_start,
                 pred_end=0,
                 pred_key=PREDICATE_KEY,
                 chunk_func=lambda x: x,
                 line_filter=lambda line: line.startswith("#") and not line.startswith("# #"),  # skip comments
                 label_mappings=None,
                 regex_mapping=False,
                 map_with_regex_post=False,
                 sense_mappings=None,
                 pred_filter=None,

                 sense_predictions_suffix=None):
        """
        Construct an CoNLL reader for SRL.
        :param index_field_map: map from indices to corresponding fields
        :param pred_start: first column index containing semantic roles
        :param pred_end: number of columns following semantic role columns (e.g. 1, if there is only a single extra column)
        :param pred_key: prediction key
        :param chunk_func: function applied to IOB labeling to get final chunking
        :param line_filter: predicate used to identify lines in input which should not be processed
        :param label_mappings: dict of label key onto a dictionary of mappings from input to output labels
        :param regex_mapping: if `True`, treat mappings as regular expressions, e.g. {"^([RC]-)?(\\S+)\\$(\\S+)$": "\\1\\3"}
        :param sense_mappings: dict of original sense key (with lemma) to mapped sense keys
        :param pred_filter: filter function applied to predicate lemma, skips non-matching instances
        """
        super(ConllSrlReader, self).__init__(index_field_map,
                                             line_filter=line_filter,
                                             label_field=None,
                                             chunk_func=chunk_func)
        self._pred_start = pred_start
        self._pred_end = pred_end
        self._predicate_key = pred_key
        self._pred_index = [key for key, val in self._index_field_map.items() if val == pred_key][0]
        self.is_predicate = lambda x: x[self._pred_index] is not '-'
        self.prop_count = 0
        self._label_mappings = dict(label_mappings) if label_mappings is not None else {LABEL_KEY: {}}
        if label_mappings is not None and not (regex_mapping or map_with_regex_post):
            # add continuation/reference mappings if they aren't already there
            for _target_mappings in self._label_mappings.values():
                c_mappings = {'C-' + k: v for k, v in _target_mappings.items()}
                r_mappings = {'R-' + k: v for k, v in _target_mappings.items()}
                _target_mappings.update(c_mappings)
                _target_mappings.update(r_mappings)
        self._regex_mapping = regex_mapping
        self._map_with_regex_post = map_with_regex_post
        self._sense_mappings = dict(sense_mappings) if sense_mappings else None
        self._pred_filter = pred_filter if pred_filter is not None else lambda x: True
        if sense_mappings:
            self._no_mappings = set()
            self._is_rs_mapping = re.match("^\\S+\\.\\d+$", next(iter(self._sense_mappings.keys())))

        self._sense_predictions_suffix = sense_predictions_suffix
        self._sense_predictions = {}

    def read_file(self, path):
        if self._sense_predictions_suffix is not None and file_io.file_exists(path + self._sense_predictions_suffix):
            with file_io.FileIO(path + self._sense_predictions_suffix, 'r') as lines:
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split()
                    if len(fields) < 3:
                        tf.logging.warning("Unexpected format for predicted senses: %s" % line)
                        continue
                    instance_id, _, pred = fields[:3]
                    self._sense_predictions[int(instance_id)] = pred
        for instance in super(ConllSrlReader, self).read_file(path):
            if self._sense_predictions_suffix is not None:
                if instance[constants.INSTANCE_INDEX] in self._sense_predictions:
                    instance[constants.SENSE_PRED_KEY] = self._sense_predictions[instance[constants.INSTANCE_INDEX]]
                else:
                    instance[constants.SENSE_PRED_KEY] = instance[constants.SENSE_KEY]
            yield instance

    def read_instances(self, rows):
        instances = []
        fields = self.read_fields(rows)
        for predicate_index, all_labels in self.read_predicates(rows).items():
            instance = dict(fields)  # copy instance dictionary and add labels
            for label_key, labels in all_labels.items():
                instance[label_key] = labels
            instance[PREDICATE_INDEX_KEY] = predicate_index
            instance[constants.PREDICATE_LEMMA] = instance[self._predicate_key][predicate_index]
            if not self._pred_filter(instance[constants.PREDICATE_LEMMA]):
                continue
            instance[constants.PREDICATE_FORM] = instance[WORD_KEY][predicate_index]
            instance[INSTANCE_INDEX] = self.prop_count
            instances.append(instance)
            self.prop_count += 1
            if HEAD_KEY in instance:
                instance[HEAD_KEY] = [int(x) for x in instance[HEAD_KEY]]
            if SENSE_KEY in instance:
                instance[constants.ROLESET_KEY] = str(instance[SENSE_KEY][predicate_index])

                sense = instance[constants.ROLESET_KEY]
                # if LABEL_KEY in instance and 'B-ARGM-LVB' in instance[LABEL_KEY]:
                #     sense = 'HN'

                if self._sense_mappings:
                    instance[constants.SENSE_PRED_KEY] = sense
                    if self._is_rs_mapping:
                        if re.match("^\\d\\d$", sense):  # PropBank roleset, e.g. 01
                            sense = instance[constants.PREDICATE_LEMMA] + '.' + sense  # e.g. swim.01
                    else:
                        sense = instance[constants.PREDICATE_LEMMA]

                    mapped = self._sense_mappings.get(sense)
                    if not mapped:
                        if sense not in self._no_mappings:
                            # don't spam logs, but warn for missing mappings
                            tf.logging.warning("No mapping for %s" % sense)
                            self._no_mappings.add(sense)
                        mapped = ['None']
                    sense = mapped
                instance[SENSE_KEY] = sense

            instance[MARKER_KEY] = [index == predicate_index and '1' or '0'
                                    for index in range(0, len(all_labels[LABEL_KEY]))]

        return instances

    def read_predicates(self, rows):
        pred_indices = []
        pred_cols = defaultdict(list)
        for token_idx, row_fields in enumerate(rows):
            if self.is_predicate(row_fields):
                pred_indices.append(token_idx)
            for index in range(self._pred_start, len(row_fields) - self._pred_end):
                pred_cols[index - self._pred_start].append(row_fields[index])

        index_to_labels = {}
        # convert from CoNLL05 labels to IOB labels
        for key, val in pred_cols.items():
            index_to_labels[key] = {label_key: convert_conll_to_bio(val, label_mappings=label_mapping,
                                                                    map_with_regex=self._regex_mapping,
                                                                    map_with_regex_post=self._map_with_regex_post)
                                    for label_key, label_mapping in self._label_mappings.items()}

        assert len(pred_indices) <= len(index_to_labels), (
                'Unexpected number of predicate columns: %d instead of %d'
                ', check that predicate start and end indices are correct: %s' % (len(index_to_labels), len(pred_indices), rows))
        # create predicate dictionary with keys as predicate word indices and values as corr. lists of labels (1 for each token)
        predicates = {i: index_to_labels[index] for index, i in enumerate(pred_indices)}
        return predicates


class CoNLLSrlPhraseReader(ConllSrlReader):
    def __init__(self, index_field_map, pred_start, pred_end=0, pred_key="predicate", chunk_func=lambda x: x,
                 line_filter=lambda line: False):
        super().__init__(index_field_map, pred_start, pred_end, pred_key, chunk_func, line_filter)

    def read_files(self, path, extension=".txt", phrase_ext=".phrases"):
        """
        Reads CoNLL files with their corresponding phrase files.
        :param path: directory or single file
        :param extension: extension of files in directory
        :param phrase_ext: extension of phrase files
        :return: CoNLL SRL instances divided into phrases
        """
        if extension and os.path.isdir(path):
            for root, dir_names, file_names in os.walk(path):
                for file_name in fnmatch.filter(file_names, '*' + extension):
                    phrase_file = re.sub(extension + "$", phrase_ext, file_name)
                    for instance in self.read_file(os.path.join(root, file_name), os.path.join(root, phrase_file)):
                        yield instance
        for instance in self.read_file(path):
            yield instance

    def read_file(self, path, phrase_path=None, phrase_ext=".phrases"):
        """
        Read CoNLL SRL phrase instances from a file at a given path.
        :param path: path to single CoNLL-formatted file
        :param phrase_path: phrase file path
        :param phrase_ext: extension of phrase files
        :return: CoNLL instances
        """
        if not phrase_path:
            phrase_path = re.sub("\\..*?$", phrase_ext, path)
        if not os.path.isfile(phrase_path):
            raise ValueError('Missing SRL phrase file "{}" for path "{}"'.format(phrase_path, path))

        with file_io.FileIO(path, 'r') as conll_file, file_io.FileIO(phrase_path, 'r') as phrase_file:
            current, current_phrases = [], []
            for line, chunk_line in zip(conll_file, phrase_file):
                line, chunk_line = line.strip(), chunk_line.strip()
                if (not line and chunk_line) or (not chunk_line and line):
                    raise ValueError(
                        'Misaligned phrase and CoNLL files: {} vs. {} in {} and {}'.format(chunk_line, line, phrase_path, path))
                if not line or self.line_filter(line):
                    if current:
                        for instance in self.read_instances([line.split() for line in current], phrases=current_phrases):
                            yield instance
                        current, current_phrases = [], []
                    continue
                current.append(line)
                current_phrases.append(chunk_line)
            if current:
                for instance in self.read_instances([line.split() for line in current], phrases=current_phrases):
                    yield instance

    def read_instances(self, rows, phrases=None):
        if not phrases:
            raise ValueError("Phrases not provided for instance: {}".format(rows))
        instances = []
        for index, labels in self.read_predicates(rows).items():
            instance = self._read_phrases(rows, phrase_labels=phrases, predicate_index=index, labels=labels[LABEL_KEY])
            # noinspection PyTypeChecker
            instance[INSTANCE_INDEX] = self.prop_count
            # noinspection PyTypeChecker
            instance[SENTENCE_INDEX] = self._sentence_count
            instances.append(instance)
            self.prop_count += 1
        self._sentence_count += 1
        return instances

    def _read_phrases(self, rows, phrase_labels, predicate_index, labels):
        new_labels = []  # label per phrase
        predicate_chunk_index = -1  # index of phrase containing the predicate
        phrases = []  # list of phrases, each phrase represented by a list of fields from the input file
        curr_chunk = []  # the phrase currently being updated
        prev_label = None
        if not (len(rows) == len(phrase_labels) == len(labels)):
            raise AssertionError(
                'Unequal number of rows phrases, and labels: {} vs. {} vs. {}'.format(len(rows), len(phrase_labels), len(labels)))

        for token_index, (row, curr_label) in enumerate(zip(rows, phrase_labels)):
            if end_of_chunk(prev_label, curr_label):
                phrases.append(curr_chunk)
                curr_chunk = []
            elif curr_chunk:
                curr_chunk.append(row)

            if start_of_chunk(prev_label, curr_label):
                curr_chunk.append(row)
            if predicate_index == token_index:
                predicate_chunk_index = len(phrases)
            prev_label = curr_label
        if curr_chunk:
            phrases.append(curr_chunk)

        word_index = 0
        new_index = -1
        fixed_phrases = []
        for index, phrase in enumerate(phrases):
            if index == predicate_chunk_index:
                for row in phrase:
                    if word_index == predicate_index:
                        new_index = len(fixed_phrases)
                    fixed_phrases.append([row])
                    new_labels.append(labels[word_index])
                    word_index += 1
            else:
                fixed_phrases.append(phrase)
                new_labels.append(labels[word_index])
                word_index += len(phrase)

        instance = defaultdict(list)
        for phrase in [self.read_fields(phrase) for phrase in fixed_phrases]:
            for key, val in phrase.items():
                instance[key].append(val)
        instance[LABEL_KEY] = new_labels
        instance[MARKER_KEY] = [index == new_index and '1' or '0' for index in range(0, len(new_labels))]
        return instance


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


def conllx_reader():
    return ConllDepReader(index_field_map={0: ID_KEY, 1: WORD_KEY, 2: LEMMA_KEY, 3: POS_KEY, 4: XPOS_KEY,
                                           5: FEAT_KEY, 6: HEAD_KEY, 7: DEPREL_KEY, 8: ENHANCED_DEPS_KEY, 9: MISC_KEY},
                          label_field=DEPREL_KEY)


def conll_2005_reader(phrase=False):
    fields = {0: WORD_KEY, 1: POS_KEY, 2: PARSE_KEY, 3: NAMED_ENTITY_KEY, 4: SENSE_KEY, 5: PREDICATE_KEY}
    if phrase:
        return CoNLLSrlPhraseReader(fields, pred_start=6)
    return ConllSrlReader(fields, pred_start=6)


def conll_2012_reader(phrase=False):
    fields = {3: WORD_KEY, 4: POS_KEY, 5: PARSE_KEY, 6: PREDICATE_KEY, 7: SENSE_KEY}

    reader = CoNLLSrlPhraseReader(fields, pred_start=11, pred_end=1) if phrase else ConllSrlReader(
        fields, pred_start=11, pred_end=1)

    reader.is_predicate = lambda line: line[6] is not '-' and line[7] is not '-'
    return reader


class LengthFilter(object):
    def __init__(self, length, reader, length_key=lambda x: len(x[WORD_KEY])):
        super().__init__()
        self.length = length
        self.reader = reader
        self.length_key = length_key

    def read_file(self, *args, **kwargs):
        for instance in self.reader.read_file(*args, **kwargs):
            if not self.length_key(instance) > self.length:
                yield instance


class KeyFilter(object):
    def __init__(self, reader, filter_key, include, exclude):
        super().__init__()
        self.reader = reader
        self.filter_key = filter_key
        self.include = include if include else []
        self.exclude = exclude if exclude else []

    def read_file(self, *args, **kwargs):
        for instance in self.reader.read_file(*args, **kwargs):
            if self.include and instance[self.filter_key] in self.include:
                yield instance
            elif self.exclude and instance[self.filter_key] not in self.exclude:
                yield instance


class DuplicateWithUncased(object):
    def __init__(self, reader, every_k=2):
        super().__init__()
        self.reader = reader
        self._every_k = every_k

    def read_file(self, *args, **kwargs):
        count = 0
        for instance in self.reader.read_file(*args, **kwargs):
            if count % self._every_k == 0:
                new_instance = dict(instance)
                new_instance[WORD_KEY] = [w.lower() for w in instance[WORD_KEY]]
                yield new_instance
            yield instance
            count += 1


def get_reader(reader_config, training_config=None, is_test=False):
    """
    Return a corpus reader for a given config.
    :param reader_config: reader configuration
    :param training_config: training configuration
    :param is_test: prepare reader for testing
    """

    def reader_from_name(reader_name):
        if reader_name == 'conll_2003':
            return conll_2003_reader()
        elif reader_name == 'conll_2009':
            return conll_2009_reader()
        elif reader_name == 'conllx':
            return conllx_reader()
        elif reader_name == 'conll_2005':
            return conll_2005_reader()
        elif reader_name == 'conll_2005_phrase':
            return conll_2005_reader(phrase=True)
        elif reader_name == 'conll_2012':
            return conll_2012_reader()
        elif reader_name == 'conll_2012_phrase':
            return conll_2012_reader(phrase=True)
        elif reader_name == 'ptb_pos':
            return ptb_pos_reader()
        elif reader_name == 'tsv':
            return TsvReader()
        elif reader_name == 'semlink':
            return SemlinkReader()
        raise ValueError('Unexpected reader type: %s' % reader_name)

    if isinstance(reader_config, str):
        reader = reader_from_name(reader_config)
    else:
        if reader_config.get('field_index_map'):
            field_index_map = reader_config.field_index_map
            index_field_map = {val: key for key, val in field_index_map.items()}
            if reader_config.get('pred_start'):
                reader = ConllSrlReader(index_field_map=index_field_map,
                                        pred_start=reader_config.get('pred_start'),
                                        pred_end=reader_config.get('pred_end', 0),
                                        label_mappings=reader_config.get('label_mappings'),
                                        regex_mapping=reader_config.get('map_with_regex', False),
                                        map_with_regex_post=reader_config.get('map_with_regex_post', False),
                                        sense_mappings=reader_config.get('sense_mappings'),
                                        pred_filter=reader_config.get('pred_filter'),
                                        sense_predictions_suffix=reader_config.get('sense_predictions_suffix'))
                if SENSE_KEY in field_index_map and PREDICATE_KEY in field_index_map:
                    def is_predicate(line):
                        return line[field_index_map[PREDICATE_KEY]] is not '-' and any('(V*' in f for f in line)

                    reader.is_predicate = is_predicate
            else:
                reader = ConllReader(index_field_map)
        elif reader_config.get('readers'):
            reader = MultiConllReader([get_reader(reader) for reader in reader_config.readers], reader_config.suffixes)
        else:
            raise ValueError("Unexpected reader type: " + reader_config)
    if not is_test and training_config and training_config.max_length > 0:
        reader = LengthFilter(length=training_config.max_length, reader=reader)
    if not is_test and training_config and training_config.duplicate_uncased > 0:
        reader = DuplicateWithUncased(reader=reader, every_k=training_config.duplicate_uncased)
    if training_config and training_config.filter_key and (training_config.exclude or training_config.include):
        reader = KeyFilter(reader=reader, filter_key=training_config.filter_key, include=training_config.include,
                           exclude=training_config.exclude)
    return reader


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


def conll2semlink(conll_path, out_file, reader=None):
    if not reader:
        reader = get_reader(Params(**{
            "field_index_map": {
                "word": 2,
                "pos": 3,
                "sense": 4,
                "predicate": 5
            },
            "pred_start": 6
        }))
    with open(out_file, mode='wt') as out:
        for instance in reader.read_file(conll_path):
            text = ' '.join([word for word in instance[WORD_KEY]])
            for word in instance[WORD_KEY]:
                if ' ' in word:
                    print('A space ' ' was found in a token in an instance w/ text: %s' % text)
            sense = instance[SENSE_KEY]
            token = instance[PREDICATE_INDEX_KEY]
            sentence = instance[INSTANCE_INDEX]
            # noinspection PyTypeChecker
            predicate = instance[PREDICATE_KEY][token]
            # noinspection PyStringFormat
            out.write('%s %d %d %s %s\t%s\n' % (conll_path, sentence, token, predicate, sense, text))


PTB_MAPPINGS = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LCB-': '{',
    '-RCB-': '}',
    '-LSB-': '[',
    '-RSB-': ']',
    '\'\'': '\"',
    '``': '\"',
}


def unescape_ptb(ptb_token: str) -> str:
    """
    Convert PTB-style token back to original form.
    >>> unescape_ptb('-LRB-')
    '('

    :param ptb_token: PTB-style token
    :return: un-PTB-escaped token
    """
    return PTB_MAPPINGS.get(ptb_token, ptb_token)
