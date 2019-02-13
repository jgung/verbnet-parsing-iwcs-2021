import argparse
import glob
import os
import re
from collections import defaultdict, Counter, namedtuple
from typing import Dict, Optional, Callable, Any, List, TextIO
from xml.etree import ElementTree

from tfnlp.common.chunk import labels_to_spans, convert_conll_to_bio, spans_to_conll_labels
from tfnlp.common.utils import read_json

_PREDICATE = 'predicate'
_ROLESET = 'roleset'
_ROLES = 'roles'
_ROLE = 'role'
_ID = 'id'
_NUMBER = 'n'
_FT = 'f'

_VALID_NUMBERS = {'0', '1', '2', '3', '4', '5', '6', 'A', 'M'}

_ARG_PATTERN = r'(A[\dA])'
_MOD_PATTERN = r'AM-'
_NUMBER_PATTERN = r'(?:ARG|A)([A\d])'
_ARG_STR = r'(\S+)?ARG(\S(?:-\S+)?)'
_CORE_ARG_PREFIX = 'AC-'


def get_argument_function_mappings(frames_dir: str,
                                   add_co_marker: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Return a dictionary from roleset IDs (e.g. 'take.01') to dictionaries mapping numbered arguments to function tags.

    >>> mappings = get_argument_function_mappings('/path/to/frames/')

    >>> mappings['take.01']
    {'0': 'PAG', '1': 'PPT', '2': 'DIR', '3': 'GOL'}
    >>> mappings['take.01']['0']
    'PAG'

    :param frames_dir: directory containing PropBank frame XML files.
    :param add_co_marker: if 'True', add '2' if function is already present, and is secondary, e.g. co-patient -> PPT2
    :return: mappings from arguments to function tags, split by roleset ID
    """
    mappings = defaultdict(dict)
    for framefile in glob.glob(os.path.join(frames_dir, '*.xml')):
        frame = ElementTree.parse(framefile).getroot()
        lemma = re.findall('(\S+)\.xml', os.path.basename(framefile))[0]
        for predicate in frame.findall(_PREDICATE):
            for roleset in predicate.findall(_ROLESET):
                rs_id = lemma + '.' + re.findall('^\S+\.(\S+)$', roleset.get(_ID))[0]
                rs_mappings = mappings[rs_id]
                for roles in roleset.findall(_ROLES):
                    # sometimes, there will be multiple of the same FT, e.g. two PAGs, in which case we add 'PAG2'
                    fts = set()
                    # roles should be in alphanumeric order, so we properly assign co-patient/co-agent
                    sorted_roles = sorted(roles.findall(_ROLE), key=lambda r: r.get(_NUMBER).upper())
                    for role in sorted_roles:
                        number = role.get(_NUMBER).upper()
                        if number not in _VALID_NUMBERS:
                            raise ValueError('Unexpected number format: %s' % number)
                        ft = role.get(_FT).upper()
                        if add_co_marker:
                            if ft in fts:
                                ft = ft + '2'
                            else:
                                fts.add(ft)
                        rs_mappings[number] = ft
    return dict(mappings)


def get_number(role: str) -> Optional[str]:
    """
    Returns the PropBank number associated with a particular role for different formats, or 'None' if not a numbered argument.
    >>> get_number('A3')
    '3'
    >>> get_number('C-ARG3')
    '3'
    >>> get_number('ARGM-TMP')
    None

    :param role: role string, e.g. 'ARG4' or 'A4'
    :return: single-character role number string, e.g. '4'
    """
    numbers = re.findall(_NUMBER_PATTERN, role, re.IGNORECASE)
    if not numbers:
        return None
    return numbers[0].upper()


def arg_to_a(role: str):
    """
    Convert 'ARGM-TMP' form of role label to 'AM-TMP' form.
    :param role: original role label
    :return: shortened role, with 'A' instead of 'ARG'
    """
    return re.sub(_ARG_STR, r'\1A\2', role, re.IGNORECASE).upper()


def apply_numbered_arg_mappings(roleset_id: str,
                                role: str,
                                mappings: Dict[str, Dict[str, str]],
                                ignore_unmapped: bool = False,
                                append: bool = False,
                                combine_modifiers: bool = False,
                                arga_mapping: str = 'PAG') -> Optional[str]:
    """
    Apply argument mappings for a given roleset and role.
    >>> apply_numbered_arg_mappings('take.01', 'A4', mappings)
    'GOL'
    >>> apply_numbered_arg_mappings('take.01', '(A4*', mappings, append=True)
    '(A4-GOL*'

    :param roleset_id: roleset ID, e.g. 'take.01'
    :param role: role string, e.g. 'A4'
    :param mappings: dictionary of mappings from numbered arguments by roleset
    :param ignore_unmapped: if 'True', return unmodified role string if mapping is not present
    :param append: if 'True', append mapping with a hyphen instead of replacing
    :param combine_modifiers: if 'True', remove "ARGM-" from modifier labels
    :param arga_mapping: mapping for ARGA, if not already existing
    :return: mapped role, or 'None' if no mapping exists and ignore_unmapped is set to 'False'
    """
    if role == 'V':
        return role
    if combine_modifiers:
        role = re.sub(_MOD_PATTERN, '', role)

    roleset_map = mappings.get(roleset_id)
    if roleset_map is None:
        if roleset_id.endswith('LV'):
            return role
        role_number = get_number(role)
        if not role_number:
            if ignore_unmapped:
                return role
            return None
        raise ValueError('Missing roleset in mappings: %s' % roleset_id)

    role_number = get_number(role)  # e.g. 'A4' -> '4'
    mapped = roleset_map.get(role_number)  # e.g. '4' -> 'GOL'
    if not mapped and role_number == 'A' and arga_mapping:
        mapped = arga_mapping
    if not mapped:
        if ignore_unmapped:
            return role
        return None
    if append:
        # we're going to append the mapped label to the result, e.g. A4 -> GOL => 'A4-GOL'
        return re.sub(_ARG_PATTERN, '\\1-' + mapped, role)
    elif combine_modifiers:
        # just returned the mapped role, if we are combining with modifiers
        return re.sub(_NUMBER_PATTERN, mapped, role)
    else:
        # differentiate from modifiers by adding a core argument prefix to the mapped result, e.g. A4 -> GOL => 'AC-GOL'
        return re.sub(_NUMBER_PATTERN, _CORE_ARG_PREFIX + mapped, role)


class CoNllProcessor(object):
    """
    Read a CoNLL 2012-formatted file and perform some processing operation over individual sentences.

    :param lemma_col: column of lemma
    :param roleset_col: column for roleset ID
    :param arg_start: column of first argument
    :param arg_end: end index of arguments
    """

    def __init__(self,
                 tag: str = '',
                 lemma_col: int = 6,
                 roleset_col: int = 7,
                 arg_start: int = 11,
                 arg_end: int = -1) -> None:
        super().__init__()
        self.tag = tag
        self.lemma_col = lemma_col
        self.roleset_col = roleset_col
        self.arg_start = arg_start
        self.arg_end = arg_end
        self.ignore_fn = lambda l: l.startswith('#')
        self.skip_rs = lambda r: r == '-'

    def process_file(self, conll_file: str):
        """
        Perform processing on a single file.
        :param conll_file: path to input file
        """
        out_file = os.path.basename(conll_file) + '.' + self.tag
        context = self._open_context(out_file)

        sentence_number = 0
        with open(conll_file, 'r') as conll_lines:
            sentence = []
            rolesets = []  # keep ordered list of rolesets to be processed at end of sentence
            for line in conll_lines:
                line = line.strip()
                if not line or self.ignore_fn(line):
                    if sentence:
                        self._process_sentence(sentence, rolesets, context)
                        sentence = []
                        rolesets = []
                        sentence_number += 1
                        if sentence_number % 1000 == 0:
                            print('...processed %dK sentences' % sentence_number / 1000)
                    continue
                fields = line.split()

                number = fields[self.roleset_col]  # e.g. '01'
                roleset = fields[self.lemma_col] + '.' + number  # e.g. 'take.01;

                if not self.skip_rs(fields[self.roleset_col]):
                    rolesets.append(roleset)

                sentence.append(fields)

            if sentence:
                self._process_sentence(sentence, rolesets, context)

        self._end(context, out_file)

    def _open_context(self, output_file: str) -> Any:
        """
        Open a new processing context.
        :param output_file: name of output file
        """
        return None

    def _process_sentence(self, rows: List[List[str]], rolesets: List[str], context) -> None:
        """
        Perform processing on a single sentence
        :param rows: list of CoNLL rows, each split into individual fields (e.g. path, token, pos, ...)
        :param rolesets: list of rolesets in original order
        :param context: processing context
        """
        pass

    def _end(self, context, output_file: str) -> None:
        """
        Perform final processing using accumulated context.
        :param context: processing context
        :param output_file: name of output file
        """
        pass


class CoNllArgMapper(CoNllProcessor):
    """
    Map arguments based on their rolesets, output result to provided file path.
    :param mapping_fn: argument mapping function, from rolesets to roleset-specific argument mappings
    :param out_file: path to output file
    """

    def __init__(self, mapping_fn: Callable[[str, str], Optional[str]], out_file: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mapping_fn = mapping_fn
        self.out_file = out_file

    def _open_context(self, output_file: str) -> TextIO:
        out_path = os.path.join(self.out_file, output_file + '.conll')
        print('writing mapped props to %s' % out_path)
        return open(out_path, 'w')

    def _process_sentence(self, rows: List[List[str]], rolesets: List[str], context: TextIO) -> None:

        mapped_args = []
        for i, rs in enumerate(rolesets):
            original = [row[self.arg_start + i] for row in rows]
            spans = labels_to_spans(convert_conll_to_bio(original))
            mapped_spans = [(self.mapping_fn(rs, label), start, end) for (label, start, end) in spans]
            mapped_args.append(spans_to_conll_labels(mapped_spans, len(rows)))

        for i, field_list in enumerate(rows):
            mapped = [args[i] for args in mapped_args]
            new_line = ' '.join(field_list[:self.arg_start] + mapped + field_list[self.arg_end:])
            context.write(new_line + '\n')
        context.write('\n')

    def _end(self, context: TextIO, output_file: str) -> None:
        context.close()


class CoNllArgCounter(CoNllProcessor):
    """
    Generate counts for mappings from original arguments to mapped arguments.
    :param mapping_fn: argument mapping function, from rolesets to roleset-specific argument mappings
    :param out_file: path to output file
    """

    Context = namedtuple('Context', ['original_counts', 'mapped_counts'])

    def __init__(self, mapping_fn, out_file, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mapping_fn = mapping_fn
        self.out_file = out_file

    def _open_context(self, output_file: str) -> Context:
        context = self.Context(defaultdict(Counter), Counter())
        return context

    def _process_sentence(self, rows: List[List[str]], rolesets: List[str], context: Context) -> None:
        for i, rs in enumerate(rolesets):
            original = [row[self.arg_start + i] for row in rows]
            spans = labels_to_spans(convert_conll_to_bio(original))
            for label, start, end in spans:
                mapped = self.mapping_fn(rs, label)
                context.original_counts[arg_to_a(label)][mapped] += 1
                context.mapped_counts[mapped] += 1

    def _end(self, context: Context, output_file: str) -> None:
        out_path = os.path.join(self.out_file, output_file + '.counts.tsv')
        with open(out_path, 'w') as out:
            print('writing mappings counts to %s' % out_path)
            numbers = set(context.original_counts.keys())
            key_list = [key for key, val in sorted(context.mapped_counts.items(), key=lambda x: x[1], reverse=True)]
            out.write('\t%s\n' % '\t'.join(key_list))
            for number in sorted(numbers):
                total = sum([val for key, val in context.original_counts[number].items()])
                out.write('%s\t%s\t%d\n'
                          % (number, '\t'.join([str(context.original_counts[number][key]) for key in key_list]), total))
            out.write('\t%s\n' % '\t'.join([str(context.mapped_counts[key]) for key in key_list]))


class CoNllPhraseWriter(CoNllProcessor):
    """
    Apply mappings to arguments in a CoNLL file, then output all mapped phrases to a file.
    :param mapping_fn: argument mapping function, from rolesets to roleset-specific argument mappings
    :param out_file: path to output file
    """

    def __init__(self, mapping_fn, out_file, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mapping_fn = mapping_fn
        self.out_file = out_file

    def _open_context(self, output_file: str) -> defaultdict:
        return defaultdict(Counter)

    def _process_sentence(self, rows: List[List[str]], rolesets: List[str], context: defaultdict) -> None:
        for i, rs in enumerate(rolesets):
            original = [row[self.arg_start + i] for row in rows]
            spans = labels_to_spans(convert_conll_to_bio(original))
            for label, start, end in spans:
                mapped_label = self.mapping_fn(rs, label)
                context[mapped_label][rs + ' ' + ' '.join([row[3] for row in rows[start:end]])] += 1

    def _end(self, context: defaultdict, output_file: str) -> None:
        out_path = os.path.join(self.out_file, output_file + '.phrases.tsv')
        with open(out_path, 'w') as out:
            print('writing phrases to %s' % out_path)
            for phrase_label, phrase in context.items():
                for span, count in phrase.items():
                    out.write('%s\t%d\t%s\n' % (arg_to_a(phrase_label), count, span))


class AggregateProcessor(CoNllProcessor):
    def __init__(self, processors: List[CoNllProcessor], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processors = processors

    def _open_context(self, output_file: str) -> list:
        return [processor._open_context(output_file) for processor in self.processors]

    def _process_sentence(self, rows: List[List[str]], rolesets: List[str], contexts) -> None:
        for context, processor in zip(contexts, self.processors):
            processor._process_sentence(rows, rolesets, context)

    def _end(self, contexts, output_file: str) -> None:
        for context, processor in zip(contexts, self.processors):
            processor._end(context, output_file)


def _add_c_r_mappings(mappings: Dict[str, str]) -> Dict[str, str]:
    updated = {**mappings}
    for key, val in mappings.items():
        updated['C-' + key] = 'C-' + val
        updated['R-' + key] = 'R-' + val
    return updated


def main(opts):
    mappings = get_argument_function_mappings(opts.frames)

    def mapping_fn(rs, r):
        r = arg_to_a(r)
        return apply_numbered_arg_mappings(rs, r, mappings,
                                           ignore_unmapped=True,
                                           combine_modifiers=opts.combine,
                                           append=opts.append)

    mapping_function = mapping_fn
    if opts.mappings:
        # apply additional mappings to the output of the mapping function, returning original label if not mapped
        json_mappings = _add_c_r_mappings(read_json(opts.mappings))

        def updated_mapping_fn(rs, r):
            r = arg_to_a(r)

            mapped = apply_numbered_arg_mappings(rs, r, mappings,
                                                 ignore_unmapped=True,
                                                 combine_modifiers=opts.combine,
                                                 append=opts.append)

            return json_mappings.get(mapped, r)

        mapping_function = updated_mapping_fn

    if opts.mappings:
        tag = 'mappings'
    elif opts.combine:
        tag = 'combined'
    elif opts.append:
        tag = 'split'
    else:
        tag = 'core-mod'

    mode_map = {
        'map': CoNllArgMapper(mapping_function, opts.output, tag=tag),
        'count': CoNllArgCounter(mapping_function, opts.output, tag=tag),
        'phrases': CoNllPhraseWriter(mapping_function, opts.output, tag=tag)
    }

    mode_map['map'] = AggregateProcessor([mode_map['map'], mode_map['count']], tag=tag)

    processor = mode_map[opts.mode]

    print('running mappings script in "%s/%s" mode...' % (tag, opts.mode))

    for file in [f for f in opts.input.split(',') if f]:
        print('processing %s...' % file)
        processor.process_file(file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--frames', type=str, required=True, help='Path to directory containing frame file XMLs')
    argparser.add_argument('--input', type=str, required=True, help='Input CoNLL 2012 file')
    argparser.add_argument('--output', type=str, required=True, help='Directory to store output files')
    argparser.add_argument('--append', action='store_true',
                           help='Append mappings instead of replacing original label, e.g. "A0-PPT"')
    argparser.add_argument('--combine', action='store_true', help='Combine modifiers (AM-TMP) w/ function tags (TMP)')
    argparser.add_argument('--mode', type=str, default='map', choices=['map', 'count', 'phrases'],
                           help='Mode to apply mappings')
    argparser.add_argument('--mappings', type=str, help='Path to JSON mappings file')
    argparser.set_defaults(append=False)
    argparser.set_defaults(combine=False)
    main(argparser.parse_args())
