import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Dict, Optional, Callable
from xml.etree import ElementTree

_PREDICATE = 'predicate'
_ROLESET = 'roleset'
_ROLES = 'roles'
_ROLE = 'role'
_ID = 'id'
_NUMBER = 'n'
_FT = 'f'

VALID_NUMBERS = {'0', '1', '2', '3', '4', '5', '6', 'A', 'M'}


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
                        if number not in VALID_NUMBERS:
                            raise ValueError('Unexpected number format: %s' % number)
                        ft = role.get(_FT).upper()
                        if add_co_marker:
                            if ft in fts:
                                ft = ft + '2'
                            else:
                                fts.add(ft)
                        rs_mappings[number] = ft
    return dict(mappings)


NUMBER_PATTERN = r'(?:ARG|A)([A\d])'


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
    numbers = re.findall(NUMBER_PATTERN, role, re.IGNORECASE)
    if not numbers:
        return None
    return numbers[0].upper()


ARG_PATTERN = r'((?:ARG|A)[\dA])'


def apply_numbered_arg_mappings(roleset_id: str,
                                role: str,
                                mappings: Dict[str, Dict[str, str]],
                                ignore_unmapped: bool = False,
                                append: bool = False,
                                arga_mapping: str = 'PAG') -> Optional[str]:
    """
    Apply argument mappings for a given roleset and role.
    >>> apply_numbered_arg_mappings('take.01', 'A4', mappings)
    'GOL'
    >>> apply_numbered_arg_mappings('take.01', 'A4', mappings, append=True)
    'A4-GOL'

    :param roleset_id: roleset ID, e.g. 'take.01'
    :param role: role string, e.g. 'A4'
    :param mappings: dictionary of mappings from numbered arguments by roleset
    :param ignore_unmapped: if 'True', return unmodified role string if mapping is not present
    :param append: if 'True', append mapping with a hyphen instead of replacing
    :param arga_mapping: mapping for ARGA, if not already existing
    :return: mapped role, or 'None' if no mapping exists and ignore_unmapped is set to 'False'
    """
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
        return re.sub(ARG_PATTERN, '\\1-' + mapped, role)
    return re.sub(NUMBER_PATTERN, mapped, role)


def map_conll_file(conll_file: str,
                   out_file: str,
                   mappings: Callable[[str, str], Optional[str]],
                   lemma_col: int = 6,
                   roleset_col: int = 7,
                   arg_start: int = 11,
                   arg_end: int = -1,
                   ignore_fn=lambda l: l.startswith('#'),
                   skip_rs=lambda r: r == '-'):
    """
    Read a CoNLL 2012-formatted file, use a supplied mapping function to convert arguments, output mapped file.
    :param conll_file: input CoNLL file
    :param out_file: output CoNLL file with mapped arguments
    :param mappings: argument mapping function, from rolesets to roleset-specific argument mappings
    :param lemma_col: column of lemma
    :param roleset_col: column for roleset ID
    :param arg_start: column of first argument
    :param arg_end: end index of arguments
    :param ignore_fn: function to ignore commented lines
    :param skip_rs: function to skip particular rolesets
    """

    if conll_file == out_file:
        raise ValueError('Input file cannot be the same as output file')

    with open(conll_file, 'r') as conll_lines, open(out_file, 'w') as conll_out:

        def process_lines(field_lists, roleset_list):
            for field_list in field_lists:
                mapped = []
                for col, arg in enumerate(field_list[arg_start:arg_end]):
                    if not (arg == '*' or arg == '*)' or '-V*' in arg or '(V*' in arg):
                        arg = mappings(roleset_list[col], arg)
                        mapped.append(arg)
                    else:
                        mapped.append(arg)
                new_line = ' '.join(field_list[:arg_start] + mapped + field_list[arg_end:])
                conll_out.write(new_line + '\n')

        sentence = []
        rolesets = []  # keep ordered list of rolesets to be processed at end of sentence
        for line in conll_lines:
            line = line.strip()
            if not line or ignore_fn(line):
                if sentence:
                    process_lines(sentence, rolesets)
                    sentence = []
                    rolesets = []
                conll_out.write('\n')
                continue
            fields = line.split()

            number = fields[roleset_col]  # e.g. '01'
            roleset = fields[lemma_col] + '.' + number  # e.g. 'take.01;

            if not skip_rs(fields[roleset_col]):
                rolesets.append(roleset)

            sentence.append(fields)

        if sentence:
            process_lines(sentence, rolesets)


def main(opts):
    mappings = get_argument_function_mappings(opts.frames)

    def mapping_fn(rs, r):
        return apply_numbered_arg_mappings(rs, r, mappings, ignore_unmapped=True, append=opts.append)

    map_conll_file(opts.input, opts.output, mapping_fn)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--frames', type=str, required=True, help='Path to directory containing frame file XMLs')
    args.add_argument('--input', type=str, required=True, help='Input CoNLL 2012 file')
    args.add_argument('--output', type=str, required=True, help='Path to output mapped CoNLL 2012 file')
    args.add_argument('--append', action='store_true', help='Append mappings instead of replacing original label')
    args.set_defaults(append=False)
    main(args.parse_args())
