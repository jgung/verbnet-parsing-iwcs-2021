import glob
import os
import re
from collections import defaultdict
from typing import Dict, Optional
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
        for predicate in frame.findall(_PREDICATE):
            for roleset in predicate.findall(_ROLESET):
                rs_mappings = mappings[roleset.get(_ID)]
                for roles in roleset.findall(_ROLES):
                    fts = set()  # sometimes, there will be multiple of the same FT, e.g. two PAG, in which case we add 'PAG2'

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
    return mappings


NUMBER_PATTERN = r'(?:ARG|A)([A\d]|M(?=-))'


def get_number(role: str) -> str:
    """
    Returns the PropBank number associated with a particular role for different formats, or 'M' if not a numbered argument.
    >>> get_number('A3')
    '3'
    >>> get_number('C-ARG3')
    '3'
    >>> get_number('ARGM-TMP')
    'M'

    :param role: role string, e.g. 'ARG4' or 'A4'
    :return: single-character role number string, e.g. '4'
    """
    numbers = re.findall(NUMBER_PATTERN, role, re.IGNORECASE)
    if not numbers:
        raise ValueError('Unsupported or invalid PropBank role format: %s' % role)
    return numbers[0]


def apply_numbered_arg_mappings(roleset_id: str,
                                role: str,
                                mappings: Dict[str, Dict[str, str]],
                                ignore_unmapped: bool = False) -> Optional[str]:
    """
    Apply argument mappings for a given roleset and role.
    :param roleset_id: roleset ID, e.g. 'take.01'
    :param role: role string, e.g. 'A4'
    :param mappings: dictionary of mappings from numbered arguments by roleset
    :param ignore_unmapped: if 'True', return unmodified role string if mapping is not present
    :return: mapped role, or 'None' if no mapping exists and ignore_unmapped is set to 'False'
    """
    roleset_map = mappings[roleset_id]
    if not roleset_map:
        raise ValueError('Missing roleset in mappings: %s' % roleset_id)

    role_number = get_number(role)  # e.g. 'A4' -> '4'
    mapped = roleset_map.get(role_number)  # e.g. '4' -> 'GOL'
    if not mapped:
        if ignore_unmapped:
            return role
        return None
    return mapped
