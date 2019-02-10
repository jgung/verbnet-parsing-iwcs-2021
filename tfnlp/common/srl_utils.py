import glob
import os
import re
from collections import defaultdict
from typing import Dict
from xml.etree import ElementTree

_PREDICATE = 'predicate'
_ROLESET = 'roleset'
_ROLES = 'roles'
_ROLE = 'role'
_ID = 'id'
_NUMBER = 'n'
_FT = 'f'


def get_argument_function_mappings(frames_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Return a dictionary from roleset IDs (e.g. 'take.01') to dictionaries mapping numbered arguments to function tags.

    >>> mappings = get_argument_function_mappings('/path/to/frames/')
    >>> mappings['take.01']
    {'0': 'PAG', '1': 'PPT', '2': 'DIR', '3': 'GOL'}
    >>> mappings['take.01']['0']
    'PAG'

    :param frames_dir: directory containing PropBank frame XML files.
    :return: mappings from arguments to function tags, by roleset ID
    """
    mappings = defaultdict(dict)
    for framefile in glob.glob(os.path.join(frames_dir, '*.xml')):
        frame = ElementTree.parse(framefile).getroot()
        for predicate in frame.findall(_PREDICATE):
            for roleset in predicate.findall(_ROLESET):
                rs_mappings = mappings[roleset.get(_ID)]
                for roles in roleset.findall(_ROLES):
                    for role in roles.findall(_ROLE):
                        number = role.get(_NUMBER).upper()
                        ft = role.get(_FT).upper()
                        rs_mappings[number] = ft
    return mappings


NUMBER_PATTERN = r'(?:A|ARG)([A\d]|M(?=-))'


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
    :return: role number string
    """
    numbers = re.findall(NUMBER_PATTERN, role, re.IGNORECASE)
    if not numbers:
        raise ValueError('Unsupported or invalid PropBank role format: %s' % role)
    return numbers[0]
