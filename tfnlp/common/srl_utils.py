import glob
import os
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
