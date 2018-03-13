import json

import os

import pickle


class Params(dict):
    def __init__(self, **kwargs):
        super(Params).__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
            self[key] = val


def _convert_to_attributes(dictionary):
    for key, val in dictionary.items():
        if isinstance(val, dict):
            dictionary[key] = _convert_to_attributes(val)
        elif isinstance(val, list):
            result = []
            for entry in val:
                if isinstance(entry, dict):
                    result.append(_convert_to_attributes(entry))
                else:
                    result.append(entry)
            dictionary[key] = result
    return Params(**dictionary)


def read_jsons(json_string):
    """
    Read a JSON string as an attribute dictionary--all dicts are recursively converted to attribute dictionaries.
    :param json_string: JSON string
    :return: attribute dictionary for input JSON
    """
    json_dict = json.loads(json_string)
    return _convert_to_attributes(json_dict)


def read_json(json_path):
    """
    Read a JSON file as an attribute dictionary--all dicts are recursively converted to attribute dictionaries.
    :param json_path: path to JSON file
    :return: attribute dictionary for input JSON
    """
    with open(json_path, 'r') as lines:
        json_dict = json.load(lines)
        return _convert_to_attributes(json_dict)


def serialize(serializable, out_path, out_name=None):
    if out_name:
        out_name = out_name if out_name.endswith(".pkl") else "{}.pkl".format(out_name)
    path = os.path.join(out_path, out_name) if out_name else out_path
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    try:
        os.makedirs(parent_path)
    except OSError:
        if not os.path.isdir(parent_path):
            raise
    with open(path, mode="wb") as out_file:
        pickle.dump(serializable, out_file)


def deserialize(in_path, in_name=None):
    if in_name:
        in_name = in_name if in_name.endswith(".pkl") else "{}.pkl".format(in_name)
    path = os.path.join(in_path, in_name) if in_name else in_path
    with open(path, mode="rb") as in_file:
        return pickle.load(in_file)
