import json
import os
import pickle

from tensorflow.python.lib.io import file_io


class Params(dict):
    def __init__(self, **kwargs):
        super(Params).__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
            self[key] = val

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self[key] = value


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
    with file_io.FileIO(json_path, 'r') as lines:
        json_dict = json.load(lines)
        return _convert_to_attributes(json_dict)


def write_json(value, json_path):
    with file_io.FileIO(json_path, 'w') as json_out:
        json_out.write(json.dumps(value, indent=4, sort_keys=True))


def serialize(serializable, out_path, out_name=None, overwrite=False):
    if out_name:
        out_name = out_name if out_name.endswith(".pkl") else "{}.pkl".format(out_name)
    path = os.path.join(out_path, out_name) if out_name else out_path
    parent_path = os.path.abspath(os.path.join(path, os.path.pardir))
    try:
        os.makedirs(parent_path)
    except OSError:
        if not os.path.isdir(parent_path):
            raise
    if not overwrite and os.path.exists(path):
        raise AssertionError("Pre-existing vocabulary file at %s. Set `overwrite` to `True` to ignore." % path)
    with file_io.FileIO(path, mode="wb") as out_file:
        pickle.dump(serializable, out_file)


def deserialize(in_path, in_name=None):
    if in_name:
        in_name = in_name if in_name.endswith(".pkl") else "{}.pkl".format(in_name)
    path = os.path.join(in_path, in_name) if in_name else in_path
    with file_io.FileIO(path, mode="rb") as in_file:
        return pickle.load(in_file)


def binary_np_array_to_unicode(np_string_array):
    return [bstr.decode('utf-8') for bstr in np_string_array.tolist()]
