import argparse
import copy
import os
import sys

from tfnlp.common.utils import read_json, write_json

JSON_EXT = ".json"


def build_config(base_config: dict, override_configs: dict = None, override_params: dict = None, config_dir: str = None):
    """
    Build a combined configuration from multiple overriding configurations and specific parameter overrides.
    :param base_config: base configuration to add to/override
    :param override_configs: dict of override configurations
    :param override_params: dict of parameter overrides
    :param config_dir: path from which to resolve relative path references in config file
    :return: combined/overridden configuration
    """
    result = copy.deepcopy(base_config)

    def _sub_str(value):
        if value.endswith(JSON_EXT):
            if value.startswith("."):
                value = os.path.join(config_dir, value)
            return read_json(value)
        return value

    def _sub_values_list(_params):
        res = []
        for value in _params:
            if isinstance(value, dict):
                value = _sub_dict(value)
            elif isinstance(value, str):
                value = _sub_str(value)
            elif isinstance(value, list):
                value = _sub_values_list(value)
            res.append(value)
        return res

    def _sub_dict(_params):
        res = {}
        for key, val in _params.items():
            if isinstance(val, str):
                val = _sub_str(val)
            elif isinstance(val, list):
                val = _sub_values_list(val)
            elif isinstance(val, dict):
                val = _sub_dict(val)
            res[key] = val
        return res

    # resolve config references
    if config_dir:
        if os.path.isfile(config_dir):
            config_dir = os.path.abspath(os.path.join(config_dir, os.pardir))
        result = _sub_dict(result)

    # add separate top-level configs provided as parameters
    if override_configs:
        for k, config in override_configs.items():
            if '.' in k:
                raise ValueError('Override configuration keys can only be top-level parameters (no nested params)')
            result[k] = read_json(config)

    # add specific parameter changes (possibly nested)
    if override_params:
        for k, param_value in override_params.items():
            # cast value
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass

            nested_keys = k.split('.')
            curr_param = result
            for nested_key in nested_keys[:len(nested_keys) - 1]:
                if not curr_param.get(nested_key):
                    curr_param[nested_key] = {}
                curr_param = curr_param[nested_key]
            curr_param[nested_keys[-1]] = param_value

    return result


def default_args():
    parser = argparse.ArgumentParser(
        description="Simplifies combining multiple configuration files and specific parameter overrides")
    parser.add_argument('--base', type=str, help='base JSON configuration file', required=True)
    parser.add_argument('--overrides', nargs="*", type=str, help='space-separated list of keys and corresponding '
                                                                 'JSON configuration files')
    parser.add_argument('--params', type=str, help='comma-separated list of parameters with subfields separated by periods, '
                                                   'e.g. "optimizer.lr=0.1"')
    parser.add_argument('--output', type=str, help='output path for combined JSON configuration file', required=True)

    return parser


def _validate_and_parse_args():
    parser = default_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def read_config(config_path: str, config_overrides: list, param_overrides: str):
    base_config = read_json(config_path)

    configs = config_overrides

    override_configs = {}
    if configs:
        if len(configs) % 2 == 1:
            raise ValueError('Expecting an even number of values (key, config) pairs')
        for k, config in zip(configs[::2], configs[1::2]):
            override_configs[k] = read_json(config)

    override_params = {}
    if param_overrides:
        key_value_pairs = [t.split('=') for t in param_overrides.split(',') if t.strip()]
        override_params = {t[0]: t[1] for t in key_value_pairs}

    result = build_config(base_config, override_configs, override_params, config_path)
    return result


if __name__ == '__main__':
    args = _validate_and_parse_args()
    write_json(read_config(args.base, args.overrides, args.params), args.output)
