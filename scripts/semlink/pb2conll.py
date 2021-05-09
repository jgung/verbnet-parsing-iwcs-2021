import argparse
import os
import re
import subprocess
import tempfile
from collections import defaultdict

import sys


def _separate_props_by_file(props_file):
    props_by_path = defaultdict(list)
    with open(props_file) as props:
        for prop in props:
            if not prop:
                continue
            prop_path = prop.split()[0]
            props_by_path[prop_path].append(prop.split())
    for props_path in props_by_path.keys():
        props_by_path[props_path] = sorted(props_by_path[props_path], key=lambda x: (int(x[1]), int(x[2])))
    return props_by_path


def _remove_comments(filepath, filtered_file, line_predicate=lambda x: not x.startswith('*x*')):
    with open(filepath) as lines:
        for line in lines:
            if line_predicate(line):
                filtered_file.write(line + '\n')
    filtered_file.flush()
    return filtered_file


def _combine_treebank(props_file, treebank_file, script_path, noi=True, all_roles=False):
    args = ["-st", "-ft"]
    if noi:
        args.append("-noi")
    if all_roles:
        args.append("-al")
    return subprocess.check_output(["perl", script_path] + args + [
        os.path.abspath(treebank_file),
        os.path.abspath(props_file)], universal_newlines=True)


def link_treebank_propbank(props_by_path,
                           treebank_dir,
                           script_path,
                           output_dir=None,
                           path_mapping_fn=lambda x: x.lower(),
                           combined=None,
                           noi=True,
                           path_filter=lambda x: True,
                           all_roles=False,
                           pretty=False):
    if not output_dir:
        output_dir = treebank_dir

    if combined:
        combined = open(combined, mode='wt')

    count, prop_count = 0, 0
    for dirpath, dirnames, filenames in os.walk(treebank_dir):
        if not dirnames:
            base_dir_path = dirpath.replace(treebank_dir + '/', '')
            for filename in filenames:
                filepath = path_mapping_fn(os.path.join(base_dir_path, filename))
                if filepath not in props_by_path:
                    continue
                if path_filter and not path_filter(filepath):
                    continue
                count += 1

                with _remove_comments(os.path.join(dirpath, filename),
                                      tempfile.NamedTemporaryFile(mode='wt')) as tbfile:
                    props = props_by_path[filepath]
                    prop_count += len(props)
                    with tempfile.NamedTemporaryFile(mode='wt') as propfile:
                        for prop in props:
                            propfile.write(' '.join(prop) + '\n')
                        propfile.flush()
                        # create intermediate directories
                        directory = os.path.abspath(os.path.join(os.path.join(output_dir, filepath), os.path.pardir))
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        with open(os.path.join(output_dir, filepath + '.props'), mode='wt') as outfile:
                            result = _combine_treebank(propfile.name, tbfile.name, script_path, noi=noi, all_roles=all_roles)
                            if not pretty:
                                result = re.sub(' +', ' ', result)
                            outfile.write(result)
                            if combined:
                                combined.write(result)
    if combined:
        combined.close()

    print("Processed %d TreeBank file(s) with %d props." % (count, prop_count))


def options():
    parser = argparse.ArgumentParser(description="Convert PropBank pointer files to the official CoNLL-2005 format.")
    parser.add_argument('--pb', type=str, required=True, help='PropBank pointers file')
    parser.add_argument('--tb', type=str, required=True, help='TreeBank root directory, e.g. treebank_3/parsed/mrg')
    parser.add_argument('--script', type=str, default='scripts/semlink/link_tbpb_vn.pl', help='link_tbpb.pl official script')
    parser.add_argument('--include-inputs', dest='noi', action='store_false',
                        help='include input tokens in output (opposite of -noi in original script)')
    parser.add_argument('--o', type=str, help='(optional) CoNLL output base directory')
    parser.add_argument('--combined', type=str, help='(optional) combined output path')
    parser.add_argument('--filter', type=str, help='(optional) path regex filter, e.g. ".*WSJ/(0[2-9]|1[0-9]|2[01])/.*" ')
    parser.add_argument('--all', action='store_true', help='include all role labels instead of filtering out unexpected ones')
    parser.add_argument('--pretty', action='store_true', help='preserve output formatting instead of removing extra spaces')
    parser.set_defaults(noi=True)
    parser.set_defaults(all=False)
    parser.set_defaults(pretty=False)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def main():
    _opts = options()
    _props_by_path = _separate_props_by_file(_opts.pb)
    path_filter = None

    if _opts.filter:
        pattern = re.compile('^' + _opts.filter + '$', re.IGNORECASE)

        def filter_func(pathstr):
            return bool(pattern.match(pathstr))

        path_filter = filter_func

    link_treebank_propbank(_props_by_path, treebank_dir=_opts.tb, script_path=_opts.script, output_dir=_opts.o,
                           combined=_opts.combined, noi=_opts.noi, path_filter=path_filter, all_roles=_opts.all,
                           pretty=_opts.pretty)


if __name__ == '__main__':
    main()
