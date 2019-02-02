import argparse
import glob
import os
import re
import sys

feature_matcher = re.compile(r'Extracting features from (\S+)')
overall_matcher = re.compile(r'Overall.*\s+(\S+)$')


def options():
    parser = argparse.ArgumentParser(description="Convert log files to tab-separated file w/ performance at each epoch.")
    parser.add_argument('--log', type=str, required=True, help='log file')
    parser.add_argument('--o', type=str, help='(optional) output path')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def compute_lines(log_file):
    results = []
    test_file = ''
    epoch = 1
    for line in open(log_file):
        score = overall_matcher.search(line)
        if score:
            results.append('%d\t%s\t%s\n' % (epoch, test_file, score.group(1)))
            epoch += 1
            continue

        feats = feature_matcher.search(line)
        if feats:
            epoch = 0
            test_file = feats.group(1)
    return results


if __name__ == '__main__':
    opts = options()
    log_files = []

    if os.path.isdir(opts.log):
        log_files.extend(glob.glob("%s/*.log" % opts.log))
    else:
        log_files.append(opts.log)

    lines = []
    for f in log_files:
        lines.extend(compute_lines(f))

    output = opts.o if opts.o else opts.log + '.stats.txt'
    with open(output, 'wt') as out:
        out.writelines(lines)
