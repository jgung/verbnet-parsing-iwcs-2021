import argparse
import os
import re

REMOVE = [".*\\.tfrecords$", "^eval$", "^events.*", ".*ckpt-.*", "^checkpoint$", "graph.pbtxt"]


def clean_experiment_dir(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for pattern in REMOVE:
            for file in filter(lambda x: re.match(pattern, x), files):
                os.remove(os.path.join(root, file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Base path to clean', required=True)
    clean_experiment_dir(parser.parse_args().path)
