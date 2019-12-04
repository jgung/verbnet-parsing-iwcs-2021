from collections import Counter

from tfnlp.common import constants
from tfnlp.corpus_count_filter import default_reader
from tfnlp.readers import get_reader


def main():
    base_path = "data/datasets/thesis/unified/all/"
    datasets = [base_path + "train.txt", base_path + "test.txt", base_path + "dev.txt"]

    reader = get_reader(default_reader())
    counter = Counter()
    for path in datasets:
        print("Reading from %s" % path)
        for instance in reader.read_file(path):
            result = instance[constants.PREDICATE_LEMMA] + "." + instance[constants.ROLESET_KEY]
            counter[result] += 1
    with open('roleset-counts.tsv', mode="w") as out:
        for k, c in counter.items():
            out.write("{}\t{}\n".format(k, c))


if __name__ == '__main__':
    main()
