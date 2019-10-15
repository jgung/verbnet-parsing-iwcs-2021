import argparse
import os

from tfnlp.common.utils import write_json
from tfnlp.compute_stats import get_count_tuples, output_probs
from tfnlp.readers import get_reader


def generate_stats(counts, name, output):
    base_path = os.path.join(output, os.path.basename(name))

    sense_counts, roleset_counts = counts

    # compute counts/probabilities of individual senses
    output_probs(sense_counts, base_path + '.counts.tsv')
    # compute and output entropy of roleset distributions
    output_probs(roleset_counts, base_path + '.rs.counts.tsv')


def main(opts):
    os.makedirs(opts.output, exist_ok=True)

    ks = [int(k) for k in opts.k.split(',')]
    print('Processing on following thresholds: {}'.format(', '.join([str(k) for k in ks])))
    reader = get_reader(opts.reader)

    print("Reading corpus at %s..." % opts.train)
    train_tuples, _ = get_count_tuples(reader, opts.train, True)
    print("Reading corpus at %s..." % opts.dev)
    dev_tuples, _ = get_count_tuples(reader, opts.dev, True)

    train = {p: sum(c.values()) for p, c in train_tuples}

    for k in ks:
        test_count = 0
        result = []
        for pred, count in dev_tuples:
            train_count = train.get(pred, 0)
            count = sum(count.values())
            if train_count <= k:
                result.append(pred)
                test_count += count

        write_json({
            "include": result,
        }, opts.output + '/' + str(k) + '.json')
        print("Count for %d: %d" % (k, test_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Training corpus for computing predicate counts', required=True)
    parser.add_argument('--dev', type=str, help='Development corpus used to generate predicate filter based on counts',
                        required=True)
    parser.add_argument('--output', type=str, help='Output path', required=True)
    parser.add_argument('--reader', type=str, default='conll_2012', help='Reader type')
    parser.add_argument('--k', type=str, default='0', help='Thresholds to use for count in training data (0, or OOV by default)')
    main(parser.parse_args())
