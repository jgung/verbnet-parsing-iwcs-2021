import argparse
import math
import os
from collections import defaultdict, Counter

from tfnlp.common.constants import SENSE_KEY, PREDICATE_KEY, TOKEN_INDEX_KEY
from tfnlp.readers import get_reader


def get_counts(instances, key1_func, key2_func):
    counts = defaultdict(Counter)
    for instance in instances:
        key1 = key1_func(instance)  # e.g. predicate
        key2 = key2_func(instance)  # e.g. sense

        key1_counts = counts[key1]
        key1_counts[key2] += 1

    return counts


def probabilities(counts):
    total = sum(counts.values())
    return {k: v / total for (k, v) in counts.items()}


def entropy(counts):
    total = 0
    for k, v in probabilities(counts).items():
        total += v * math.log(v)
    return -total


def get_count_tuples(reader, dataset):
    result = reader.read_file(dataset)

    predicate_counts = get_counts(result,
                                  lambda i: i[PREDICATE_KEY][i[TOKEN_INDEX_KEY]],
                                  lambda i: next(iter([s for s in i[SENSE_KEY] if s is not '-']), '-'))

    count_tuples = [(p, sum(c.values()), c) for p, c in predicate_counts.items()]
    count_tuples = [(p, c) for p, _, c in sorted(count_tuples, key=lambda x: x[1])]

    return count_tuples


def kl(dist1, dist2):
    result = 0
    for k, dist1_k in dist1.items():
        dist2_k = dist2[k]
        result += dist1_k * math.log(dist1_k/dist2_k)
    return result


def smooth(keys, target_dist, add_smoothing=1):
    for k in keys:
        if k in target_dist:
            target_dist[k] += add_smoothing
        else:
            target_dist[k] = add_smoothing
    for k in target_dist.keys():
        if k not in keys:
            target_dist[k] += add_smoothing


def main(opts):
    reader = get_reader(opts.reader)

    counts1 = get_count_tuples(reader, opts.first)
    counts2 = get_count_tuples(reader, opts.second)

    os.makedirs(opts.output, exist_ok=True)

    # compute and output KL divergence stats
    with open(os.path.join(opts.output, 'kl.tsv'), 'wt') as out:
        for predicate, sense_counts in counts1:
            probs = probabilities(sense_counts)
            for predicate2, sense_counts2 in counts2:
                if predicate2 != predicate:
                    continue
                smooth(sense_counts.keys(), sense_counts2)
                probs2 = probabilities(sense_counts2)
                kl_divergence = kl(probs, probs2)
                out.write('{}\t{}\n'.format(predicate, kl_divergence))

    # compute and output entropy of sense distributions
    with open(os.path.join(opts.output, os.path.basename(opts.first) + '.entropy.tsv'), 'wt') as out:
        for predicate, sense_counts in counts1:
            ent = entropy(sense_counts)
            out.write('{}\t{}\n'.format(predicate, ent))
    with open(os.path.join(opts.output, os.path.basename(opts.second) + '.entropy.tsv'), 'wt') as out:
        for predicate, sense_counts in counts2:
            ent = entropy(sense_counts)
            out.write('{}\t{}\n'.format(predicate, ent))
    # compute counts/probabilities of individual senses
    with open(os.path.join(opts.output, os.path.basename(opts.first) + '.counts.tsv'), 'wt') as out:
        for predicate, sense_counts in counts1:
            probs = probabilities(sense_counts)
            for sense, count in sorted(probs.items(), key=lambda x: x[1]):
                out.write("{}.{}\t{}\n".format(predicate, sense, count))
    with open(os.path.join(opts.output, os.path.basename(opts.second) + '.counts.tsv'), 'wt') as out:
        for predicate, sense_counts in counts2:
            probs = probabilities(sense_counts)
            for sense, count in sorted(probs.items(), key=lambda x: x[1]):
                out.write("{}.{}\t{}\n".format(predicate, sense, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', type=str, help='Prior corpus (compute KL divergence of this corpus stats)', required=True)
    parser.add_argument('--second', type=str, help='Posterior corpus (compute KL divergence of prior corpus form this corpus',
                        required=True)
    parser.add_argument('--output', type=str, help='Output directory where to save stats', required=True)
    parser.add_argument('--reader', type=str, default='conll_2012', help='Reader type')
    main(parser.parse_args())

