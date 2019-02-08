import argparse
import math
import os
from collections import defaultdict, Counter

from tfnlp.common.constants import SENSE_KEY, PREDICATE_KEY, PREDICATE_INDEX_KEY, LABEL_KEY
from tfnlp.readers import get_reader

CORE_ROLES = {'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA'}


def get_counts(instances, key1_func, key2_func):
    counts = defaultdict(Counter)
    for instance in instances:
        key1 = key1_func(instance)  # e.g. predicate
        key2 = key2_func(instance)  # e.g. sense

        key1_counts = counts[key1]
        key1_counts[key2] += 1

    return counts


def get_role_counts(instances, only_core=False):
    counts = defaultdict(Counter)
    for instance in instances:
        labels = [label[2:] for label in instance[LABEL_KEY] if label.startswith('B-') and label != 'B-V']
        if only_core:
            labels = [label for label in labels if label in CORE_ROLES]
        predicate = instance[PREDICATE_KEY][instance[PREDICATE_INDEX_KEY]]
        for label in labels:
            counts[predicate][label] += 1
    return counts


def probabilities(counts):
    total = sum(counts.values())
    return {k: v / total for (k, v) in counts.items()}


def entropy(counts):
    total = 0
    for k, v in probabilities(counts).items():
        total += v * math.log2(v)
    return -total


def get_count_tuples(reader, dataset, only_core=False):
    result = list(reader.read_file(dataset))

    predicate_counts = get_counts(result,
                                  lambda i: i[PREDICATE_KEY][i[PREDICATE_INDEX_KEY]],
                                  lambda i: next(iter([s for s in i[SENSE_KEY] if s is not '-']), '-'))

    roleset_counts = get_role_counts(result, only_core)

    rs_tuples = [(p, sum(c.values()), c) for p, c in roleset_counts.items()]
    rs_tuples = [(p, c) for p, _, c in sorted(rs_tuples, key=lambda x: x[1])]

    count_tuples = [(p, sum(c.values()), c) for p, c in predicate_counts.items()]
    count_tuples = [(p, c) for p, _, c in sorted(count_tuples, key=lambda x: x[1])]

    return count_tuples, rs_tuples


def kl(dist1, dist2):
    result = 0
    for k, dist1_k in dist1.items():
        dist2_k = dist2[k]
        result += dist1_k * math.log2(dist1_k / dist2_k)
    return result


def smooth(keys, target_dist, add_smoothing=1):
    smoothed = {k: v + 1 for k, v in target_dist.items()}
    for k in keys:
        if k not in target_dist:
            smoothed[k] = add_smoothing
    return smoothed


def output_entropy(counts, output_path):
    with open(output_path, 'wt') as out:
        for predicate, sense_counts in counts:
            ent = entropy(sense_counts)
            out.write('{}\t{}\n'.format(predicate, ent))


def output_probs(counts, output_path):
    with open(output_path, 'wt') as out:
        for predicate, sense_counts in counts:
            probs = probabilities(sense_counts)
            for sense, prob in sorted(probs.items(), key=lambda x: x[1]):
                count = sense_counts[sense]
                out.write("{}.{}\t{}\t{}\n".format(predicate, sense, prob, count))


def output_kl(counts1, counts2, output_path):
    with open(output_path, 'wt') as out:
        for predicate, sense_counts in counts1:
            probs = probabilities(smooth(sense_counts.keys(), sense_counts))
            for predicate2, sense_counts2 in counts2:
                if predicate2 != predicate:
                    continue
                sense_counts2 = smooth(sense_counts.keys(), sense_counts2)
                probs2 = probabilities(sense_counts2)
                kl_divergence = kl(probs, probs2)
                out.write('{}\t{}\n'.format(predicate, kl_divergence))


def average_kl(counts1, counts2, predicates=None):
    total_counts = 0
    total_kl = 0
    for predicate, sense_counts in counts1:
        if predicates and predicate not in predicates:
            continue
        sense_counts = smooth(sense_counts.keys(), sense_counts)
        probs = probabilities(sense_counts)
        predicate_counts = sum(sense_counts.values())

        total_counts += predicate_counts
        for predicate2, sense_counts2 in counts2:
            if predicate2 != predicate:
                continue
            sense_counts2 = smooth(sense_counts.keys(), sense_counts2)
            probs2 = probabilities(sense_counts2)
            total_kl += kl(probs, probs2) * predicate_counts
    if total_counts == 0:
        return -1
    return total_kl / total_counts


def average_entropy(counts, predicates):
    total_entropy = 0
    total_count = 0
    for predicate, sense_counts in counts:
        if predicate not in predicates:
            continue
        count = sum(sense_counts.values())
        ent = entropy(sense_counts) * count
        total_count += count
        total_entropy += ent
    if total_count == 0:
        return -1
    return total_entropy / total_count


def entropies(counts, predicates=None):
    total = 0
    marginals = Counter()
    for predicate, predicate_counts in counts:
        if predicates and predicate not in predicates:
            continue
        for k, v in predicate_counts.items():
            marginals[k] += v
            total += v

    pred_marginal_entropy = 0  # H(predicate)
    joint_entropy = 0  # H(predicate,roleset)
    for predicate, predicate_counts in counts:
        if predicates and predicate not in predicates:
            continue
        for p in [c / total for c in predicate_counts.values()]:
            joint_entropy -= p * math.log2(p)
        predicate_prob = sum(predicate_counts.values()) / total
        pred_marginal_entropy -= predicate_prob * math.log2(predicate_prob)

    marginal_entropy = 0
    for p in [c / total for c in marginals.values()]:
        marginal_entropy -= p * math.log2(p)

    conditional_entropy = joint_entropy - pred_marginal_entropy  # H(roleset,predicate)
    return conditional_entropy, joint_entropy, marginal_entropy


def get_top_k_predicates(counts, k=100):
    filtered = [(p, sum(c.values()), c) for p, c in counts]
    filtered = [p for p, _, c in sorted(filtered, key=lambda x: x[1], reverse=True)][:k]
    return set(filtered)


def get_predicate_count(counts):
    total = 0
    for predicate, sense_counts in counts:
        total += sum(sense_counts.values())
    return total


def output_summary(counts, output_path):
    total = 0
    with open(output_path, 'wt') as out:
        for predicate, sense_counts in counts:
            count = sum(sense_counts.values())
            total += count
            out.write('{}\t{}\n'.format(predicate, count))
        out.write('Total\t{}\n'.format(total))


def generate_stats(counts, name, output):
    base_path = os.path.join(output, os.path.basename(name))

    sense_counts, roleset_counts = counts

    # compute and output entropy of sense distributions
    output_entropy(sense_counts, base_path + '.sense.entropy.tsv')
    # compute and output entropy of roleset distributions
    output_entropy(roleset_counts, base_path + '.rs.entropy.tsv')

    # compute counts/probabilities of individual senses
    output_probs(sense_counts, base_path + '.counts.tsv')
    # compute and output entropy of roleset distributions
    output_probs(roleset_counts, base_path + '.rs.counts.tsv')

    output_summary(sense_counts, base_path + '.summary.tsv')


def kl_pairs(corpora, counts, output, k):
    pairs = []
    for i in range(0, len(corpora) - 1, 2):
        pairs.append((corpora[i], corpora[i + 1]))

    top_predicates = get_top_k_predicates(counts[pairs[0][0]][0], k) if k else None

    with open(os.path.join(output, 'kl-{}.tsv'.format(k)), 'wt') as out:
        out.write('Posterior\tPrior\tSense KL\tRS KL\n')
        for posterior, prior in pairs:
            posterior_sense_counts, posterior_rs_counts = counts[posterior]
            prior_sense_counts, prior_rs_counts = counts[prior]

            sense_kl = average_kl(posterior_sense_counts, prior_sense_counts, predicates=top_predicates)
            rs_kl = average_kl(posterior_rs_counts, prior_rs_counts, predicates=top_predicates)

            posterior = os.path.basename(posterior)
            prior = os.path.basename(prior)
            out.write('{}\t{}\t{}\t{}\n'.format(posterior, prior, sense_kl, rs_kl))

            output_kl(posterior_sense_counts, prior_sense_counts,
                      os.path.join(output, 'kl-{}-{}-sense.tsv'.format(posterior, prior)))
            output_kl(posterior_rs_counts, prior_rs_counts,
                      os.path.join(output, 'kl-{}-{}-rolesets.tsv'.format(posterior, prior)))


def summary_stats(corpora, counts, output, k):
    top_predicates = get_top_k_predicates(counts[corpora[0]][0], k)

    with open(os.path.join(output, 'summary-{}.tsv'.format(k)), 'wt') as out:
        out.write('Corpus\tAverage Sense Entropy\tAverage RS Entropy\tRS Entropy\tRS Conditional Entropy\tRS Joint Entropy\t'
                  '# Predicates\n')
        for corpus in corpora:
            avg_entropy = average_entropy(counts[corpus][0], top_predicates)
            conditional_entropy, joint_entropy, marginal_entropy = entropies(counts[corpus][1])
            avg_rs_entropy = average_entropy(counts[corpus][1], top_predicates)
            total = get_predicate_count(counts[corpus][0])
            out.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(os.path.basename(corpus), avg_entropy, avg_rs_entropy,
                                                            marginal_entropy, conditional_entropy, joint_entropy, total))


def main(opts):
    os.makedirs(opts.output, exist_ok=True)

    reader = get_reader(opts.reader)
    corpora = [corpus for corpus in opts.corpora.split(',') if corpus.strip()]
    counts = {}
    for name in corpora:
        if name not in counts:
            print("Reading corpus at %s..." % name)
            counts[name] = get_count_tuples(reader, name, opts.core)

    if opts.kl:
        kl_pairs(corpora, counts, opts.output, opts.k)
    summary_stats(corpora, counts, opts.output, opts.k)

    for corpus in corpora:
        generate_stats(counts[corpus], corpus, opts.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpora', type=str, help='Comma-separated paths to corpora', required=True)
    parser.add_argument('--output', type=str, help='Path to save location for corpus statistics', required=True)
    parser.add_argument('--reader', type=str, default='conll_2012', help='Reader type')
    parser.add_argument('--kl', action='store_true',
                        help='Compute KL divergence between consecutive pairs of comma-separated corpus paths')
    parser.add_argument('--k', type=int, help='Top 100 predicates to consider')
    parser.add_argument('--core', action='store_true', help='Only count core roles')
    parser.set_defaults(kl=False)
    parser.set_defaults(core=False)
    main(parser.parse_args())
