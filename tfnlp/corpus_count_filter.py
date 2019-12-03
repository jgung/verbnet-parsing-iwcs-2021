import argparse
import os
from collections import defaultdict

from tfnlp.common import constants
from tfnlp.common.chunk import chunk
from tfnlp.common.utils import read_json
from tfnlp.readers import get_reader


OUTPUT_FIELDS = [constants.ID_KEY, constants.INSTANCE_INDEX, constants.TOKEN_INDEX_KEY, constants.WORD_KEY, constants.POS_KEY,
                 constants.PARSE_KEY, constants.PREDICATE_KEY, constants.SENSE_KEY, constants.DEPREL_KEY, constants.HEAD_KEY]


def props_by_pred(reader, dataset):
    result = list(reader.read_file(dataset))

    by_pred = defaultdict(list)
    for instance in result:
        instance[constants.INSTANCE_INDEX] = [instance[constants.INSTANCE_INDEX] for _ in instance[constants.WORD_KEY]]
        instance[constants.SENTENCE_INDEX] = [instance[constants.SENTENCE_INDEX] for _ in instance[constants.WORD_KEY]]
        instance[constants.LABEL_KEY] = chunk(instance[constants.LABEL_KEY], conll=True)
        pred = by_pred[instance[constants.PREDICATE_KEY][instance[constants.PREDICATE_INDEX_KEY]]]
        pred.append(instance)
    return by_pred


def write_instance(inst, writer, fields=None):
    if fields is None:
        fields = OUTPUT_FIELDS
    for i, fields in enumerate(zip(*[inst[k] for k in fields])):
        propositions = inst[constants.LABEL_KEY][i]
        writer.write('%s - - * %s\n' % (' '.join([str(f) for f in fields]), propositions))
    writer.write('\n')


def main(opts):
    os.makedirs(opts.output, exist_ok=True)

    if '-' in opts.k:
        start, end = opts.k.split('-')
        ks = range(int(start), int(end) + 1)
    else:
        ks = [int(k) for k in opts.k.split(',')]
    print('Processing on following thresholds: {}'.format(', '.join([str(k) for k in ks])))
    try:
        reader = get_reader(opts.reader)
    except ValueError:
        reader = get_reader(read_json(opts.reader))

    print("Reading corpus at %s..." % opts.train)
    train_instances = props_by_pred(reader, opts.train)
    print("Reading corpus at %s..." % opts.dev)
    dev_instances = props_by_pred(reader, opts.dev)

    for k in ks:
        test_count = 0

        with open(opts.output + '/' + str(k) + '.txt', mode='wt') as out_file:
            for pred, devs in dev_instances.items():
                train_count = train_instances.get(pred, [])
                count = len(devs)
                if len(train_count) <= k:
                    test_count += count
                    for inst in devs:
                        write_instance(inst, out_file)
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
