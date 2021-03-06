import argparse
import os

from tfnlp.common import constants
from tfnlp.common.chunk import chunk
from tfnlp.common.utils import read_json
from tfnlp.corpus_count_filter import SEMLINK_OUTPUT_FIELDS, default_reader, semlink_reader, OUTPUT_FIELDS
from tfnlp.readers import get_reader


def read_corpus(reader, dataset):
    result = list(reader.read_file(dataset))

    updated = []
    for instance in result:
        instance[constants.INSTANCE_INDEX] = [instance[constants.INSTANCE_INDEX] for _ in instance[constants.WORD_KEY]]
        instance[constants.SENTENCE_INDEX] = [instance[constants.SENTENCE_INDEX] for _ in instance[constants.WORD_KEY]]
        instance[constants.LABEL_KEY] = chunk(instance[constants.LABEL_KEY], conll=True)
        pred_idx = instance[constants.PREDICATE_INDEX_KEY]
        instance[constants.SENSE_KEY] = [instance[constants.SENSE_KEY] if i == pred_idx else '-'
                                         for i in range(0, len(instance[constants.WORD_KEY]))]
        instance[constants.PREDICATE_KEY] = [instance[constants.PREDICATE_KEY][pred_idx] if i == pred_idx else '-'
                                             for i in range(0, len(instance[constants.PREDICATE_KEY]))]
        updated.append(instance)
    return updated


def write_instance(inst, writer, fields=None, ner=True):
    if fields is None:
        fields = OUTPUT_FIELDS
    for i, fields in enumerate(zip(*[inst[k] for k in fields])):
        propositions = inst[constants.LABEL_KEY][i]
        writer.write('%s %s%s\n' % (' '.join([str(f) for f in fields]), '- - * ' if ner else '', propositions))
    writer.write('\n')


def main(opts):
    os.makedirs(opts.output, exist_ok=True)

    try:
        if opts.reader == "semlink":
            reader = get_reader(semlink_reader())
        else:
            reader = get_reader(default_reader() if opts.reader is None else opts.reader)
    except ValueError:
        reader = get_reader(read_json(opts.reader))

    print("Reading corpus at %s..." % opts.train)
    train_instances = read_corpus(reader, opts.train)

    sizes = [float(x) for x in opts.k.split(',') if x.strip()]
    for size in sizes:
        test_count = 0

        with open(opts.output + '/' + str(size).replace(".", "_") + '.txt', mode='wt') as out_file:
            for inst in train_instances[:int(len(train_instances) * size)]:
                test_count += 1
                write_instance(inst, out_file, SEMLINK_OUTPUT_FIELDS if opts.reader == "semlink" else None,
                               ner=opts.reader != "semlink")
        print("Count %f: %d" % (size, test_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Training corpus for computing predicate counts', required=True)
    parser.add_argument('--output', type=str, help='Output path', required=True)
    parser.add_argument('--reader', type=str, help='Reader type')
    parser.add_argument('--k', type=str, default='0.01,0.033,0.066,0.1,0.333,0.666',
                        help='Percentages to bin by (default="0.01,0.033,0.066,0.1,0.333,0.666")')
    main(parser.parse_args())
