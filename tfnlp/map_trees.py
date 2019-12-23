from tfnlp.common import constants
from tfnlp.common.chunk import chunk
from tfnlp.corpus_count_filter import semlink_reader, default_reader
from tfnlp.readers import get_reader

SEMLINK_OUTPUT_FIELDS = [constants.INSTANCE_INDEX, constants.TOKEN_INDEX_KEY, constants.WORD_KEY, constants.POS_KEY,
                         constants.DEPREL_KEY, constants.HEAD_KEY, constants.SENSE_KEY, constants.PREDICATE_KEY]


def write_instances(inst, propositions, writer):
    output_fields = list(SEMLINK_OUTPUT_FIELDS)
    senses = ['-' for _ in inst[constants.WORD_KEY]]
    predicates = ['-' for _ in inst[constants.WORD_KEY]]
    inst[constants.SENSE_KEY] = senses
    inst[constants.PREDICATE_KEY] = predicates

    for prop in propositions:
        k = prop[constants.PREDICATE_INDEX_KEY]
        output_fields.append(k)
        inst[k] = prop[constants.LABEL_KEY]
        senses[k] = prop[constants.SENSE_KEY]
        predicates[k] = prop[constants.PREDICATE_KEY]

    for i, fields in enumerate(zip(*[inst[k] for k in output_fields])):
        writer.write('%s\n' % (' '.join([str(f) for f in fields])))
    writer.write('\n')


if __name__ == '__main__':
    parsed_path = "data/datasets/thesis/conll2005/train.dep.txt"
    props_path = "data/datasets/thesis/semlink/train.txt"

    parsed_reader = get_reader(default_reader())
    parsed_instances = {}
    for instance in parsed_reader.read_file(parsed_path):
        parsed_instances[' '.join(instance[constants.WORD_KEY])] = instance

    semlink_reader = get_reader(semlink_reader())

    total = 0
    with open(props_path + '.dep.txt', 'w') as out_file:
        cur_sent = None
        prev_key = ''
        props = []
        for instance in semlink_reader.read_file(props_path):
            key = ' '.join(instance[constants.WORD_KEY])
            if key != prev_key:
                if cur_sent is not None:
                    cur_sent[constants.INSTANCE_INDEX] = [cur_sent[constants.INSTANCE_INDEX] for _ in
                                                          cur_sent[constants.WORD_KEY]]
                    cur_sent[constants.SENTENCE_INDEX] = [cur_sent[constants.SENTENCE_INDEX] for _ in
                                                          cur_sent[constants.WORD_KEY]]
                    mapped = parsed_instances.get(prev_key)
                    if not mapped:
                        print(key)
                        continue
                    else:
                        cur_sent[constants.DEPREL_KEY] = mapped[constants.DEPREL_KEY]
                        cur_sent[constants.HEAD_KEY] = mapped[constants.HEAD_KEY]
                    write_instances(cur_sent, props, out_file)
                    total += len(props)
                    cur_sent = None
                    props = []
                cur_sent = instance
            prev_key = key
            props.append({
                constants.LABEL_KEY: chunk(instance[constants.LABEL_KEY], conll=True),
                constants.SENSE_KEY: instance[constants.SENSE_KEY],
                constants.PREDICATE_INDEX_KEY: instance[constants.PREDICATE_INDEX_KEY],
                constants.PREDICATE_KEY: instance[constants.PREDICATE_KEY][instance[constants.PREDICATE_INDEX_KEY]]
                          })
        if props:
            cur_sent[constants.INSTANCE_INDEX] = [cur_sent[constants.INSTANCE_INDEX] for _ in
                                                  cur_sent[constants.WORD_KEY]]
            cur_sent[constants.SENTENCE_INDEX] = [cur_sent[constants.SENTENCE_INDEX] for _ in
                                                  cur_sent[constants.WORD_KEY]]
            key = ' '.join(cur_sent[constants.WORD_KEY])
            mapped = parsed_instances.get(key)
            if not mapped:
                print(key)
            else:
                cur_sent[constants.DEPREL_KEY] = mapped[constants.DEPREL_KEY]
                cur_sent[constants.HEAD_KEY] = mapped[constants.HEAD_KEY]
            write_instances(cur_sent, props, out_file)
            total += len(props)
        print("Mapped %d props" % total)
