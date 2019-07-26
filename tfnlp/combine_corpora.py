from tfnlp.common import constants
from tfnlp.readers import ConllReader

PROPS_KEY = "_PROPS"

FIELDS = [constants.ID_KEY, constants.INSTANCE_INDEX, constants.TOKEN_INDEX_KEY, constants.WORD_KEY, constants.POS_KEY,
          constants.PARSE_KEY, constants.PREDICATE_KEY, constants.SENSE_KEY]


class CoNll2012Reader(ConllReader):

    def read_instances(self, rows):
        """
        Read a list of instances from the given list of row strings.
        :param rows: CoNLL rows
        :return: list of instances
        """
        fields = self.read_fields(rows)

        propositions = {}
        for i, (idx, predicate) in enumerate(
                [(k, pred) for k, pred in enumerate(fields[constants.PREDICATE_KEY]) if pred not in ['–', '-']]):
            labels = []
            for row in rows:
                labels.append(row[11 + i])
            propositions[idx] = labels

        fields[PROPS_KEY] = propositions
        return [fields]


def write_instance(inst, writer):
    prop_cols = sorted(inst[PROPS_KEY].items(), key=lambda x: x[0])
    for i, fields in enumerate(zip(*[inst[k] for k in FIELDS])):
        propositions = ' '.join([col[1][i] for col in prop_cols])
        writer.write('%s - - * %s\n' % (' '.join(fields), propositions))
    writer.write('\n')


if __name__ == '__main__':
    base_path = 'data/datasets/lre2019/'
    inputs = [
        base_path + 'all_test.FUNCTIONTAG.conll',
        base_path + 'all_test.VERBAL.conll',
        base_path + 'all_test.NOMINAL.conll',
        base_path + 'all_test.ADJECTIVAL.conll'
    ]
    out_path = base_path + 'all.test.FVNA.txt'

    reader = CoNll2012Reader({k: v for k, v in enumerate(FIELDS)})

    with open(out_path, mode='wt') as out_file:
        for ft_view, verb_view, noun_view, adj_view in zip(*(reader.read_file(path) for path in inputs)):
            props = ft_view[PROPS_KEY]
            newInstance = {**ft_view}

            indices = set()

            def update_props(view):
                view_props = view[PROPS_KEY]
                for pred_idx, newLabels in view_props.items():
                    indices.add(pred_idx)
                    props[pred_idx] = [ol + "$" + nl for ol, nl in zip(props[pred_idx], newLabels)]

            update_props(verb_view)
            update_props(noun_view)
            update_props(adj_view)

            for pred_index in list(props.keys()):
                if pred_index not in indices:
                    del props[pred_index]
                    newInstance[constants.SENSE_KEY][pred_index] = '-'
                    newInstance[constants.PREDICATE_KEY][pred_index] = '-'

            write_instance(newInstance, out_file)