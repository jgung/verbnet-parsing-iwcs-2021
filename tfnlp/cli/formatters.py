from tfnlp.common.eval import get_parse_prediction
from tfnlp.common import constants
from tfnlp.common.bert import BERT_SUBLABEL
from tfnlp.common.constants import WORD_KEY
from tfnlp.common.utils import binary_np_array_to_unicode


def get_formatter(config):
    def _tagger_formatter(result, original_input=None):
        target = config.features.targets[0].name

        labels = [res for res in binary_np_array_to_unicode(result[target]) if res != BERT_SUBLABEL]
        if not original_input or WORD_KEY not in original_input:
            return '\n'.join(labels)
        result = []
        for word, label in zip(original_input[WORD_KEY], labels):
            result.append(word + ' ' + label)
        return '\n'.join(result)

    def _classifier_formatter(result, ignored=None):
        target = config.features.targets[0].name
        prediction = result[target].decode('utf-8')
        return prediction

    def _parser_formatter(result, original_input):
        feats = next(iter([target for target in config.features.targets if target.name == constants.DEPREL_KEY]), None)
        fields = [[str(r) for r in range(1, len(original_input[WORD_KEY]) + 1)], original_input[WORD_KEY]]
        if constants.POS_KEY in result:
            fields.append(binary_np_array_to_unicode(result[constants.POS_KEY]))

        heads, labels = get_parse_prediction(result[constants.ARC_PROBS], result[constants.REL_PROBS], )
        if feats:
            feats = {val: key for key, val in feats.indices.items()}
            labels = [feats[rel] for rel in labels]

        fields = fields + [[str(l) for l in labels[1:]], [str(s) for s in heads[1:]]]
        return '\n'.join([' '.join(line) for line in zip(*fields)]) + '\n'

    head_type = [head.task for head in config.heads][0]
    formatters = {
        constants.CLASSIFIER_KEY: _classifier_formatter,
        constants.TAGGER_KEY: _tagger_formatter,
        constants.NER_KEY: _tagger_formatter,
        constants.SRL_KEY: _tagger_formatter,
        constants.PARSER_KEY: _parser_formatter,
        constants.TOKEN_CLASSIFIER_KEY: _classifier_formatter
    }
    if head_type not in formatters:
        raise ValueError("Unsupported head type: " + head_type)
    return formatters[head_type]
