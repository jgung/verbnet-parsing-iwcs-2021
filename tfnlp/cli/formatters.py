from tfnlp.common import constants
from tfnlp.common.constants import WORD_KEY
from tfnlp.common.utils import binary_np_array_to_unicode


def get_formatter(config):
    def _tagger_formatter(result, original_input=None):
        target = config.features.targets[0].name

        labels = binary_np_array_to_unicode(result[target])
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

    def _no_op_formatter(result, ignored=None):
        return str(result)

    head_type = [head.type for head in config.heads][0]
    formatters = {
        constants.CLASSIFIER_KEY: _classifier_formatter,
        constants.TAGGER_KEY: _tagger_formatter,
        constants.NER_KEY: _tagger_formatter,
        constants.SRL_KEY: _tagger_formatter,
        constants.BIAFFINE_SRL_KEY: _tagger_formatter,
        constants.PARSER_KEY: _no_op_formatter
    }
    if head_type not in formatters:
        raise ValueError("Unsupported head type: " + head_type)
    return formatters[head_type]
