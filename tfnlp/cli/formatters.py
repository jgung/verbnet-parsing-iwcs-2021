from tfnlp.common import constants
from tfnlp.common.utils import binary_np_array_to_unicode


def get_formatter(config):
    def _tagger_formatter(result):
        target = config.features.targets[0].name
        formatted = []
        for labels in result[target]:
            formatted.append(' '.join(binary_np_array_to_unicode(labels)))
        return '\n'.join(formatted)

    def _classifier_formatter(result):
        target = config.features.targets[0].name
        return result[target][0].decode('utf-8')

    head_type = [head.type for head in config.heads][0]
    formatters = {
        constants.CLASSIFIER_KEY: _classifier_formatter,
        constants.TAGGER_KEY: _tagger_formatter,
        constants.NER_KEY: _tagger_formatter,
        constants.SRL_KEY: _tagger_formatter,
    }
    if head_type not in formatters:
        raise ValueError("Unsupported head type: " + head_type)
    return formatters[head_type]
