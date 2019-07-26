from nltk import word_tokenize, sent_tokenize

from tfnlp.common import constants


def get_parser(config):
    head_type = [head.task for head in config.heads][0]
    parsers = {
        constants.TAGGER_KEY: default_parser,
        constants.SRL_KEY: srl_parser,
        constants.NER_KEY: default_parser,
        constants.PARSER_KEY: dep_input_parser,
    }
    if head_type not in parsers:
        raise ValueError("ITL unsupported head type: " + head_type)
    return parsers[head_type]


def default_parser(sentence):
    def _feats(sent):
        tokens = word_tokenize(sent)
        return {constants.WORD_KEY: tokens, constants.LABEL_KEY: ['O'] * len(tokens)}

    sentences = sent_tokenize(sentence)
    return [_feats(sent) for sent in sentences]


def dep_input_parser(sentence):
    def _feats(sent):
        tokens = word_tokenize(sent)
        return {constants.WORD_KEY: tokens, constants.HEAD_KEY: [0] * len(tokens)}

    sentences = sent_tokenize(sentence)
    return [_feats(sent) for sent in sentences]


def srl_parser(sentence):
    sentences = sent_tokenize(sentence)

    results = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        words = [token[1:] if token[0] == '+' else token for token in tokens]
        for index, token in enumerate(tokens):
            if token.startswith('+'):
                markers = ['1' if i == index else '0' for i in range(0, len(tokens))]
                feats = {constants.WORD_KEY: words,
                         constants.LABEL_KEY: ['O'] * len(tokens),
                         'ft': ['O'] * len(tokens),
                         constants.MARKER_KEY: markers,
                         constants.PREDICATE_INDEX_KEY: index}
                results.append(feats)

    return results
