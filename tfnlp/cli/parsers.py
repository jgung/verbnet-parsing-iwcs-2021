from nltk import word_tokenize, sent_tokenize

from tfnlp.common import constants


def default_parser(sentence):
    def _feats(sent):
        tokens = word_tokenize(sent)
        return {constants.WORD_KEY: tokens, constants.LABEL_KEY: ['O'] * len(tokens)}

    sentences = sent_tokenize(sentence)
    return [_feats(sent) for sent in sentences]
