import re
from typing import Iterable, List, Tuple

from tfnlp.common.constants import BEGIN, BEGIN_, CONLL_CONT, CONLL_END, CONLL_START, END, END_, IN, IN_, OUT, SINGLE, SINGLE_


def labels_to_spans(labeling: Iterable[str]) -> List[Tuple[str, int, int]]:
    """
    Given an IOB/BESIO chunking produce a list of labeled spans--triples of (label, start index, end index exclusive).
    >>> labels_to_spans(['O', 'B-PER', 'I-PER', 'O', 'B-ORG'])
    [('PER', 1, 3), ('ORG', 4, 5)]

    :param labeling: list of IOB/BESIO labels
    :return: list of spans
    """

    def _start_of_chunk(curr):
        curr_tag, _ = _get_val_and_tag(curr)
        return curr_tag in {'S', 'B'}

    def _end_of_chunk(curr):
        curr_tag, _ = _get_val_and_tag(curr)
        return curr_tag in {'E', 'S'}

    besio = chunk(labeling, besio=True)

    result = []
    curr_label, start = None, None
    for index, label in enumerate(besio):
        if _start_of_chunk(label):
            if curr_label:
                result.append((curr_label, start, index))
            curr_label, start = _get_val_and_tag(label)[1], index
        if _end_of_chunk(label):
            result.append((curr_label, start, index + 1))
            curr_label = None
    if curr_label:
        result.append((curr_label, start, len(besio)))

    return result


def spans_to_conll_labels(spans: List[Tuple[str, int, int]], length) -> List[str]:
    """
    Convert from span tuples to the CoNLL-2005/2012 chunk format.
    >>> spans_to_conll_labels([('PER', 1, 3), ('ORG', 4, 5)], length=6)
    ['*', '(PER*', '*)', '*', '(ORG*)', '*']

    :param spans: triples consisting of label, start index (inclusive), and end index (exclusive)
    :param length: number of tokens in original sentence
    :return: list of corresponding CoNLL-2005/2012 labels
    """
    result = []
    last = 0
    for label, start, end in spans:
        result.extend(['*'] * (start - last))

        span_len = end - start
        first = '(' + label + '*'
        if span_len == 1:
            result.append(first + ')')
        else:
            result.append(first)
            result.extend(['*'] * (span_len - 2))
            result.append('*)')

        last = end

    result.extend(['*'] * (length - len(result)))

    return result


def chunk(labeling: Iterable[str], besio=False, conll=False) -> List[str]:
    """
    Convert an IO/BIO/BESIO-formatted sequence of labels to BIO, BESIO, or CoNLL-2005 formatted.
    :param labeling: original labels
    :param besio: (optional) convert to BESIO format, `False` by default
    :param conll: (optional) convert to CoNLL-2005 format, `False` by default
    :return: converted labels
    """
    if conll:
        besio = True
    result = []
    prev_type = None
    curr = []
    for label in labeling:
        if label == OUT:
            state, chunk_type = OUT, ''
        else:
            split_index = label.index('-')
            state, chunk_type = label[:split_index], label[split_index + 1:]
        if state == IN and chunk_type != prev_type:  # new chunk of different type
            state = BEGIN
        if state in [BEGIN, OUT] and curr:  # end of chunk
            result += _to_besio(curr) if besio else curr
            curr = []
        if state == OUT:
            result.append(state)
        else:
            curr.append(state + "-" + chunk_type)
        prev_type = chunk_type
    if curr:
        result += _to_besio(curr) if besio else curr
    if conll:
        result = [_to_conll(label) for label in result]
    return result


def convert_conll_to_bio(labels, label_mappings=None, map_with_regex=False):
    """
    Convert CoNLL-style sequence labels to BIO labels. [`(X`, `*` `)`] => [`B-X`, `I-X`, `I-X`]
    :param labels: list of CoNLL labels
    :param label_mappings: dict mapping labels
    :param map_with_regex: if `True`, treat mappings as regular expressions
    :return: list of BIO labels
    """

    def _get_label(_label):
        result = _label.replace(CONLL_START, "").replace(CONLL_END, "").replace(CONLL_CONT, "")
        if label_mappings is not None:
            if map_with_regex:
                for search, repl in label_mappings.items():
                    match = re.search(search, result)
                    if match is not None:
                        return re.sub(search, repl, result)
            return label_mappings.get(result, result)
        return result

    current = None
    results = []
    for token in labels:
        if token.startswith(CONLL_START):
            label = _get_label(token)
            results.append(BEGIN_ + label)
            current = label
        elif current and CONLL_CONT in token:
            results.append(IN_ + current)
        else:
            results.append(OUT)

        if token.endswith(CONLL_END):
            current = None
    return results


def _to_besio(iob_labeling):
    if len(iob_labeling) == 1:
        return [SINGLE + iob_labeling[0][1:]]
    return iob_labeling[:-1] + [END + iob_labeling[-1][1:]]


BESIO_PREFIX = '^[BESI]-'


def _to_conll(iob_label):
    label_type = iob_label
    label_type = re.sub(BESIO_PREFIX, '', label_type)

    if iob_label.startswith(BEGIN_):
        return "(" + label_type + "*"
    if iob_label.startswith(SINGLE_):
        return "(" + label_type + "*)"
    if iob_label.startswith(END_):
        return "*)"
    return "*"


def chunk_besio(labeling):
    return chunk(labeling, besio=True)


def chunk_conll(labeling):
    return chunk(labeling, conll=True)


def end_of_chunk(prev, curr):
    prev_val, prev_tag = _get_val_and_tag(prev)
    curr_val, curr_tag = _get_val_and_tag(curr)
    if prev_val == OUT:
        return True
    if not prev_val:
        return False
    if prev_tag != curr_tag or prev_val == 'E' or curr_val == 'B' or curr_val == 'O' or prev_val == 'O':
        return True
    return False


def start_of_chunk(prev, curr):
    prev_val, prev_tag = _get_val_and_tag(prev)
    curr_val, curr_tag = _get_val_and_tag(curr)
    if prev_tag != curr_tag or curr_val == 'B' or curr_val == 'O':
        return True
    return False


def _get_val_and_tag(label):
    if not label:
        return '', ''
    if label == 'O':
        return label, ''
    return label.split('-', 1)
