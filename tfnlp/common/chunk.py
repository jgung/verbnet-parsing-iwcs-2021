from tfnlp.common.constants import BEGIN, BEGIN_, CONLL_CONT, CONLL_END, CONLL_START, END, END_, IN, IN_, OUT, SINGLE, SINGLE_


def chunk(labeling, besio=False, conll=False):
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


def convert_conll_to_bio(labels):
    """
    Convert CoNLL-style sequence labels to BIO labels. [`(X`, `*` `)`] => [`B-X`, `I-X`, `I-X`]
    :param labels: list of CoNLL labels
    :return: list of BIO labels
    """

    def _get_label(_label):
        return _label.replace(CONLL_START, "").replace(CONLL_END, "").replace(CONLL_CONT, "")

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


def _to_conll(iob_label):
    label_type = iob_label
    for suffix in [BEGIN_, END_, SINGLE_, IN_]:
        label_type = label_type.replace(suffix, "")

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
