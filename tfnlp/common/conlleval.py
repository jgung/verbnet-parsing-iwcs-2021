"""
CoNLL 2000/2003 evaluation script adapted to Python from original perl script available at
https://www.clips.uantwerpen.be/conll2003/ner/. Original author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>
"""

import argparse
import re
from collections import Counter

BIES = {'B-', 'I-', 'E-', 'S-'}


class ConllEvaluation(object):

    def __init__(self, correct_chunk, found_correct, found_guessed, n_tokens, accuracy):
        super().__init__()
        self.gold_count = correct_chunk
        self.correct_count = found_correct
        self.predicted_count = found_guessed

        self.n_tokens = n_tokens
        self.accuracy = accuracy
        self.gold_total = sum((val for label, val in correct_chunk.items()))
        self.correct_total = sum((val for label, val in found_correct.items()))
        self.predicted_total = sum((val for label, val in found_guessed.items()))

        self.precision = {}
        self.recall = {}
        self.fb1 = {}

        # sort chunk type names
        last_type = None  # temporary storage for detecting duplicates
        self.sorted_types = []
        for label in sorted((*found_correct.keys(), *found_guessed.keys())):
            if not last_type or last_type != label:
                self.sorted_types.append(label)
            last_type = label

        for label in self.sorted_types:
            if label not in found_guessed:
                self.precision[label] = 0
            else:
                self.precision[label] = 100 * correct_chunk[label] / found_guessed[label]
            if label not in found_correct:
                self.recall[label] = 0
            else:
                self.recall[label] = 100 * correct_chunk[label] / found_correct[label]
            if self.precision[label] + self.recall[label] == 0:
                self.fb1[label] = 0
            else:
                self.fb1[label] = 2 * self.precision[label] * self.recall[label] / (
                        self.precision[label] + self.recall[label])

        # compute overall precision, recall and FB1 (default values are 0.0)
        self.overall_precision, self.overall_recall, self.overall_fb1 = 0, 0, 0
        if self.predicted_total > 0:
            self.overall_precision = 100 * self.gold_total / self.predicted_total
        if self.correct_total > 0:
            self.overall_recall = 100 * self.gold_total / self.correct_total
        if self.overall_precision + self.overall_recall > 0:
            self.overall_fb1 = 2 * self.overall_precision * self.overall_recall / (self.overall_precision + self.overall_recall)

    def to_conll_output(self):
        output = []
        # print overall performance
        processed = "processed %d tokens with %d phrases; " % (self.n_tokens, self.correct_total)
        output.append("%sfound: %d phrases; correct: %d." % (processed, self.predicted_total, self.gold_total))
        if self.n_tokens > 0:
            acc_str = "accuracy: {:>6.2f}%; ".format(self.accuracy)
            prec_str = "precision: {:>6.2f}%; ".format(self.overall_precision)
            rec_str = "recall: {:>6.2f}%; ".format(self.overall_recall)
            fb1_str = "FB1: {:>6.2f}".format(self.overall_fb1)
            output.append("%s%s%s%s" % (acc_str, prec_str, rec_str, fb1_str))

        for i in self.sorted_types:
            precision = self.precision[i]
            recall = self.recall[i]
            fb1 = self.fb1[i]
            output.append("{:>17}: precision: {:>6.2f}%; recall: {:>6.2f}%; FB1: {:>6.2f}  {}".format(
                i, precision, recall, fb1, self.predicted_count[i]))
        return '\n'.join(output)

    def to_latex_output(self):
        output = ["        & Precision &  Recall  & F$_{\\beta=1} \\\\\\hline"]
        for index, i in enumerate(self.sorted_types):
            precision = self.precision[i]
            recall = self.recall[i]
            fb1 = self.fb1[i]
            output.append("{:7} &  {:>6.2f}\\% & {:>6.2f}\\% & {:>6.2f} \\\\{}"
                          .format(i, precision, recall, fb1, '\\hline' if index == len(self.sorted_types) - 1 else ''))
        precision = self.overall_precision
        recall = self.overall_recall
        fb1 = self.overall_fb1
        output.append("Overall &  {:>6.2f}\\% & {:>6.2f}\\% & {:>6.2f} \\\\\\hline".format(precision, recall, fb1))
        return '\n'.join(output)


def conll_eval_lines(lines, delimiter=' ', raw=False, out_tag='O', boundary='-X-'):
    """
    Perform CoNLL 2000/2003 evaluation script on input iterable lines corresponding to the standard input.
    :param lines: lines with token items separated by delimiter character and blank lines separating sentences
    :param delimiter: delimiter separating features for each token in a line
    :param raw: whether or not to accept non IOB inputs
    :param out_tag: tag which indicates a token is not part of a chunk
    :param boundary: boundary word in first column indicating an end of a sequence/instance
    :return: evaluation with precision, recall, and F1 for each type
    """
    correct_tags = 0  # number of correct chunk tags
    in_correct = False  # currently processed chunk is correct until now
    last_correct = "O"  # previous chunk tag in corpus
    last_correct_type = ""  # type of previously identified chunk tag
    last_guessed = "O"  # previously identified chunk tag
    last_guessed_type = ""  # type of previous chunk tag in corpus
    nbr_of_features = -1  # number of features per line
    token_counter = 0  # token counter (ignores sentence breaks)

    correct_chunk = Counter()
    found_correct = Counter()
    found_guessed = Counter()

    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        features = list(line.split(delimiter))
        if nbr_of_features < 0:
            nbr_of_features = len(features)
        elif nbr_of_features != len(features) and features:
            raise AssertionError('Unexpected number of features: %d (%d)' % (len(features), nbr_of_features))
        if len(features) == 0 or features[0] == boundary:
            features = [boundary, "O", "O"]
        elif len(features) < 2:
            raise AssertionError("conlleval: unexpected number of features in line %s" % line)
        if raw:
            if features[-1] == out_tag:
                features[-1] = "O"
            if features[-2] == out_tag:
                features[-2] = "O"
            if features[-1] != "O" and (len(features[-1]) < 2 or (features[-1][:2] not in BIES)):
                features[-1] = "B-%s" % features[-1]
            if features[-2] != "O" and (len(features[-2]) < 2 or (features[-2][:2] not in BIES)):
                features[-2] = "B-%s" % features[-2]
        # 20040126 ET code which allows hyphens in the types
        match = re.match("^([^-]*)-(.*)$", features[-1])
        if match:
            guessed = match.group(1)  # current guessed chunk tag
            guessed_type = match.group(2)  # type of current guessed chunk tag
        else:
            guessed = features[-1]
            guessed_type = ""
        match = re.match("^([^-]*)-(.*)$", features[-2])
        if match:
            correct = match.group(1)  # current corpus chunk tag (I,O,B)
            correct_type = match.group(2)  # type of current corpus chunk tag (NP,VP,etc.)
        else:
            correct = features[-2]
            correct_type = ""
        guessed_type = guessed_type if guessed_type else ""
        correct_type = correct_type if correct_type else ""

        first_item = features[0]  # first feature (for sentence boundary checks)

        # 1999-06-26 sentence breaks should always be counted as out of chunk
        if features == boundary:
            guessed = "O"

        if in_correct:
            if (end_of_chunk(last_correct, correct, last_correct_type, correct_type)
                    and end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
                    and last_guessed_type == last_correct_type):
                in_correct = False
                correct_chunk[last_correct_type] += 1
            elif (end_of_chunk(last_correct, correct, last_correct_type, correct_type)
                  != end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
                  or guessed_type != correct_type):
                in_correct = False

        if (start_of_chunk(last_correct, correct, last_correct_type, correct_type)
                and start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
                and guessed_type == correct_type):
            in_correct = True

        if start_of_chunk(last_correct, correct, last_correct_type, correct_type):
            found_correct[correct_type] += 1
        if start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type):
            found_guessed[guessed_type] += 1
        if first_item != boundary:
            if correct == guessed and guessed_type == correct_type:
                correct_tags += 1
            token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        correct_chunk[last_correct_type] += 1

    accuracy = 100 * correct_tags / token_counter
    eval_result = ConllEvaluation(correct_chunk, found_correct, found_guessed, token_counter, accuracy)

    return eval_result


def end_of_chunk(prev_tag, tag, prev_type, curr_type):
    """
    Checks if a chunk ended between the previous and current word.

    Note: this code is capable of handling other chunk representations than the default CoNLL-2000 ones, see EACL'99 paper of
    Tjong Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
    :param prev_tag: previous chunk tag
    :param tag: current chunk tag
    :param prev_type: previous type
    :param curr_type: current type
    :return: `True` if arguments indicate the end of a chunk
    """
    chunk_end = False
    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag == "E" and tag == "E":
        chunk_end = True
    if prev_tag == "E" and tag == "I":
        chunk_end = True
    if prev_tag == "E" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag != "O" and prev_tag != "." and prev_type != curr_type:
        chunk_end = True

    # corrected 1998-12-22: these chunks are assumed to have length 1
    if prev_tag == "]":
        chunk_end = True
    if prev_tag == "[":
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, curr_type):
    """
    Checks if a chunk started between the previous and current word.

    Note: This code is capable of handling other chunk representations than the default CoNLL-2000 ones, see EACL'99 paper of
    Tjong Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006

    :param prev_tag: previous chunk tag
    :param tag: current chunk tag
    :param prev_type: previous type
    :param curr_type: current type
    :return: `True` if arguments indicate the start of a chunk
    """
    chunk_start = False
    if prev_tag == "B" and tag == "B":
        chunk_start = True
    if prev_tag == "I" and tag == "B":
        chunk_start = True
    if prev_tag == "O" and tag == "B":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if prev_tag == "E" and tag == "E":
        chunk_start = True
    if prev_tag == "E" and tag == "I":
        chunk_start = True
    if prev_tag == "O" and tag == "E":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if tag != "O" and tag != "." and prev_type != curr_type:
        chunk_start = True

    # corrected 1998-12-22: these chunks are assumed to have length 1
    if tag == "[":
        chunk_start = True
    if tag == "]":
        chunk_start = True

    return chunk_start


def options(args=None):
    parser = argparse.ArgumentParser(description="evaluate result of processing CoNLL-2000 shared task")
    parser.add_argument('--input', type=str, required=True, help='Path to input file with predictions and gold tags')
    parser.add_argument('-d', type=str, default=' ', help='alternative delimiter tag (default is single space)')
    parser.add_argument('-o', type=str, default=' ', help='alternative outside tag (default is O)')
    parser.add_argument('-l', dest='latex', action='store_true', help='generate LaTeX output for tables like '
                                                                      'in http://cnts.uia.ac.be/conll2003/ner/example.tex')
    parser.add_argument('-r', dest='raw', action='store_true',
                        help='accept raw result tags (without B- and I- prefix; assumes one word per chunk')
    parser.set_defaults(raw=False)
    parser.set_defaults(latex=False)
    return parser.parse_args(args)


def main():
    _opts = options()
    with open(_opts.input) as _lines:
        res = conll_eval_lines(_lines, delimiter=_opts.d, raw=_opts.raw, out_tag=_opts.o)
        if _opts.latex:
            print(res.to_latex_output())
        else:
            print(res.to_conll_output())


if __name__ == '__main__':
    main()
