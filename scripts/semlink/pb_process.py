import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from xml.dom import minidom

import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENTENCE_FIELD = 1
TOKEN_FIELD = 2
ROLE_START = 6

PB_CORE_ARGS = "ARG([^M])"  # filter out props with core PB args, missing args, or typos

SEMLINK_PB_TO_VN = "^(\S+?)\.([^;]+);VN=(\S+)"  # extract VN sense and lemma from sense, e.g. "say.01;VN=37.7"
SEMLINK_TO_VN = r"\1.\3"  # replace original sense with VerbNet sense, like "say.37.7"
SEMLINK_TO_PB = r"\1.\2"  # replace original sense with PropBank sense, like "say.01"

ROLE = "^(\S+:\d+)-(\S+)$"  # label from full role annotation, e.g. 0:2*15:0-ARG1-to --> ARG1-to
CLEAN_ROLE = "^(ARGM-[a-zA-Z0-9]+|([a-zA-Z0-9]+)(?:\[([^\]]+)\])?).*$"  # e.g. ARG1-to --> ARG1, ARG1[Theme]-from --> ARG1[Theme]
BRACKET_SPACE = "\[\s+\]|(?<=\[)\s+|\s+(?=\])"  # clean up extra whitespace within bracketed mapped VN roles

CLEAN_MAPPING_ERRORS = " -by"  # 16:1-Agent -by --> 16:1-Agent-by
COMBINED_ROLES_FORMAT = '%s$%s'  # ARG1$Theme


def write_counts(outpath, counts):
    total = sum(counts.values())
    sorted_counts = sorted([(k, v) for k, v in counts.items()], key=lambda x: x[1], reverse=True)
    with open(outpath, mode='wt') as out:
        out.write(str(total) + '\n')
        for k, v in sorted_counts:
            out.write('{}\t{}\n'.format(k, v))


class PropStats(object):
    def __init__(self, props):
        super(PropStats, self).__init__()
        self.props = props
        self.lemmas = Counter()
        self.predicates = Counter()
        self.roles = Counter()
        self.senses = Counter()
        self._count()

    def _count(self):
        for prop in self.props:
            predicate = prop.predicate
            separator_index = predicate.index(".")
            lemma, sense = predicate[:separator_index], predicate[separator_index + 1:]
            self.lemmas[lemma] += 1
            self.predicates[predicate] += 1
            self.senses[sense] += 1
            for role in prop.roles:
                self.roles[role.label] += 1

    def write_all(self, outpath):
        write_counts(outpath + '.lemmas.txt', self.lemmas)
        write_counts(outpath + '.preds.txt', self.predicates)
        write_counts(outpath + '.roles.txt', self.roles)
        write_counts(outpath + '.senses.txt', self.senses)


class Proposition(object):
    def __init__(self, proppath, sentence, token, annotator, predicate, atts, roles):
        super(Proposition, self).__init__()
        self.proppath = proppath
        self.sentence = int(sentence)
        self.token = int(token)
        self.annotator = annotator
        self.predicate = predicate
        self.atts = atts
        self.roles = roles
        self.cols = [proppath, self.sentence, self.token, annotator, predicate]


class Role(object):
    def __init__(self, pointer, label):
        super(Role, self).__init__()
        self.pointer = pointer
        self.label = label


class VnMapping(object):
    def __init__(self, lemma, roleset, vncls, rolemap):
        super(VnMapping, self).__init__()
        self.lemma = lemma
        self.roleset = roleset
        self.vncls = vncls
        self.rolemap = rolemap


def read_role_mappings_json(mappings_json):
    with open(mappings_json) as mappings:
        return json.loads(mappings.read())


def read_mappings_xml(mappings_xml):
    """
    Generate a mapping dictionary from rolesets to lists of VnMapping objects.
    :param mappings_xml: pb-vn mappings XML file
    :return: mapping dict
    """
    lemma_map = {}
    roleset_map = defaultdict(list)
    for predicate in minidom.parse(mappings_xml).getElementsByTagName("predicate"):
        lemma = predicate.attributes['lemma'].value
        if lemma in lemma_map:
            raise ValueError('Repeat lemma found in mappings file {}: {}'.format(mappings_xml, lemma))
        roles = defaultdict(list)
        lemma_map[lemma] = roles

        for argmap in predicate.getElementsByTagName("argmap"):
            roleset = argmap.attributes['pb-roleset'].value
            vncls = argmap.attributes['vn-class'].value
            rolemap = {}
            mapping = VnMapping(lemma, roleset, vncls, rolemap)

            roles[roleset].append(mapping)
            roleset_map[roleset].append(mapping)

            for role in argmap.getElementsByTagName("role"):
                pbarg = role.attributes['pb-arg'].value
                vntheta = role.attributes['vn-theta'].value
                if not vntheta.strip():
                    logger.warning("Empty SemLink pb-arg to vn-arg mapping for %s ARG%s" % (roleset, pbarg))
                    continue
                if pbarg in rolemap:
                    raise ValueError('Non deterministic mapping for {} arg {}'.format(lemma, pbarg))
                rolemap[pbarg] = vntheta
    return roleset_map


def filter_props(props, filter_pattern):
    pattern = re.compile(filter_pattern)
    result = []
    for prop in props:
        if bool(pattern.search(prop)):
            continue
        result.append(prop)
    return result


def map_props(props, search_pattern, search_repl_pattern):
    result = []
    for prop in props:
        result.append(re.sub(search_pattern, search_repl_pattern, prop))
    return result


def map_roles_semlink(props, roleset_mappings, filter_incomplete=True, use_vnrole=True, use_vnpb=False, use_vncls=True):
    result = []
    for prop in props:
        vnmappings = roleset_mappings.get(prop.predicate)
        if filter_incomplete and not vnmappings:
            continue

        if len(vnmappings) != 1:
            logger.warning("Non-deterministic mapping for %s" % prop.predicate)
            continue
        vnmapping = vnmappings[0]

        complete = True
        for role in prop.roles:
            role_mappings = vnmapping.rolemap

            match = re.search(PB_CORE_ARGS, role.label)
            if match and match.group(1):
                mapped_role = role_mappings.get(match.group(1))
                if not mapped_role:
                    logger.warning("No mapping for %s %s" % (prop.predicate, role.label))
                    if filter_incomplete:
                        complete = False
                        break
                if use_vnpb:
                    role.label = COMBINED_ROLES_FORMAT % (match.group(0), mapped_role.capitalize())
                elif use_vnrole:
                    role.label = mapped_role.capitalize()
        if complete:
            if use_vncls:
                prop.predicate = vnmapping.lemma + '.' + vnmapping.vncls
            result.append(prop)

    return result


def read_props(propspath):
    result = []
    skipped = []
    with open(propspath) as f:
        for prop in f:
            prop = _preprocess_prop(prop)
            if not prop:
                continue
            fields = prop.split()
            if not len(fields) > 6:
                raise ValueError('Unexpected number of fields (%d vs. 6): %s' % (len(fields), f))

            roles = []
            for rolestr in fields[ROLE_START:]:
                role = _read_role(rolestr)
                if role:
                    roles.append(role)
            if len(roles) != len(fields[ROLE_START:]):
                skipped.append(prop)
                continue

            result.append(Proposition(*fields[:ROLE_START], roles))
    if skipped:
        if logger.level == logging.DEBUG:
            logger.debug("Skipping %d props:\n\n%s\n" % (len(skipped), '\n'.join(skipped)))
        logger.warning('Skipping %d props due to invalid role formats' % len(skipped))

    return result


def _preprocess_prop(prop):
    prop = prop.strip()
    prop = re.sub(CLEAN_MAPPING_ERRORS, "", prop)
    prop = re.sub(BRACKET_SPACE, "", prop)  # remove whitespace that shouldn't be there, e.g. 5:1-ARG1[ ] --> ARG1[]
    return prop


def _read_role(rolestr):
    role_search = re.search(ROLE, rolestr, re.IGNORECASE)
    if not role_search:
        logger.warning('Unexpected role format: %s' % rolestr)
        return None
    return Role(role_search.group(1), role_search.group(2))


def sort_props(props, sort_cols):
    if not sort_cols:
        return props

    def sort_tuple(prop):
        return tuple(prop.cols[col] if col == SENTENCE_FIELD or col == TOKEN_FIELD else prop.cols[col] for col in sort_cols)

    return sorted(props, key=sort_tuple)


def filter_role_labels(props, filter_fn):
    result = []
    for prop in props:
        if filter_fn(prop.roles):
            continue
        result.append(prop)
    return result


def process_roles(props, process_fn):
    result = []
    skipped = []
    for prop in props:
        if not process_fn(prop.roles):
            skipped.append(prop)
            continue
        result.append(prop)
    if skipped:
        if logger.level == logging.DEBUG:
            logger.debug("Filtering out %d props:\n\n%s\n" % (len(skipped), '\n'.join(format_props(skipped))))
        logger.warning('Filtered out %d props' % len(skipped))
    return result


def process_predicates(props, process_fn):
    result = []
    skipped = []
    for prop in props:
        prop.predicate = process_fn(prop.predicate)
        if not prop.predicate:
            skipped.append(prop)
            continue
        result.append(prop)
    if skipped:
        if logger.level == logging.DEBUG:
            logger.debug("Filtering out %d props:\n\n%s\n" % (len(skipped), '\n'.join(format_props(skipped))))
        logger.warning('Filtered out %d props' % len(skipped))
    return result


def map_predicate(predicate, search, replace):
    if search and replace:
        return re.sub(search, replace, predicate)
    return predicate


def clean_role_labels(roles):
    for role in roles:
        match = re.search(CLEAN_ROLE, role.label)
        cleaned = match.group(1)
        cleaned = cleaned.strip()
        if not cleaned:
            logger.warning('Unexpected role format: %s' % role.label)
            return False
        role.label = cleaned
    return True


def map_roles(roles, mappings):
    if mappings:
        for role in roles:
            role.label = mappings.get(role.label, role.label)
    return True


def fix_semlink_errors(roles):
    for role in roles:
        label = role.label
        label = label.replace("announcement", "Topic")  # erroneous mapping in Semlink 1.0/1.1
        if label.startswith("ARGM-"):
            if label == "ARGM-TM":
                label = "ARGM-TMP"
            if len(label) != 8:
                logger.warning('Unexpected ARGM-* role: %s' % label)
                return False
        role.label = label
    return True


def extract_semlink_roles(roles, vn, vnpb, filter_incomplete):
    all_mapped = True
    for role in roles:
        match = re.search(CLEAN_ROLE, role.label)
        if not match.group(3) and bool(re.search(PB_CORE_ARGS, role.label)):
            all_mapped = False
        if match.group(3):
            if vnpb:
                role.label = COMBINED_ROLES_FORMAT % (match.group(2), match.group(3))
            elif vn:
                role.label = match.group(3)  # e.g. "ARG1[Theme]" --> "Theme"
            else:
                role.label = match.group(2)  # e.g. "ARG1[Theme]" --> "ARG1"
    return not filter_incomplete or all_mapped


def format_props(props):
    if not props:
        return []
    path_len = len(max(props, key=lambda p: len(p.proppath)).proppath)
    anns_len = len(max(props, key=lambda p: len(p.annotator)).annotator)
    pred_len = len(max(props, key=lambda p: len(p.predicate)).predicate)
    atts_len = len(max(props, key=lambda p: len(p.atts)).atts)

    result = []
    for prop in props:
        roles = ' '.join(['{}-{}'.format(role.pointer, role.label) for role in prop.roles])
        result.append('{:{path_len}} {:4d} {:4d} {:{ann_len}} {:{pred_len}} {:{att_len}} {}'.format(
            prop.proppath, prop.sentence, prop.token, prop.annotator, prop.predicate, prop.atts, roles,
            path_len=path_len, ann_len=anns_len, pred_len=pred_len, att_len=atts_len))
    return result


def transform_props(propspath, outpath, search_pattern=None, search_repl_pattern=None, sort_cols=None, vn_roles=False,
                    vnpb_roles=False, vncls=False, filter_incomplete=False, mappings=None, semlink_mappings=None):
    props = read_props(propspath)
    logger.info('Read %d props' % len(props))

    props = sort_props(props, sort_cols)

    props = process_roles(props, clean_role_labels)
    if not semlink_mappings:
        props = process_roles(props, lambda roles: extract_semlink_roles(roles, vn_roles, vnpb_roles, filter_incomplete))
    props = process_roles(props, fix_semlink_errors)
    props = process_roles(props, lambda roles: map_roles(roles, mappings))

    props = process_predicates(props, lambda pred: map_predicate(pred, search_pattern, search_repl_pattern))

    if semlink_mappings:
        props = map_roles_semlink(props, semlink_mappings, filter_incomplete=filter_incomplete, use_vnrole=vn_roles,
                                  use_vnpb=vnpb_roles, use_vncls=vncls)

    props = sort_props(props, sort_cols)

    PropStats(props).write_all(outpath)
    with open(outpath, mode='wt') as out:
        formatted = format_props(props)
        for prop in formatted:
            out.write(prop + '\n')

    logger.info('Wrote %d props to %s' % (len(props), outpath))
    return props


def options():
    parser = argparse.ArgumentParser(description="Utility for sorting, filtering, and transforming PropBank pointers.")
    parser.add_argument('--pb', type=str, required=True, help='PropBank pointers file')
    parser.add_argument('--o', type=str, help='(optional) output path')
    parser.add_argument('--search', type=str, help='(optional) pointer regex search for mapping')
    parser.add_argument('--replace', type=str, help='(optional) pointer replacement regex for mapping')
    parser.add_argument('--sort-columns', default="0,1,2", dest='sort_cols', type=str,
                        help='(optional) comma-separated column indices in props to sort by (e.g. "4,0,1" to sort first by '
                             'sense, then by path and sentence)')
    parser.add_argument('--semlink', action='store_true',
                        help='use default settings for processing SemLink data (remove pointers with PropBank core arguments, '
                             'change senses to VN)')
    parser.add_argument('--filter-incomplete', dest='filter_incomplete',
                        action='store_true', help='filter any incompletely mapped propositions in SemLink')
    parser.add_argument('--mappings', type=str, help='(optional) thematic role mappings JSON')
    parser.add_argument('--semlink-mappings', dest='semlink_mappings', type=str, help='(optional) SemLink PB VN mappings XML')
    parser.add_argument('--vn', action='store_true', help='Extract VN roles instead of PB roles')
    parser.add_argument('--vnpb', '--pbvn', dest='vnpb', action='store_true', help='Extract both PB and VN roles')
    parser.add_argument('--vncls', action='store_true', help='When "--semlink-mappings" is provided, use VN classes instead of '
                                                             'PB rolesets')
    parser.add_argument('--level', default='INFO', help='logging level (e.g. INFO, DEBUG, etc.)')
    parser.set_defaults(semlink=False)
    parser.set_defaults(vn=False)
    parser.set_defaults(vnpb=False)
    parser.set_defaults(vncls=False)
    parser.set_defaults(filter_incomplete=False)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def main():
    _opts = options()

    logger.level = getattr(logging, _opts.level.upper(), 10)

    search_pattern = _opts.search
    replace_pattern = _opts.replace
    if _opts.semlink:
        if not search_pattern:
            search_pattern = SEMLINK_PB_TO_VN
        if not replace_pattern:
            if _opts.vncls:
                replace_pattern = SEMLINK_TO_VN
            else:
                replace_pattern = SEMLINK_TO_PB

    if _opts.mappings:
        _opts.mappings = read_role_mappings_json(_opts.mappings)

    sort_cols = [int(col) for col in _opts.sort_cols.split(',')] if _opts.sort_cols else []
    if not _opts.o:
        _opts.o = _opts.pb + '.out'

    semlink_mappings = None
    if _opts.semlink_mappings:
        semlink_mappings = read_mappings_xml(_opts.semlink_mappings)

    transform_props(_opts.pb, outpath=_opts.o, search_pattern=search_pattern,
                    search_repl_pattern=replace_pattern, sort_cols=sort_cols, vn_roles=_opts.vn, vnpb_roles=_opts.vnpb,
                    vncls=_opts.vncls, filter_incomplete=_opts.filter_incomplete,
                    mappings=_opts.mappings, semlink_mappings=semlink_mappings)


if __name__ == '__main__':
    main()
