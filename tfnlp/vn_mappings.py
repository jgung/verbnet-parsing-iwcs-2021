import re
from collections import defaultdict, Counter

from tfnlp.common.utils import write_json


def generate_pbvn_sense_mappings():
    file = 'pbvn_mappings.tsv'

    vn_mappings = defaultdict(list)
    with open(file, mode='r') as mappings:
        for line in mappings:
            fields = line.split('\t')
            vncls = fields[1].strip()
            pbrs = fields[0].strip()
            mappings = vn_mappings[pbrs]
            if vncls not in mappings:
                mappings.append(vncls)

    write_json(vn_mappings, 'pbvn_mappings.json')


def generate_rs_mappings():
    file = 'pbvn_mappings.tsv'

    vn_mappings = {}
    with open(file, mode='r') as mappings:
        for line in mappings:
            fields = line.split('\t')
            vncls = fields[0]
            pbrs = fields[1]
            pb = fields[2]
            vn = fields[3]
            nm = ['NONE', 'NM']
            if pb in nm or vn in nm:
                continue
            vncls = re.sub('-.*$', '', vncls)

            role_mappings = vn_mappings.get(vncls, {})
            vn_mappings[vncls] = role_mappings

            vn_roles = role_mappings.get(pb, Counter())
            role_mappings[pb] = vn_roles
            vn_roles[vn] += 1

    write_json(vn_mappings, 'pbvn_mappings.json')


def add_rs():
    with open('data/datasets/thesis/semlink/pb-dev.txt', mode='r') as rs, \
            open('data/datasets/thesis/semlink/dev.txt', mode='r') as lines, \
            open('data/datasets/thesis/semlink/dev.rs.txt', mode='w') as w:
        for sense, line in zip(rs, lines):
            line = line.strip()
            fields = line.split()
            if len(fields) > 4:
                fields[4] = sense.strip()
            w.write(' '.join(fields) + '\n')


if __name__ == '__main__':
    generate_pbvn_sense_mappings()
