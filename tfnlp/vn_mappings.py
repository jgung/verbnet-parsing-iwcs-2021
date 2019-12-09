import re
from collections import Counter

from common.utils import write_json

if __name__ == '__main__':
    file = 'pbvn_mappings.tsv'

    vn_mappings = {}
    with open(file, mode='r') as mappings:
        for line in mappings:
            fields = line.split('\t')
            vncls = fields[0]
            pbrs = fields[1]
            pb = fields[2]
            vn = fields[3]
            NM = ['NONE', 'NM']
            if pb in NM or vn in NM:
                continue
            vncls = re.sub('-.*$', '', vncls)

            role_mappings = vn_mappings.get(vncls, {})
            vn_mappings[vncls] = role_mappings

            vn_roles = role_mappings.get(pb, Counter())
            role_mappings[pb] = vn_roles
            vn_roles[vn] += 1
    write_json(vn_mappings, 'pbvn_mappings.json')
