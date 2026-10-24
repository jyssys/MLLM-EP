"""Preserve failed attempts and register distinct reviewed retry IDs."""
import json
import argparse
from .common import ROOT,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='dp');p.add_argument('--suffix',default='_r1')
    a=p.parse_args()
    plan=json.loads((ROOT/'plans'/f'{a.plan}.json').read_text())
    blocks=plan.get('groups',[dict(jobs=plan.get('jobs',[]))])
    for group in blocks:
        if group.get('barrier'):group['barrier']+=a.suffix
        for job in group['jobs']:
            job['id']+=a.suffix
            if job.get('barrier'):job['barrier']+=a.suffix
            job['repair']='Common prompt group across selected heterogeneous remainder rows. CPU regression passes.'
    name='dp_retry' if a.plan=='dp' and a.suffix=='_r1' else a.plan+a.suffix
    path=ROOT/'plans'/f'{name}.json'
    assert not path.exists()
    write(path,plan)

if __name__=='__main__':main()
