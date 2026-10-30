"""Create immutable run configurations from a previous configuration."""
import argparse
import json
from .common import ROOT,write

def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('name');p.add_argument('overrides',nargs='?',default='{}');a=p.parse_args()
    cfg=json.loads((ROOT/'configs'/a.source).read_text());cfg.update(json.loads(a.overrides));cfg['name']=a.name
    dest=ROOT/'configs'/f'{a.name}.json'
    if dest.exists():raise FileExistsError(dest)
    write(dest,cfg)

if __name__=='__main__':main()
