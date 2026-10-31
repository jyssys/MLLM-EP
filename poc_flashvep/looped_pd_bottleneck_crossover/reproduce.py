"""Clone one frozen configuration into an unused namespace."""
import argparse
import subprocess
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('source_config');p.add_argument('--name',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    assert a.name and all(ch.isalnum() or ch=='_' for ch in a.name),'Use a snake_case run name'
    source=ROOT/'configs'/a.source_config;cfg=json.loads(source.read_text());cfg['name']=a.name
    assert not (ROOT/'raw'/a.name).exists()
    path=ROOT/'configs'/(a.name+'.json');assert not path.exists();write(path,cfg)
    command=[str(PYTHON),'-m','poc_flashvep.looped_pd_bottleneck_crossover.dispatch',path.name]
    print(' '.join(command),flush=True)
    if a.execute:subprocess.run(command,cwd=REPO,check=True)

if __name__=='__main__':main()
