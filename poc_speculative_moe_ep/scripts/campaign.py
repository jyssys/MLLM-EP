"""Resume completed versioned arms; bounded sequential GPU transactions."""
import argparse
import json
import subprocess
from poc_speculative_moe_ep.contract import ROOT,PYTHON


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prefix',required=True)
    p.add_argument('--gammas',default='0,2,4,8')
    p.add_argument('--lane',choices=['trace','clean','bypass'],default='trace')
    p.add_argument('--requests',type=int,default=16)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args()
    for g in map(int,a.gammas.split(',')):
        assert g in (0,2,4,8)
        name=f'{a.prefix}_g{g}'
        status=ROOT/'raw'/name/'status.json'
        if status.exists():
            if json.loads(status.read_text())['ok']:
                print('RESUME_SKIP '+name,flush=True);continue
            raise RuntimeError('failed arm preserved; choose a new versioned prefix')
        command=[PYTHON,'-m','poc_speculative_moe_ep.scripts.launch','--name',name,'--timeout',str(a.timeout),
                 '--gamma',str(g),'--lane',a.lane,'--requests',str(a.requests),'--start',str(a.start),
                 '--repeats',str(a.repeats),'--tokens',str(a.tokens),'--warmup','3']
        subprocess.run(command,check=True,cwd=ROOT.parent)


if __name__=='__main__':main()
