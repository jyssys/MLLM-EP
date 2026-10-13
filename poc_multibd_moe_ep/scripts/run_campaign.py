"""Fixed experiment manifest; no adaptive scheduling or request selection."""
import argparse
import os
import subprocess
import sys
from poc_multibd_moe_ep.contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibration','held_out','tp_control'],required=True)
    a=p.parse_args()
    if a.phase=='calibration':
        cases=[(r,mode,'calibration',8,1 if mode=='trace' else 3,1 if mode=='trace' else 3,'ep4')
               for r in [1,4,2,3] for mode in ['clean','timing','trace']]
    elif a.phase=='held_out':
        cases=[(r,mode,'held_out',24,1 if mode=='trace' else 3,1 if mode=='trace' else 3,'ep4')
               for r in [3,2,4,1] for mode in ['clean','timing','trace']]
    else:
        cases=[(r,'clean','held_out',24,3,3,'tp4') for r in [1,4,2,3]]
    write(ROOT/f'artifacts/{a.phase}_campaign_plan.json',dict(phase=a.phase,cases=cases,
        checkpoint=MODEL_ID,revision=MODEL_REVISION,selection_frozen=True,
        max_tokens=4096,max_nfe=1024,temperature=0,no_DMax=True,
        headline='clean lane, final useful output tokens / complete generation wall',
        raw_runtime_acceptances='reported separately; includes positions removed at EOS',
        full_request_TPF='returned committed output tokens / total model forwards',
        PCE='TPS gain / useful TPF gain',same_request_warmups_retained=True))
    for r,mode,split,limit,repeats,warmup,backend in cases:
        target=ROOT/f'artifacts/R{r}_{backend}_{split}_{mode}'
        command=[PYTHON,'-m','poc_multibd_moe_ep.scripts.run_sweep','--R',str(r),
            '--output',str(target),'--mode',mode,'--backend',backend,'--split',split,
            '--limit',str(limit),'--repeats',str(repeats),'--warmup',str(warmup),'--max-tokens','4096']
        print('START',target.name,flush=True)
        target.mkdir(parents=True,exist_ok=True)
        with (target/'console.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ)
            try:code=proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.terminate()
                raise
        write(target/'process_status.json',dict(exit_code=code))
        print('FINISH',target.name,code,flush=True)
        if code:raise RuntimeError('Bounded campaign stopped at failure: '+target.name)


if __name__=='__main__':main()
