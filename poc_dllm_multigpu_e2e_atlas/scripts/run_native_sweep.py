"""Bounded primary context sweep. Each process owns one native graph set."""
import argparse
import os
import subprocess
from poc_dllm_multigpu_e2e_atlas.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefixes',type=int,nargs='+',default=PREFIXES)
    args=parser.parse_args();ledger=[]
    for prefix in args.prefixes:
        assert prefix in PREFIXES
        for fallback in [False,True] if prefix==16384 else [False]:
            name=f'results/raw/native_p{prefix}_'+('eager_fallback_v1' if fallback else 'v1')
            path=ROOT/name
            command=[PYTHON,'-m','poc_dllm_multigpu_e2e_atlas.scripts.run_guarded','--output',name,'--timeout','900','--',
                PYTHON,'-m','poc_dllm_multigpu_e2e_atlas.scripts.run_clean_baseline','--output',str(path),
                '--prefix',str(prefix),'--limit','8','--repeats','3']
            if fallback:command.append('--eager-prefill')
            result=subprocess.run(command,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK))
            ledger.append(dict(prefix=prefix,eager_prefill=fallback,code=result.returncode,output=name))
            write(ROOT/'results/native_campaign.json',ledger)
            if result.returncode==0:break
            if prefix!=16384 or fallback:raise SystemExit(result.returncode)


if __name__=='__main__':main()
