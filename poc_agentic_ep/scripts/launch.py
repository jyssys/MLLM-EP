"""Existing bounded burner/experiment transaction, explicit authorized mask."""
import argparse
import os
import subprocess
from poc_agentic_ep.contract import ROOT,PYTHON,MASK


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--timeout',type=int,default=3600)
    a,remaining=p.parse_known_args();out=ROOT/'raw'/a.name
    log=ROOT/'raw'/(a.name+'.log');control=ROOT/'raw'/(a.name+'.control.json')
    if out.exists() or log.exists():raise FileExistsError('append-only name required')
    log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK,AGENT_EP_OUTPUT=str(out),AGENT_EP_CONTROL=str(control),
        SD_EP_OUTPUT=str(out),NATIVE_LP_OUTPUT=str(out),PYTHONPATH=str(ROOT/'hooks')+':'+str(ROOT.parent),
        VLLM_NO_USAGE_STATS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    cmd=[PYTHON,'-m','poc_speculative_moe_ep.scripts.gpu_session','--timeout',str(a.timeout),
        PYTHON,'-m','poc_agentic_ep.scripts.replay_ep4','--output',str(out),'--control',str(control),*remaining]
    with log.open('x') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT.parent)
        print(f'LAUNCHED pid={proc.pid} log={log}',flush=True);code=proc.wait()
    print(f'COMPLETE exit={code}',flush=True);raise SystemExit(code)


if __name__=='__main__':main()
