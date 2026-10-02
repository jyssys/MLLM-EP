"""One versioned bounded GPU session with durable stdout/stderr log."""
import argparse
import os
import subprocess
from poc_speculative_moe_ep.contract import ROOT,PYTHON,MASK


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--timeout',type=int,default=1200)
    a,run_args=p.parse_known_args()
    output=ROOT/'raw'/a.name
    logpath=ROOT/'raw'/(a.name+'.log')
    if output.exists() or logpath.exists():raise FileExistsError('append-only run ID required')
    logpath.parent.mkdir(parents=True,exist_ok=True)
    control=ROOT/'raw'/(a.name+'.control.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK,SD_EP_OUTPUT=str(output),SD_EP_CONTROL=str(control),
             PYTHONPATH=str(ROOT/'hooks')+':'+str(ROOT.parent),VLLM_NO_USAGE_STATS='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    command=[PYTHON,'-m','poc_speculative_moe_ep.scripts.gpu_session','--timeout',str(a.timeout),
             PYTHON,'-m','poc_speculative_moe_ep.scripts.run',
             '--output',str(output),'--control',str(control),*run_args]
    with logpath.open('x') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent)
        print(f'LAUNCHED pid={proc.pid} log={logpath}',flush=True)
        code=proc.wait()
    print(f'COMPLETED exit={code}',flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
