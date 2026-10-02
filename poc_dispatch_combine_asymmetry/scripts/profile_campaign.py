"""Small, separate Nsight lane; no profiler timing becomes a headline."""
import os
import subprocess
import sys
from pathlib import Path

from poc_dispatch_combine_asymmetry.contract import ROOT, write


def main():
    root=ROOT/'artifacts/nsight';root.mkdir(parents=True,exist_ok=True)
    ledger=[]
    for model,arms in [('mllm','visual,text'),('llm','text')]:
        out=ROOT/'artifacts/runs'/f'{model}_nsight_v4'
        env=os.environ.copy();env.update(ASYMMETRY_OUTPUT=str(out),ASYMMETRY_NSYS='1')
        cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
             '--capture-range=cudaProfilerApi','--capture-range-end=none',
             '--output',str(root/f'{model}_native_v4'),sys.executable,'-m',
             'poc_dispatch_combine_asymmetry.scripts.run','--model',model,'--arms',arms,
             '--sizes','high','--repeats','1','--lane','clean','--output',str(out),
             '--control',env['ASYMMETRY_CONTROL']]
        with (root/f'{model}_profile.log').open('w') as log:
            code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
        ledger.append({'model':model,'command':cmd,'returncode':code})
        write(root/'ledger.json',ledger)
        if code:raise RuntimeError(f'profiler unavailable {model}, no retry')
    write(Path(os.environ['NATIVE_LP_OUTPUT'])/'status.json',{'ok':True})


if __name__=='__main__':main()
