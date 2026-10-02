"""Finite continuation: native LLM control then separate internal/overhead lane."""
import json
import os
import subprocess
import sys
from pathlib import Path

from poc_dispatch_combine_asymmetry.contract import ROOT, write


def main():
    jobs=[('llm_controls_v3','llm','text','clean',10,3,5),
          ('mllm_internal_v3','mllm','visual,text','internal',3,0,5)]
    ledger=[]
    for tag,model,arms,lane,repeats,internal,overhead in jobs:
        out=ROOT/'artifacts/runs'/tag
        env=os.environ.copy();env['ASYMMETRY_OUTPUT']=str(out)
        cmd=[sys.executable,'-m','poc_dispatch_combine_asymmetry.scripts.run',
             '--model',model,'--arms',arms,'--sizes','high,low,medium','--lane',lane,
             '--repeats',str(repeats),'--append-internal',str(internal),'--append-overhead',str(overhead),
             '--control',env['ASYMMETRY_CONTROL'],'--output',str(out)]
        with (ROOT/f'artifacts/{tag}.log').open('w') as log:
            code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        ledger.append({'tag':tag,'command':cmd,'returncode':code})
        write(ROOT/'artifacts/campaign_ledger.json',ledger)
        if code:raise RuntimeError(f'failed {tag}; no retry')
    write(Path(os.environ['NATIVE_LP_OUTPUT'])/'status.json',{'ok':True})


if __name__=='__main__':main()
