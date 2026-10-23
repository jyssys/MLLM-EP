"""Resume-safe independent GPU queues using the audited dispatcher."""
import os
import json
import subprocess
import sys
from .common import ROOT, FORK, REPO, PYTHON, digest
from .resources import stop, restore
from ..ouro_budget_joint_cg import dispatch as base

base.ROOT=ROOT
base.FORK=FORK
base.PYTHON=PYTHON
base.stop=stop
base.restore=restore
original_popen=base.subprocess.Popen

def completed(job):
    path=ROOT/'raw'/job['id']/'completed.json'
    if not path.exists():return False
    data=json.loads(path.read_text())
    for item in data['artifacts']:
        # The dispatcher owns telemetry. It can append one final sample after
        # the worker writes completed.json. Analysis hashes it after process exit.
        if item['path']=='telemetry.jsonl':continue
        artifact=path.parent/item['path']
        assert artifact.exists() and digest(artifact)==item['sha256'],('CORRUPTED_IMMUTABLE',artifact)
    return True

base.completed=completed

def scoped_popen(command, **kwargs):
    # Replace only this worker entry point. Preserve launcher/affinity checks.
    command=[v.replace('poc_flashvep.ouro_budget_joint_cg.worker','poc_flashvep.ouro_elb.worker') for v in command]
    if 'poc_flashvep.ouro_elb.worker' in command and '--job' in command:
        job=json.loads(open(command[command.index('--job')+1]).read())
        if job['kind']=='nsight':
            output=ROOT/'profiles'/job['id'];output.parent.mkdir(parents=True,exist_ok=True)
            command=command[:3]+['nsys','profile','--trace=cuda,nvtx,osrt','--cuda-graph-trace=node',
                '--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--cpuctxsw=none',
                '--force-overwrite=false','-o',str(output)]+command[3:]
    if 'env' in kwargs:
        kwargs['env']['OPENBLAS_NUM_THREADS']='1'
    return original_popen(command, **kwargs)

if __name__=='__main__':
    base.subprocess.Popen=scoped_popen
    base.main()
