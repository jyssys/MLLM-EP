"""Resume-safe independent CUDA contexts with inherited ownership checks."""
import json
from .common import ROOT,FORK,PYTHON,digest
from .resources import stop,restore
from ..ouro_budget_joint_cg import dispatch as base
base.ROOT=ROOT
base.FORK=FORK
base.PYTHON=PYTHON
base.stop=stop
base.restore=restore
original=base.subprocess.Popen

def completed(job):
    path=ROOT/'raw'/job['id']/'completed.json'
    if not path.exists():return False
    data=json.loads(path.read_text())
    for item in data['artifacts']:
        if item['path']=='telemetry.jsonl':continue
        assert digest(path.parent/item['path'])==item['sha256']
    return True
base.completed=completed

def popen(command,**kwargs):
    command=[v.replace('poc_flashvep.ouro_budget_joint_cg.worker','poc_flashvep.ouro_kv_residency.worker') for v in command]
    if 'env' in kwargs:kwargs['env']['OPENBLAS_NUM_THREADS']='1'
    if '--job' in command:
        job=json.loads(open(command[command.index('--job')+1]).read())
        if job.get('nsys'):
            out=ROOT/'profiles'/job['id'];out.parent.mkdir(parents=True,exist_ok=True)
            command=command[:3]+['nsys','profile','--trace=cuda,nvtx,osrt','--cuda-graph-trace=node',
                '--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none',
                '--cpuctxsw=none','--force-overwrite=false','-o',str(out)]+command[3:]
    return original(command,**kwargs)

if __name__=='__main__':
    base.subprocess.Popen=popen
    base.main()
