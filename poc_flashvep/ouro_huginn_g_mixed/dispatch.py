"""Resume-safe one-process-per-GPU dispatcher with native warm barriers."""
import json
from .common import ROOT, FORK, PYTHON, digest
from .resources import stop, restore
from ..ouro_budget_joint_cg import dispatch as base
base.ROOT, base.FORK, base.PYTHON = ROOT, FORK, PYTHON
base.stop, base.restore = stop, restore
original = base.subprocess.Popen

def completed(job):
    path = ROOT / 'raw' / job['id'] / 'completed.json'
    if not path.exists(): return False
    for row in json.loads(path.read_text())['artifacts']:
        if row['path'] != 'telemetry.jsonl':
            assert digest(path.parent / row['path']) == row['sha256']
    return True
base.completed = completed

def popen(command, **kwargs):
    command = [s.replace('poc_flashvep.ouro_budget_joint_cg.worker',
        'poc_flashvep.ouro_huginn_g_mixed.worker') for s in command]
    if 'env' in kwargs: kwargs['env']['OPENBLAS_NUM_THREADS'] = '1'
    return original(command, **kwargs)

if __name__ == '__main__':
    base.subprocess.Popen = popen
    base.main()
