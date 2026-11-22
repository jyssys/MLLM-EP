"""Own only v2 artifacts and preserve the user's checkout and index."""
import json
import os
from pathlib import Path
import subprocess
import time
from ..dense_ar_slo_layer_pd import common as foundation

REPO=foundation.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_longbench400_v2_20261122'
PREVIOUS=REPO/'poc_flashvep/reports/dense_ar_longbench400_2d_20261121'
BASE='5ebed239073b4fa3709361038cbec026d1c32b78'
BRANCH='flashvep/longbench400-v2-output-aware-2d-20261122'
START=1795331210
PYTHON,MODEL,MODEL_REVISION=foundation.PYTHON,foundation.MODEL,foundation.MODEL_REVISION
REVISION='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
QUIET_LOCK=foundation.QUIET_LOCK
write,digest,preservation=foundation.write,foundation.digest,foundation.preservation

def time_available(field='deadline_epoch',budget=None):
    budget=budget or json.loads((ROOT/'state/budget.json').read_text())
    return time.time()<budget[field]

def dispatch_allowed(budget=None):
    return not (ROOT/'state/STOP_DISPATCH.json').exists() and time_available('last_large_dispatch_epoch',budget)

def execution_allowed(budget=None):
    return not (ROOT/'state/STOP_EXECUTION.json').exists() and time_available('deadline_epoch',budget)

def preserve():
    old=json.loads((ROOT/'state/preservation.json').read_text())
    assert preservation()==old['signatures'],'User checkout or index changed; no automatic repair'
    for path,sha in old['frozen_predecessors'].items():assert digest(REPO/path)==sha,path

def resources():
    from ..dense_ar_native_reproduction.common import resources as previous_resources
    from ..dense_ar_bi0_axis_rootcause.resource_guard import stop
    owner=previous_resources();owner.ROOT,owner.PRIOR=ROOT,PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log'))
    owner.stop=lambda gpus:stop(owner,gpus)
    return owner

def initialize():
    for name in ('state','configs','data','raw','logs','analysis','figures','references','tools','jobs'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    path=ROOT/'state/preservation.json'
    if path.exists():preserve();return
    assert json.loads((PREVIOUS/'state/post_publication_final_check.json').read_text())['passed']
    assert subprocess.check_output(['git','rev-parse',BASE],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','branch','--list',BRANCH],cwd=REPO,text=True).strip()
    frozen={str(p.relative_to(REPO)):digest(p) for p in (REPO/'poc_flashvep/dense_ar_longbench400_pd').glob('*.py')}
    for p in (REPO/'poc_flashvep/reports').glob('dense_ar_longbench400*.md'):frozen[str(p.relative_to(REPO))]=digest(p)
    for p in (REPO/'poc_flashvep/reports').glob('dense_ar_layer_token_2d*.md'):frozen[str(p.relative_to(REPO))]=digest(p)
    write(path,dict(time=time.time(),signatures=preservation(),publication_parent=BASE,frozen_predecessors=frozen))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    owner=resources();approved=owner.burners();active=owner.processes()
    assert all(os.getsid(r['pid'])==approved.get(r['gpu']) for r in active),'Unknown GPU owner remains untouched'
    write(ROOT/'state/environment_initial.json',dict(time=time.time(),GPU_inventory=subprocess.check_output([
        'nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),active=active,approved=approved,
        model_revision=MODEL_REVISION,burner_script_SHA256=digest(Path(owner.SCRIPT))))
    write(ROOT/'state/budget.json',dict(start_epoch=START,deadline_epoch=START+20*3600,
        last_large_dispatch_epoch=START+18*3600,maximum_hours=20,closeout_reserve_hours=2))
    write(ROOT/'state/progress.json',dict(time=time.time(),status='ACTIVE',phase='CANDIDATE_AVAILABILITY_AUDIT'))
    preserve()

if __name__=='__main__':initialize();print(ROOT)
