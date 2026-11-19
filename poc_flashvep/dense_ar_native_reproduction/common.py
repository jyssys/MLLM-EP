"""Own a Native-only namespace without changing the user checkout."""
import json
import os
from pathlib import Path
import subprocess
import time
from ..dense_ar_slo_layer_pd import common as foundation

REPO=foundation.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_native_pd_real_workload_reproduction_20261119'
PREVIOUS=REPO/'poc_flashvep/reports/dense_ar_decode_safe_layer_token_offloading_20261119'
BASE='4595080c4b4d451f4e15e5eb3f61c771981e6b54'
BRANCH='flashvep/dense-ar-native-pd-real-workload-reproduction-20261119'
PYTHON,MODEL,MODEL_REVISION=foundation.PYTHON,foundation.MODEL,foundation.MODEL_REVISION
QUIET_LOCK=foundation.QUIET_LOCK
write,digest,git,preservation=foundation.write,foundation.digest,foundation.git,foundation.preservation


def preserve():
    assert preservation()==json.loads((ROOT/'state/preservation.json').read_text())['signatures']


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT,owner.PRIOR=ROOT,PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log',
        foundation.ROOT/'burner.log',foundation.PREVIOUS/'burner.log',foundation.STATIC/'burner.log'))
    return owner


def initialize():
    for name in ('state','configs','data','raw','logs','analysis','figures','references','tools'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    path=ROOT/'state/preservation.json'
    if path.exists():preserve();return
    assert (PREVIOUS/'state/STOPPED_BY_RESEARCH_REDIRECTION.json').exists()
    assert not git('branch','--list',BRANCH)
    write(path,dict(time=time.time(),signatures=preservation(),publication_parent=BASE))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    owner=resources();approved=owner.burners();active=owner.processes()
    assert all(os.getsid(r['pid'])==approved.get(r['gpu']) for r in active)
    write(ROOT/'state/environment.json',dict(time=time.time(),model_revision=MODEL_REVISION,
        GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version','--format=csv'],text=True),
        approved_burners=approved,active_processes=active,model_config=json.loads((MODEL/'config.json').read_text())))
    reports=('dense_ar_axis_service_queue_causal_decomposition.md',
        'dense_ar_axis_sweetspot_knob_discovery_final_summary.md',
        'dense_ar_axis_batch_matched_online_results.md',
        'dense_ar_prefill_heavy_layerwise_pd_feasibility_final_summary.md',
        'dense_ar_dynamic_layer_pd_crossover_final_summary.md')
    write(ROOT/'state/prior_report_hashes.json',{n:digest(REPO/'poc_flashvep/reports'/n) for n in reports})
    write(ROOT/'state/progress.json',dict(start_epoch=time.time(),status='ACTIVE',
        phase='NATIVE_RUNTIME_AUDIT_AND_STANDARD_VLLM_REFERENCE',
        forbidden='No new Layer/Token offloading measurements or fixed-length serving sweeps. Fixed lengths are diagnostic controls only.'))
    preserve()


def save(cfg):
    initialize();path=ROOT/'configs'/(cfg['name']+'.json')
    assert cfg.get('method','Native') in ('Native','Agg')
    assert not cfg.get('token_axis') and cfg.get('static_layers',0)==0
    if path.exists():assert json.loads(path.read_text())==cfg
    else:write(path,cfg)
    return cfg['name']


if __name__=='__main__':initialize();print(ROOT)
