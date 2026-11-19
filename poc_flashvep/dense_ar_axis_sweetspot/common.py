"""Preserve the published baseline and use a separate evidence namespace."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ..dense_ar_slo_layer_pd import common as foundation

REPO=foundation.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_axis_sweetspot_knob_discovery_20261119'
PREVIOUS=REPO/'poc_flashvep/reports/dense_ar_matched_axis_crossover_20261117'
BASE='4cf3f9b80a2a8618da3bdddb1140fab7f403c3c7'
BRANCH='flashvep/dense-ar-axis-sweetspot-knob-discovery-20261119'
PYTHON,MODEL,MODEL_REVISION=foundation.PYTHON,foundation.MODEL,foundation.MODEL_REVISION
QUIET_LOCK=foundation.QUIET_LOCK
write,digest,git,preservation=foundation.write,foundation.digest,foundation.git,foundation.preservation
START=datetime.datetime.fromisoformat('2026-11-19T01:14:22+09:00').timestamp()
DEADLINE=START+10*3600
PACKAGES=('dense_ar_matched_axis_pd','dense_ar_layer_token_pd','duetserve_agg_decode',
          'dense_ar_slo_layer_pd','dense_ar_elastic_layer_pd','dense_ar_dynamic_layer_pd',
          'dense_ar_layerwise_pd')
REPORTS=('dense_ar_matched_axis_crossover_final_summary.md',
         'dense_ar_layer_token_crossover_root_cause.md',
         'dense_ar_batch_matched_axis_tradeoff_protocol.md',
         'dense_ar_matched_axis_output_sweep_figures.md',
         'dense_ar_layer_token_dynamic_headroom_final_summary.md')


def preserve():
    expected=json.loads((ROOT/'state/preservation.json').read_text())['signatures']
    assert preservation()==expected, 'User checkout, index or external runtime state changed'


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT,owner.PRIOR=ROOT,PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log',
        foundation.ROOT/'burner.log',foundation.PREVIOUS/'burner.log',foundation.STATIC/'burner.log'))
    return owner


def initialize():
    for name in ('state','configs','data','raw','logs','analysis','figures','references'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    target=ROOT/'state/preservation.json'
    if target.exists():
        preserve()
        return
    assert git('rev-parse',BASE)==BASE
    assert not git('branch','--list',BRANCH), 'Preserve an existing experiment branch'
    write(target,dict(time=time.time(),signatures=preservation(),publication_parent=BASE))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    files=[p for package in PACKAGES for p in (REPO/'poc_flashvep'/package).glob('*.py')]
    files.extend(REPO/'poc_flashvep/reports'/name for name in REPORTS)
    write(ROOT/'state/frozen_predecessor_hashes.json',{str(p.relative_to(REPO)):digest(p) for p in files})
    names=[str(p.relative_to(REPO)) for p in files if p.suffix=='.py']
    blobs=subprocess.check_output(['git','hash-object','--stdin-paths'],cwd=REPO,
        input=''.join(n+'\n' for n in names).encode()).decode().splitlines()
    tree=subprocess.check_output(['git','ls-tree','-r',BASE,'--',
        *['poc_flashvep/'+p for p in PACKAGES]],cwd=REPO,text=True)
    declared={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
    assert all(declared.get(n)==b for n,b in zip(names,blobs)), 'Current predecessor differs from declared base'
    write(ROOT/'state/base_source_audit.json',dict(base=BASE,files=len(names),equal_to_base=True,
        SHA256={n:digest(REPO/n) for n in names}))
    copies={
        'configs':('frozen_SLO_regimes.json','common_output_sweep_SLO.json',
                   'verified_ByteLevel_text_priming.json','revised_cost_model_v1.json'),
        'state':('installed_runtime_sources.json','canonical_block_runtime_contract.json',
                 'frozen_new_inference_sources_v3.json','design_freeze_v3_before_performance.json'),
        'analysis':('exact_phase_norm_GPU_gate.json','phase_mask_norm_GPU_gate.json'),
        'data':('matched144.json','heldout128_8000.json','qualification.json','discovery.json')}
    for directory,entries in copies.items():
        for name in entries:
            source=PREVIOUS/directory/name
            destination=ROOT/directory/name
            shutil.copyfile(source,destination)
            assert digest(source)==digest(destination)
    owner=resources();owned=owner.burners();active=owner.processes()
    assert all(os.getsid(r['pid'])==owned.get(r['gpu']) for r in active), 'Unknown GPU owner remains untouched'
    write(ROOT/'state/environment.json',dict(time=time.time(),model_revision=MODEL_REVISION,
        GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        approved_burners=owned,active_processes=active,model_config=json.loads((MODEL/'config.json').read_text())))
    write(ROOT/'state/research_budget.json',dict(start_epoch=START,deadline_epoch=DEADLINE,
        last_new_GPU_start_epoch=DEADLINE-3600,publication_reserve_seconds=3600,
        start_local='2026-11-19T01:14:22+09:00',deadline_local='2026-11-19T11:14:22+09:00',
        scope='A ten-hour research budget. Stop weak hypotheses early. Preserve incomplete and failed work.'))
    write(ROOT/'state/progress.json',dict(start_epoch=START,status='ACTIVE',phase='EXISTING_EVENT_CAUSAL_AUDIT'))
    preserve()


def bind_legacy():
    import importlib
    for package in PACKAGES:
        module=importlib.import_module('poc_flashvep.'+package+'.common')
        for key,value in dict(ROOT=ROOT,BASE=BASE,BRANCH=BRANCH,initialize=initialize,
                              preserve=preserve,resources=resources).items():
            setattr(module,key,value)
        if package=='dense_ar_matched_axis_pd':
            module.bind_legacy=bind_legacy
    prefixes=tuple('poc_flashvep.'+p+'.' for p in PACKAGES)
    for name,module in list(sys.modules.items()):
        if name.startswith(prefixes) and hasattr(module,'ROOT'):
            module.ROOT=ROOT


def save(cfg):
    initialize()
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():
        assert json.loads(path.read_text())==cfg, 'Immutable experiment configuration changed'
    else:
        write(path,cfg)
    return cfg['name']


if __name__=='__main__':
    initialize()
    print(ROOT)
