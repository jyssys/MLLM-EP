"""Keep the new axis campaign separate from the published BI0 evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from ..dense_ar_slo_layer_pd import common as foundation

REPO = foundation.REPO
SOURCE = Path(__file__).parent
PREVIOUS = REPO/'poc_flashvep/reports/dense_ar_bi0_shortanswer200_20261120'
PREVIOUS_SOURCE = REPO/'poc_flashvep/dense_ar_bi0_shortanswer_pd'
ROOT = REPO/'poc_flashvep/reports/dense_ar_bi0_axis_rootcause_20261120'
BASE = '12e0f370a26ec544af81a90577874527ce8d4c5b'
BRANCH = 'flashvep/dense-ar-bi0-axis-rootcause-20261120'
START = 1795149432.0  # 2026-11-20 13:37:12 KST.
PYTHON, MODEL, MODEL_REVISION = foundation.PYTHON, foundation.MODEL, foundation.MODEL_REVISION
QUIET_LOCK = foundation.QUIET_LOCK
write, digest, git, preservation = foundation.write, foundation.digest, foundation.git, foundation.preservation


def preserve():
    expected = json.loads((ROOT/'state/preservation.json').read_text())['signatures']
    assert preservation() == expected


def resources():
    from ..dense_ar_native_reproduction.common import resources as prior_resources
    owner = prior_resources()
    owner.ROOT, owner.PRIOR = ROOT, PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log', PREVIOUS/'burner.log'))
    from .resource_guard import stop
    owner.stop=lambda gpus:stop(owner,gpus)
    return owner


def initialize():
    for name in ('state','configs','data','raw','logs','analysis','figures','references','tools'):
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    if (ROOT/'state/preservation.json').exists():
        preserve()
        return
    assert json.loads((PREVIOUS/'state/post_publication_final_check.json').read_text())['passed']
    assert not git('branch','--list',BRANCH)
    write(ROOT/'state/preservation.json', dict(time=time.time(), signatures=preservation(), publication_parent=BASE))
    subprocess.run(['git','branch',BRANCH,BASE], cwd=REPO, check=True)
    write(ROOT/'state/budget.json',dict(start_epoch=START, deadline_epoch=START+10*3600,
        last_large_dispatch_epoch=START+9*3600, hours=10,
        scope='Reserve the final hour for verification, publication and resource closeout.'))
    copied=[]
    paths=['data/'+name for name in ('longbench_shortanswer200_v2.json','qualification8.json',
        'qualification32.json','cost_calibration20.json','cost_validation15.json',
        'context_boundary3.json','shortanswer200_v2_sampling_frozen.json','shortanswer200_v2_truncation_manifest.json')]
    paths += ['raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json']
    paths += ['analysis/'+name for name in ('BI0_axis_state_qualification_fast128_v3.json',
        'BI0_Token_same_geometry_gate_fast128_v3.json')]
    paths += ['state/'+name for name in ('fast_native_control_state_v3_complete.json',
        'controlled_axis_structural_gate_fast128_v3.json')]
    for relative in paths:
        source,target=PREVIOUS/relative,ROOT/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert digest(source)==digest(target)
        copied.append(dict(path=relative,SHA256=digest(target),source=str(source),evidence='HISTORICAL_REUSED'))
    shutil.copytree(PREVIOUS/'tools/quality_deps',ROOT/'tools/quality_deps')
    write(ROOT/'state/inherited_evidence_manifest.json',dict(files=copied,
        scope='Frozen inputs, Natural reference and prior state gates. These copies are not new GPU measurements.'))
    old_sources={str(p.relative_to(REPO)):digest(p) for p in PREVIOUS_SOURCE.glob('*.py')}
    write(ROOT/'state/prior_source_hashes.json',old_sources)
    reports=('dense_ar_bi0_shortanswer200_layer_token_dynamic_final_summary.md',
        'dense_ar_bi0_layer_token_sweetspot_knob_analysis.md','dense_ar_bi0_online_serving_qps_results.md',
        'dense_ar_bi0_axis_selector_oracle_analysis.md','dense_ar_matched_axis_crossover_final_summary.md')
    write(ROOT/'state/prior_report_hashes.json',{name:digest(REPO/'poc_flashvep/reports'/name) for name in reports})
    owner=resources();owned,active=owner.burners(),owner.processes()
    assert all(os.getsid(r['pid'])==owned.get(r['gpu']) for r in active)
    write(ROOT/'state/environment.json',dict(time=time.time(),model_revision=MODEL_REVISION,
        GPU_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version','--format=csv'],text=True),
        approved_burners=owned,active_processes=active))
    write(ROOT/'state/progress.json',dict(time=time.time(),status='ACTIVE',phase='LAST12_REPLICATION_AND_CREDIT_AUDIT'))
    preserve()


def save(cfg):
    initialize()
    assert cfg['batch_invariant'] is False
    if cfg.get('entry_module')=='poc_flashvep.dense_ar_bi0_shortanswer_pd.serving':
        cfg['entry_module']='poc_flashvep.dense_ar_bi0_axis_rootcause.serving'
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():
        assert json.loads(path.read_text())==cfg
    else:
        write(path,cfg)
    return cfg['name']


def activate():
    """Bind reused helper modules to this process-local evidence namespace."""
    from ..dense_ar_bi0_shortanswer_pd import common as prior
    for name,value in dict(ROOT=ROOT,SOURCE=SOURCE,BASE=BASE,BRANCH=BRANCH,PREVIOUS=PREVIOUS,
        resources=resources,preserve=preserve,initialize=initialize,save=save).items():
        setattr(prior,name,value)


activate()

if __name__=='__main__':
    initialize()
    print(ROOT)
