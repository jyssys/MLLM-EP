"""Preserve prior experiments and own only this campaign namespace."""
import hashlib,json,os,subprocess,time
from pathlib import Path

REPO=Path('/home/esjung/MLLM-EP-github')
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_slo_throughput_dynamic_layer_pd_20261114'
PREVIOUS=REPO/'poc_flashvep/reports/dense_ar_decode_safe_elastic_layer_pd_20261113'
LEGACY=REPO/'poc_flashvep/reports/dense_ar_dynamic_layer_pd_crossover_20261113'
STATIC=REPO/'poc_flashvep/reports/dense_ar_layerwise_pd_feasibility_20261112'
ATOMIC=REPO/'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'
RLT=Path('/home/esjung/external/looped-pd-abc-rlt')
PYTHON=Path('/home/esjung/.venvs/looped-pd-abc/bin/python')
BASE='92e5275f1175bc08680de77180b8f5c652f2d08f'
BRANCH='flashvep/dense-ar-slo-throughput-dynamic-layer-pd-20261114'
MODEL_REVISION='b968826d9c46dd6066d109eabc6255188de91218'
MODEL=Path('/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots')/MODEL_REVISION
QUIET_LOCK=STATIC/'state/quiet_lane.lock'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def source_hash_manifest():
    path=os.environ.get('DENSE_SOURCE_HASHES_FILE')
    if path:return json.loads(Path(path).read_text())
    return json.loads(os.environ.get('DENSE_SOURCE_HASHES','{}'))


def git(*args,cwd=REPO):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def append(path,value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def preservation():
    changed=subprocess.check_output(['git','diff','--name-only','--diff-filter=AM','-z'],cwd=REPO).split(b'\0')
    return dict(worktree_HEAD=git('rev-parse','HEAD'),working_branch=git('branch','--show-current'),
        index_entries_SHA256=hashlib.sha256(subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)).hexdigest(),
        tracked_change_names_SHA256=hashlib.sha256(subprocess.check_output(['git','diff','--name-status','-z'],cwd=REPO)).hexdigest(),
        modified_tracked_files={os.fsdecode(p):digest(REPO/os.fsdecode(p)) for p in changed if p},
        private_HEAD=git('rev-parse','HEAD',cwd=RLT),
        private_diff_SHA256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest())


def preserve():
    assert preservation()==json.loads((ROOT/'state/preservation.json').read_text())['signatures'],'Existing user state changed'


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT=ROOT;owner.PRIOR=PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log',LEGACY/'burner.log',STATIC/'burner.log'))
    return owner


def initialize():
    for name in ('state','configs','raw','logs','data','analysis','figures','references'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    path=ROOT/'state/preservation.json'
    if path.exists():preserve();return
    start=time.time();write(path,dict(time=start,signatures=preservation(),publication_parent=BASE))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    frozen={str(p.relative_to(REPO)):digest(p) for package in ('dense_ar_layerwise_pd','dense_ar_dynamic_layer_pd','dense_ar_elastic_layer_pd')
        for p in (REPO/'poc_flashvep'/package).glob('*.py')}
    for name in ('dense_ar_decode_safe_elastic_layer_pd_final_summary.md','dense_ar_four_method_qps_sweep_results.md',
        'dense_ar_dynamic_layer_cut_decision_analysis.md','dense_ar_layer_cost_model_and_deadline_validation.md',
        'dense_ar_static_dynamic_matched_migration_comparison.md','dense_ar_continuous_colocated_correctness_and_baseline.md'):
        report=REPO/'poc_flashvep/reports'/name;frozen[str(report.relative_to(REPO))]=digest(report)
    write(ROOT/'state/frozen_predecessor_hashes.json',frozen)
    owner=resources();approved=owner.burners();actual=owner.processes()
    assert all(os.getsid(row['pid'])==approved.get(row['gpu']) for row in actual),'Unknown GPU owner remains untouched'
    write(ROOT/'state/environment.json',dict(time=start,approved_burners=approved,actual_GPU_processes=actual,
        GPU_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        model_config=json.loads((MODEL/'config.json').read_text()),model_config_SHA256=digest(MODEL/'config.json')))
    for name,subdir in (('installed_runtime_sources.json','state'),('frozen_SLO_regimes.json','configs'),
        ('revised_cost_model_v1.json','configs'),('revised_cost_model_v2.json','configs'),
        ('revised_cost_model_v3.json','configs'),('calibrated_cost_model_previous.json','configs')):
        write(ROOT/subdir/name,json.loads((PREVIOUS/subdir/name).read_text()))
    assert digest(ROOT/'configs/frozen_SLO_regimes.json')=='c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9'
    for name in ('qualification','calibration','discovery'):
        write(ROOT/'data'/(name+'.json'),json.loads((PREVIOUS/'data'/(name+'.json')).read_text()))
    write(ROOT/'state/progress.json',dict(start_epoch=start,phase='SLO_FRONTIER_AND_ZERO_MIGRATION_AUDIT',
        experiment_review_epoch=start+18.5*3600,closeout_reserve_seconds=5400,
        scope='A planning ceiling, not a required duration. Stop weak hypotheses early. Preserve resume records and all failed evidence.'))
    preserve()


def save(cfg):
    initialize();path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg,'Do not overwrite an immutable configuration'
    else:write(path,cfg)
    return cfg['name']


if __name__=='__main__':initialize();print(ROOT)
