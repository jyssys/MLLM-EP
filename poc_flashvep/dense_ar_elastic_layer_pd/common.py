"""Keep the dynamic campaign separate from all frozen experiments."""
import hashlib,json,os,subprocess,time
from pathlib import Path

REPO=Path('/home/esjung/MLLM-EP-github')
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_decode_safe_elastic_layer_pd_20261113'
PREVIOUS=REPO/'poc_flashvep/reports/dense_ar_dynamic_layer_pd_crossover_20261113'
STATIC=REPO/'poc_flashvep/reports/dense_ar_layerwise_pd_feasibility_20261112'
ATOMIC=REPO/'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'
RLT=Path('/home/esjung/external/looped-pd-abc-rlt')
PYTHON=Path('/home/esjung/.venvs/looped-pd-abc/bin/python')
BASE='96414d7baac14dea2d697198b2c595850ae83661'
BRANCH='flashvep/dense-ar-decode-safe-elastic-layer-pd-20261113'
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


def git(*args,cwd=REPO):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def append(path,value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def preservation():
    return dict(worktree_HEAD=git('rev-parse','HEAD'),working_branch=git('branch','--show-current'),
        index_entries_SHA256=hashlib.sha256(subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)).hexdigest(),
        private_HEAD=git('rev-parse','HEAD',cwd=RLT),
        private_diff_SHA256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest())


def preserve():
    assert preservation()==json.loads((ROOT/'state/preservation.json').read_text())['signatures'],'Existing user state changed'


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT=ROOT;owner.PRIOR=PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log',STATIC/'burner.log'))
    return owner


def initialize():
    for name in ('state','configs','raw','logs','data','analysis','figures','references'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    path=ROOT/'state/preservation.json'
    if path.exists():preserve();return
    start=time.time();write(path,dict(time=start,signatures=preservation(),publication_parent=BASE))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    frozen={str(p.relative_to(REPO)):digest(p) for package in ('dense_ar_layerwise_pd','dense_ar_dynamic_layer_pd')
        for p in (REPO/'poc_flashvep'/package).glob('*.py')}
    for name in ('dense_ar_dynamic_layer_pd_crossover_final_summary.md','dense_ar_pd_colocated_qps_crossover_analysis.md',
        'dense_ar_dynamic_layer_cost_model_validation.md','dense_ar_dynamic_layer_decode_safe_serving_results.md',
        'dense_ar_slo_calibration_and_goodput_analysis.md'):
        report=REPO/'poc_flashvep/reports'/name;frozen[str(report.relative_to(REPO))]=digest(report)
    write(ROOT/'state/frozen_predecessor_hashes.json',frozen)
    owner=resources();approved=owner.burners();actual=owner.processes()
    assert all(os.getsid(row['pid'])==approved.get(row['gpu']) for row in actual),'Unknown GPU owner remains untouched'
    write(ROOT/'state/environment.json',dict(time=start,approved_burners=approved,actual_GPU_processes=actual,
        GPU_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        model_config=json.loads((MODEL/'config.json').read_text()),model_config_SHA256=digest(MODEL/'config.json')))
    for old,new in ((PREVIOUS/'state/installed_runtime_sources.json',ROOT/'state/installed_runtime_sources.json'),
        (PREVIOUS/'configs/frozen_SLO_regimes.json',ROOT/'configs/frozen_SLO_regimes.json')):
        write(new,json.loads(old.read_text()))
    write(ROOT/'state/progress.json',dict(start_epoch=start,phase='PHASE_1_CONTINUOUS_CORRECTNESS',
        scope='Resume manifest. Existing results are frozen. A completed runtime does not imply correctness or qualified timing.'))
    preserve()


def save(cfg):
    initialize();path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg,'Do not overwrite an immutable configuration'
    else:write(path,cfg)
    return cfg['name']


if __name__=='__main__':initialize();print(ROOT)
