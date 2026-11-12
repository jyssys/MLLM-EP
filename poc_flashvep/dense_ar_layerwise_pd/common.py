"""Keep the Dense AR campaign independent of frozen Looped studies."""
import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

REPO=Path('/home/esjung/MLLM-EP-github')
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/dense_ar_layerwise_pd_feasibility_20261112'
PREVIOUS=REPO/'poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111'
ATOMIC=REPO/'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'
RLT=Path('/home/esjung/external/looped-pd-abc-rlt')
PYTHON=Path('/home/esjung/.venvs/looped-pd-abc/bin/python')
BASE='99af71b9f8da513dd368c2214ea67b6e15449658'
BRANCH='flashvep/dense-ar-layerwise-pd-feasibility-20261112'
START=datetime(2026,11,12,5,50,17,tzinfo=timezone.utc).timestamp()
DEADLINE=START+10*3600
GPU_DEADLINE=DEADLINE-3600
LONG_START_CUTOFF=GPU_DEADLINE-20*60
MODEL_REPO='Qwen/Qwen3-8B'
MODEL_REVISION='b968826d9c46dd6066d109eabc6255188de91218'
MODEL=Path('/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots')/MODEL_REVISION


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def git(*args,cwd=REPO):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def preservation():
    return dict(worktree_HEAD=git('rev-parse','HEAD'),working_branch=git('branch','--show-current'),
        index_entries_SHA256=hashlib.sha256(subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)).hexdigest(),
        private_HEAD=git('rev-parse','HEAD',cwd=RLT),
        private_diff_SHA256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest())


def preserve():
    assert preservation()==json.loads((ROOT/'state/preservation.json').read_text())['signatures'],'User checkout, index, or external runtime changed'


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT=ROOT;owner.PRIOR=PREVIOUS
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log'))
    return owner


def initialize():
    for name in ('state','configs','raw','logs','data','analysis','figures','references'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    p=ROOT/'state/preservation.json'
    if p.exists():preserve();return
    write(p,dict(time=time.time(),publication_parent=BASE,signatures=preservation()))
    (ROOT/'state/private_runtime.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=RLT))
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    write(ROOT/'state/budget.json',dict(start_epoch=START,hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE,long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=10,timezone='Asia/Seoul',
        scope='Fresh Dense AR campaign. Native correctness precedes migration performance. Final analysis, publication, and all-eight Burner restoration are inside this budget.'))
    owner=resources();verified=owner.burners()
    for old in (PREVIOUS/'state').glob('burner_*.json'):
        value=json.loads(old.read_text())
        if not set(value.get('GPUs',[]))<=set(verified):continue
        pid=value['supervisor'];proc=Path('/proc')/str(pid)
        try:
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid and owner.SCRIPT in words and '--worker' not in words:
                write(ROOT/'state'/old.name,dict(value,adopted_from=str(old)))
        except (FileNotFoundError,ProcessLookupError):pass
    write(ROOT/'state/environment.json',dict(time=time.time(),
        GPU_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        active_GPU_processes=owner.processes(),approved_burners=verified,
        model_config=json.loads((MODEL/'config.json').read_text()),model_config_SHA256=digest(MODEL/'config.json')))
    preserve()


if __name__=='__main__':initialize();print(ROOT)
