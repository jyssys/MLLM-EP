"""Isolate this campaign without changing the user's worktree or old evidence."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO=Path('/home/esjung/MLLM-EP-github')
ROOT=REPO/'poc_flashvep/reports/looped_pd_autonomous_prefill_compute_decode_headroom'
PREVIOUS=REPO/'poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom'
ATOMIC=REPO/'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'
SOURCE=Path(__file__).parent
PYTHON=Path('/home/esjung/.venvs/looped-pd-abc/bin/python')
RLT=Path('/home/esjung/external/looped-pd-abc-rlt')
BASE='2d683cce8ebc926c9d410944b705cbcb6f6031a0'
BRANCH='flashvep/looped-pd-autonomous-prefill-headroom'
START=datetime(2026,11,3,2,20,45,tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE=START+9*3600
GPU_DEADLINE=START+7.75*3600
LONG_START_CUTOFF=START+7*3600
EXTENSION=ROOT/'state/authorized_extension.json'
if EXTENSION.exists():
    extension=json.loads(EXTENSION.read_text())
    DEADLINE=extension['hard_deadline_epoch']
    GPU_DEADLINE=extension['GPU_stop_epoch']
    LONG_START_CUTOFF=extension['long_job_start_cutoff_epoch']


def setup():
    for name in ('state','configs','data','raw','logs','figures'):(ROOT/name).mkdir(parents=True,exist_ok=True)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def append(path,value):
    with Path(path).open('a') as handle:handle.write(json.dumps(value,allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def index_signature():
    return hashlib.sha256(subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)).hexdigest()


def preserve():
    v=json.loads((ROOT/'state/preservation.json').read_text())
    assert index_signature()==v['index_entries_SHA256'],'Original staged index changed'
    assert hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest()==v['private_diff_SHA256'],'External runtime changed'


def activate():
    """Redirect process-local imports before any reused worker module loads."""
    from ..looped_pd_headroom_recovery import common as old
    old.ROOT=ROOT;old.setup=setup;old.preserve=preserve
    old.BRANCH=BRANCH
    old.resources=resources
    return old.configure()


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT=ROOT;owner.PRIOR=ATOMIC
    owner.APPROVED_LOGS.update((ROOT/'burner.log',PREVIOUS/'burner.log',ATOMIC/'burner.log'))
    return owner


def initialize():
    setup()
    path=ROOT/'state/preservation.json'
    if path.exists():preserve();return
    private=subprocess.check_output(['git','diff','--binary'],cwd=RLT)
    write(path,dict(time=time.time(),base=BASE,worktree_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        working_branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        index_entries_SHA256=index_signature(),private_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RLT,text=True).strip(),
        private_diff_SHA256=hashlib.sha256(private).hexdigest()))
    (ROOT/'state/private_runtime.patch').write_bytes(private)
    subprocess.run(['git','branch',BRANCH,BASE],cwd=REPO,check=True)
    write(ROOT/'state/budget.json',dict(start_epoch=START,hard_deadline_epoch=DEADLINE,GPU_stop_epoch=GPU_DEADLINE,
        long_job_start_cutoff_epoch=LONG_START_CUTOFF,maximum_wall_hours=9,
        timezone='Asia/Seoul',scope='New campaign. No prior deadline is inherited. Publication and resource restoration are inside this budget.'))
    # Adopt only supervisor records whose workers already satisfy the owner's
    # UID, executable, SID and approved-output-log checks. Originals stay intact.
    owner=resources();verified=owner.burners()
    for p in (PREVIOUS/'state').glob('burner_*.json'):
        v=json.loads(p.read_text())
        if not set(v.get('GPUs',[]))<=set(verified):continue
        pid=v['supervisor'];proc=Path('/proc',str(pid))
        try:
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid and owner.SCRIPT in words and '--worker' not in words:
                write(ROOT/'state'/p.name,dict(v,adopted_from=str(p)))
        except (FileNotFoundError,ProcessLookupError):pass
    write(ROOT/'state/environment.json',dict(time=time.time(),GPU_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        active_GPU_processes=owner.processes(),approved_burners=verified))
    preserve()
