"""Keep this feasibility audit separate from frozen GPU campaigns."""
import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

REPO=Path('/home/esjung/MLLM-EP-github')
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/ouro_huginn_remote_ready_cross_depth_pd_20261113'
RLT=Path('/home/esjung/external/looped-pd-abc-rlt')
OLD=REPO/'poc_flashvep/reports/rcc_pd'
DENSE=REPO/'poc_flashvep/reports/dense_ar_layerwise_pd_feasibility_20261112'
PARENT='367d50336791b4c540404053eadf924544a2ca0f'
BRANCH='flashvep/remote-ready-cross-depth-pd-poc-20261113'
START=datetime(2026,11,12,15,41,5,tzinfo=timezone.utc).timestamp()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def git(*args,cwd=REPO):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def signatures():
    return dict(HEAD=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        index_SHA256=hashlib.sha256(subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)).hexdigest(),
        external_HEAD=git('rev-parse','HEAD',cwd=RLT),
        external_diff_SHA256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest())


def owner():
    from ..ouro_budget_joint_cg import resources
    resources.ROOT=ROOT;resources.PRIOR=DENSE
    resources.APPROVED_LOGS.update((ROOT/'burner.log',DENSE/'burner.log'))
    return resources


def resource_snapshot():
    service=owner();owned=service.burners();active=service.processes()
    assert all(os.getsid(r['pid'])==owned.get(r['gpu']) and Path('/proc',str(r['pid'])).stat().st_uid==os.getuid() for r in active),'Unknown GPU process remains untouched'
    return dict(time=time.time(),owned_sessions=owned,actual_GPU_processes=active,
        all_eight_active={r['gpu'] for r in active}==set(range(8)),
        NVML=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True))


def initialize():
    path=ROOT/'state/preservation.json'
    if path.exists():preserve();return
    for name in ('state','analysis','raw','references','figures'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    start=START
    write(path,dict(time=start,signatures=signatures(),scope='Read-only audit of existing evidence first. Preserve staged work and the external runtime.'))
    write(ROOT/'state/budget.json',dict(start_epoch=start,hard_deadline_epoch=start+4*3600,
        new_GPU_job_cutoff_epoch=start+3.5*3600,maximum_wall_hours=4,
        start_UTC=datetime.fromtimestamp(start,timezone.utc).isoformat(),
        policy='Stop at verified equivalence. A full four-hour run is not required.'))
    subprocess.run(['git','branch',BRANCH,PARENT],cwd=REPO,check=True)
    write(ROOT/'state/initial_resources.json',resource_snapshot())


def preserve():
    assert signatures()==json.loads((ROOT/'state/preservation.json').read_text())['signatures'],'Existing user or runtime state changed'


if __name__=='__main__':initialize();print(ROOT)
