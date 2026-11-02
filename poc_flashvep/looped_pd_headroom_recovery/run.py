"""Run one immutable namespace. Isolate timing and retain failed evidence."""
import argparse
import fcntl
import importlib
import signal
import zipfile
from .common import *


def belongs_to_session(pid,session):
    try:
        return Path('/proc',str(pid)).stat().st_uid==os.getuid() and os.getsid(pid)==session
    except (FileNotFoundError,ProcessLookupError):
        return False


def measured_requests_complete(folder,count):
    path=folder/'requests.jsonl'
    if not path.exists():return False
    try:
        values=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    except json.JSONDecodeError:return False
    return sum(not v.get('warmup',False) for v in values)>=count


def registered_profile_targets(folder):
    """Find only this namespace's verified owner processes, including new SIDs."""
    targets=[]
    for path in folder.glob('stage_owner_*.json'):
        value=json.loads(path.read_text())
        pid=value['pid']
        try:
            payload=Path(value['payload']).resolve()
            command=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
            expected=b'poc_flashvep.looped_pd_headroom_recovery.stage'
            if (value['uid']==os.getuid() and Path('/proc',str(pid)).stat().st_uid==os.getuid()
                and payload.parent==folder.resolve() and expected in command and str(payload).encode() in command
                and os.getsid(pid)==value['sid']):
                targets.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
    return targets


def execute(name, keep_quiet=True):
    setup()
    preserve()
    path=ROOT/'configs'/(name+'.json')
    cfg=json.loads(path.read_text())
    folder=ROOT/'raw'/name
    if folder.exists():
        result=folder/'dispatcher_result.json'
        if result.exists():return json.loads(result.read_text())['returncode']==0
        raise RuntimeError('Unfinished namespace requires explicit evidence review: '+name)
    owner=resources()
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Another resumed controller can have completed the same name while
        # this process waited for the lane. Recheck under the lock.
        if folder.exists():
            result=folder/'dispatcher_result.json'
            if result.exists():return json.loads(result.read_text()).get('complete',False)
            raise RuntimeError('Namespace became unfinished while waiting for the lane: '+name)
        owner.stop(list(range(8)))
        folder.mkdir()
        write(folder/'configuration.json',cfg)
        write(folder/'isolation.json',dict(time=time.time(),active=owner.processes(),approved_burners=owner.burners(),
            GPU_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free,driver_version','--format=csv'],text=True),
            lane='EXCLUSIVE_ALL_GPU_QUIET_LANE'))
        with zipfile.ZipFile(folder/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for package in ('looped_pd_headroom_recovery','looped_pd_atomic_prefill','looped_pd_bottleneck_crossover',
                'adaptive_loop_token_pd_2gpu','adaptive_recurrence_cut','looped_pd_abc','looped_pd_b_numerical_diagnosis'):
                for p in (REPO/'poc_flashvep'/package).rglob('*.py'):
                    archive.write(p,str(p.relative_to(REPO)))
            archive.write(ROOT/'state/private_runtime.patch','private_runtime.patch')
        env=environment(cfg['gpus'])
        env.update(ARC_CONFIG=str(path),ARC_RUN=name,ABC_LANE='isolated_headroom_recovery')
        hashes={str(p):digest(p) for p in (REPO/'poc_flashvep/looped_pd_headroom_recovery').rglob('*.py')}
        env['RECOVERY_SOURCE_HASHES']=json.dumps(hashes)
        command=[str(PYTHON),'-m','poc_flashvep.looped_pd_headroom_recovery.worker',str(path)]
        if cfg.get('local_nsys'):
            command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false',
                '--output='+str(folder/'p_only_trace')]+command
        stopped=False
        started=time.monotonic()
        with (ROOT/'logs'/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            append(ROOT/'commands.jsonl',dict(time=time.time(),name=name,pid=child.pid,command=command,
                source_SHA256=digest(folder/'source.zip'),config_SHA256=digest(path),
                git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
            write(ROOT/'state/active.json',dict(name=name,pid=child.pid,gpus=cfg['gpus'],time=time.time()))
            print('START',name,child.pid,flush=True)
            signature=None
            progress=time.monotonic()
            while child.poll() is None:
                paths=[folder/'requests.jsonl',folder/'cohorts.jsonl']
                current=[(p.name,p.stat().st_size) for p in paths if p.exists()]
                if current!=signature:signature=current;progress=time.monotonic()
                limit=cfg.get('startup_timeout',240)+cfg.get('timeout',300)
                reason=None
                if time.monotonic()-started>limit:reason='BOUNDED_JOB_WATCHDOG'
                profiling=cfg.get('P_only_profiler') or cfg.get('cuda_only_profile')
                measured_complete=measured_requests_complete(folder,cfg['count']) if profiling else False
                if profiling and signature and not measured_complete and time.monotonic()-progress>60:
                    reason='PROFILE_NO_RESPONSE_PROGRESS_60S'
                if profiling and measured_complete and time.monotonic()-progress>cfg.get('profile_finalization_timeout',300):
                    reason='PROFILE_FINALIZATION_TIMEOUT_AFTER_COMPLETE_COHORT'
                if reason:
                    for stackfile in folder.glob('worker_stacks_*.log'):
                        pid=int(stackfile.stem.rsplit('_',1)[1])
                        try:
                            if belongs_to_session(pid,child.pid) or pid in registered_profile_targets(folder):
                                os.kill(pid,signal.SIGUSR1)
                        except (FileNotFoundError,ProcessLookupError):pass
                    write(folder/'watchdog.json',dict(reason=reason,time=time.time(),progress=signature))
                    assert os.getsid(child.pid)==child.pid
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    stopped=True
                    break
                time.sleep(.5)
        status_path=folder/'status.json'
        leftovers=[]
        for row in owner.processes():
            if belongs_to_session(row['pid'],child.pid):leftovers.append(row)
        if leftovers:
            append(folder/'owned_session_cleanup.jsonl',dict(time=time.time(),registered_session=child.pid,GPUs=leftovers))
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            until=time.monotonic()+15
            while time.monotonic()<until:
                alive=[]
                for row in owner.processes():
                    if belongs_to_session(row['pid'],child.pid):alive.append(row)
                if not alive:break
                time.sleep(.5)
            if alive:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        targets=registered_profile_targets(folder)
        if targets:
            append(folder/'owned_profile_cleanup.jsonl',dict(time=time.time(),verified_native_target_PIDs=targets,
                scope='Only this namespace, exact module/payload, UID and recorded SID. Other profiler sessions are excluded.'))
            for pid in targets:
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            until=time.monotonic()+15
            while time.monotonic()<until and registered_profile_targets(folder):time.sleep(.5)
            for pid in registered_profile_targets(folder):
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        ok=child.returncode==0 and status.get('status')=='COMPLETE'
        write(folder/'dispatcher_result.json',dict(returncode=child.returncode,status=status.get('status'),
            watchdog=stopped,complete=ok,elapsed_seconds=time.monotonic()-started,time=time.time()))
        write(ROOT/'state/active.json',dict(status='IDLE',time=time.time()))
        print('END',name,child.returncode,status.get('status'),flush=True)
        if not keep_quiet and not owner.processes():owner.restore(list(range(8)))
        return ok


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--restore',action='store_true')
    a=p.parse_args()
    raise SystemExit(0 if execute(a.name,not a.restore) else 1)
