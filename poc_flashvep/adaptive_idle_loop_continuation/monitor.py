"""Watch only registered non-hashing jobs for stalled response or final export."""
import os
import signal
import time
import fcntl
from .common import *


def restore_missing_burners(owner):
    """Keep verified existing burners. Do not share resources with another job."""
    occupied=owner.processes();owned=owner.burners()
    for row in occupied:
        try:
            if os.getsid(row['pid'])!=owned.get(row['gpu']):return False
        except ProcessLookupError:return False
    missing=[gpu for gpu in range(8) if gpu not in owned]
    if missing:owner.restore(missing)
    return bool(missing)


def main():
    configure();state={};path=ROOT/'state/active.json';last_restore=0.
    write(ROOT/'state/monitor_owner.json',dict(pid=os.getpid(),time=time.time(),deadline=GPU_DEADLINE))
    while time.time()<GPU_DEADLINE and not (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
        try:
            active=json.loads(path.read_text());name=active.get('name');pid=active.get('pid')
            if active.get('status')=='IDLE' and time.time()-active.get('time',time.time())>60 and time.time()-last_restore>60:
                last_restore=time.time()
                with (ROOT/'state/quiet_lane.lock').open('a+') as lane:
                    try:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:pass
                    else:
                        from ..looped_pd_autonomous_headroom.common import resources,append
                        owner=resources()
                        if restore_missing_burners(owner):
                            append(ROOT/'state/idle_burner_restorations.jsonl',dict(time=time.time(),
                                idle_since=active.get('time'),owned_burners=owner.burners(),
                                scope='Approved burners after a 60-second between-job grace. The next guarded experiment stops them before measurement.'))
            if name and pid:
                cfg_path=ROOT/'configs'/(name+'.json');cfg=json.loads(cfg_path.read_text())
                folder=ROOT/'raw'/name;proc=Path('/proc',str(pid))
                words=(proc/'cmdline').read_bytes().decode().split('\0')
                owned=proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid and str(cfg_path) in words
                signature=tuple((p.name,p.stat().st_size) for p in (folder/'requests.jsonl',folder/'after_measured.json') if p.exists())
                now=time.monotonic();previous=state.get(name)
                if previous is None or signature!=previous['signature']:
                    state[name]=dict(signature=signature,last_progress=now)
                stalled=now-state[name]['last_progress']
                intervals=cfg.get('arrival_offsets',[0.])
                bound=max(90.,max((b-a for a,b in zip(intervals,intervals[1:])),default=0.)+30.)
                eligible=(owned and cfg.get('worker')=='rlt_worker' and not cfg.get('state_probe')
                    and not cfg.get('verify_transfer') and (folder/'measurement_begin.json').exists() and signature)
                write(ROOT/'state/monitor_snapshot.json',dict(time=time.time(),name=name,pid=pid,owned=owned,
                    response_inactivity_s=stalled,limit_s=bound,eligible=bool(eligible)))
                if eligible and stalled>bound and not (folder/'idle_response_watchdog.json').exists():
                    for file in folder.glob('worker_stacks_*.log'):
                        worker=int(file.stem.rsplit('_',1)[1])
                        try:
                            if Path('/proc',str(worker)).stat().st_uid==os.getuid() and os.getsid(worker)==pid:
                                os.kill(worker,signal.SIGUSR1)
                        except (FileNotFoundError,ProcessLookupError):pass
                    write(folder/'idle_response_watchdog.json',dict(time=time.time(),name=name,session=pid,
                        inactivity_s=stalled,limit_s=bound,signature=signature,
                        reason='REGISTERED_NATIVE_RESPONSE_OR_EXPORT_STALL',
                        scope='Non-hashing measured jobs only. The dispatcher retains cleanup and failure recording.'))
                    os.killpg(pid,signal.SIGTERM)
        except (FileNotFoundError,ProcessLookupError,PermissionError,json.JSONDecodeError):pass
        time.sleep(5)
    write(ROOT/'state/monitor_finished.json',dict(time=time.time()))


if __name__=='__main__':main()
