"""Stop only the registered namespace session at the fixed GPU deadline."""
import os
import signal
import time
from pathlib import Path
from .common import ROOT, GPU_DEADLINE, configure, write


def main():
    configure()
    write(ROOT/'state/deadline_guard.json',dict(pid=os.getpid(),deadline=GPU_DEADLINE,status='ARMED',time=time.time()))
    while time.time()<GPU_DEADLINE:
        if (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
            return
        time.sleep(min(30,max(.1,GPU_DEADLINE-time.time())))
    import json
    stopped=[]
    path=ROOT/'state/active.json'
    if path.exists():
        record=json.loads(path.read_text());pid=record.get('pid')
        if pid:
            try:
                proc=Path('/proc',str(pid));words=(proc/'cmdline').read_bytes().decode().split('\0')
                expected=str(ROOT/'configs'/(record['name']+'.json'))
                module=any(w.startswith(('poc_flashvep.d_local_cross_depth_batching.',
                    'poc_flashvep.looped_pd_headroom_recovery.')) for w in words)
                if expected in words and module and proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid:
                    os.killpg(pid,signal.SIGTERM);stopped.append(pid)
            except (FileNotFoundError,ProcessLookupError):
                pass
    write(ROOT/'state/GPU_BUDGET_EXHAUSTED.json',dict(time=time.time(),registered_sessions_stopped=stopped))
    from ..looped_pd_autonomous_headroom.common import resources
    until=time.monotonic()+60
    while resources().processes() and time.monotonic()<until:
        time.sleep(2)
    if not resources().processes():
        resources().restore(list(range(8)))

if __name__ == '__main__':
    main()
