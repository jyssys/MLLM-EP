"""Enforce this campaign's fixed GPU stop through owned-session checks."""
import os
import signal
import time
from .common import *


def main():
    configure();write(ROOT/'state/deadline_guard.json',dict(pid=os.getpid(),deadline=GPU_DEADLINE,time=time.time()))
    while time.time()<GPU_DEADLINE:
        if (ROOT/'state/CAMPAIGN_CLOSED.json').exists():return
        time.sleep(min(30,max(.1,GPU_DEADLINE-time.time())))
    stopped=[]
    path=ROOT/'state/active.json'
    if path.exists():
        value=json.loads(path.read_text());pid=value.get('pid')
        if pid:
            try:
                proc=Path('/proc',str(pid));words=(proc/'cmdline').read_bytes().decode().split('\0')
                expected=str(ROOT/'configs'/(value['name']+'.json'))
                if expected in words and proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid:
                    os.killpg(pid,signal.SIGTERM);stopped.append(pid)
            except (FileNotFoundError,ProcessLookupError):pass
    write(ROOT/'state/GPU_BUDGET_EXHAUSTED.json',dict(time=time.time(),registered_sessions_stopped=stopped))


if __name__=='__main__':main()
