"""Enforce the persisted GPU cutoff. Never signal unregistered user sessions."""
import signal
from .common import *


def main():
    setup()
    write(ROOT/'state/deadline_guard.json',dict(pid=os.getpid(),deadline=GPU_DEADLINE,status='ARMED',time=time.time()))
    while time.time()<GPU_DEADLINE:
        if (ROOT/'state/CAMPAIGN_CLOSED.json').exists():return
        time.sleep(min(30,max(.1,GPU_DEADLINE-time.time())))
    active=ROOT/'state/active.json'
    stopped=[]
    if active.exists():
        record=json.loads(active.read_text());pid=record.get('pid')
        if pid:
            proc=Path('/proc',str(pid))
            try:
                words=(proc/'cmdline').read_bytes().decode().split('\0')
                allowed=any('poc_flashvep.looped_pd_' in word for word in words)
                owned=proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid
                if allowed and owned:
                    os.killpg(pid,signal.SIGTERM);stopped.append(pid)
            except (FileNotFoundError,ProcessLookupError):pass
    write(ROOT/'state/GPU_BUDGET_EXHAUSTED.json',dict(time=time.time(),registered_sessions_stopped=stopped,
        action='No new GPU job is authorized. Preserve evidence and publish before the overall deadline.'))
    until=time.monotonic()+60
    while resources().processes() and time.monotonic()<until:time.sleep(2)
    if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
