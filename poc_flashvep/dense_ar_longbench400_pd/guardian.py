"""Hold the global experiment lease while independent pair leases execute."""
import fcntl
import json
import os
import signal
import time
from .common import ROOT,QUIET_LOCK,initialize,preserve,write,time_available


def main():
    initialize();stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);preserve()
        write(ROOT/'state/guardian.json',dict(pid=os.getpid(),status='ACTIVE',time=time.time(),global_lock=str(QUIET_LOCK)))
        while not stopped[0] and time_available() and not (ROOT/'state/STOP_GUARDIAN.json').exists():time.sleep(2)
        while True:
            active=[]
            for path in (ROOT/'state').glob('pair_lease_*.json'):
                row=json.loads(path.read_text())
                if row.get('status')=='RUNNING':active.append(row)
            if not active:break
            write(ROOT/'state/guardian.json',dict(pid=os.getpid(),status='DRAINING',time=time.time(),
                active_pairs=active,scope='Keep the global lease until owned pair jobs release their resources.'))
            time.sleep(2)
        preserve();write(ROOT/'state/guardian.json',dict(pid=os.getpid(),status='RELEASED',time=time.time()))


if __name__=='__main__':main()
