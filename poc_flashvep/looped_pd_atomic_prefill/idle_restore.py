"""Restore approved idle burners without racing an experiment lease."""
import fcntl
from .common import *


def managed_jobs():
    names={b'poc_flashvep.looped_pd_atomic_prefill.dispatch',b'poc_flashvep.looped_pd_atomic_prefill.campaign'}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid==os.getuid() and names.intersection((path/'cmdline').read_bytes().split(b'\0')):return True
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return False


def main():
    from .resources import inherited as resources
    setup();write(ROOT/'state/idle_restore_watch.json',dict(pid=os.getpid(),status='WAITING',time=time.time()))
    while not (ROOT/'state/CLOSED.json').exists():
        if managed_jobs():time.sleep(10);continue
        time.sleep(15)
        with (ROOT/'state/gpu_lease.lock').open('a+') as lease:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if not managed_jobs() and not resources.processes():
                resources.restore(list(range(8)))
                write(ROOT/'state/idle_restore_watch.json',dict(pid=os.getpid(),status='RESTORED_IDLE_BURNERS',time=time.time()))
        time.sleep(10)


if __name__=='__main__':main()
