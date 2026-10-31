"""Restore approved burners only after all owned campaign dispatchers stop."""
from .common import *
from .campaign import restore_if_quiet


def managed_jobs():
    modules={b'poc_flashvep.looped_pd_bottleneck_crossover.campaign',
        b'poc_flashvep.looped_pd_bottleneck_crossover.dispatch'}
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid():continue
            args=(path/'cmdline').read_bytes().split(b'\0')
            if modules.intersection(args):found.append(int(path.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return found


def main():
    write(ROOT/'state/idle_restore_watch.json',dict(pid=os.getpid(),time=time.time(),status='WAITING'))
    while managed_jobs():time.sleep(15)
    # A new dispatcher must become visible before it can lease GPUs.
    time.sleep(2)
    if managed_jobs():return main()
    restore_if_quiet()
    write(ROOT/'state/idle_restore_watch.json',dict(pid=os.getpid(),time=time.time(),status='FINISHED',
        scope='The helper never signals research workers or unrelated GPU processes.'))


if __name__=='__main__':main()
