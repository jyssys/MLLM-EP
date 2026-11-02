"""Hold the same quiet-lane lock while a human or agent edits existing source."""
import fcntl
import os
import sys
import time
from .common import ROOT,append


def main():
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        print('WAITING_FOR_EXPERIMENT_BOUNDARY',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        append(ROOT/'state/source_edit_lock.jsonl',dict(event='ACQUIRED',pid=os.getpid(),time=time.time()))
        print('SOURCE_EDIT_LOCK_HELD',flush=True)
        sys.stdin.readline()
        append(ROOT/'state/source_edit_lock.jsonl',dict(event='RELEASED',pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
