"""Keep heavy CPU trace analysis outside isolated GPU measurement intervals."""
import fcntl
import os
import time
from contextlib import contextmanager
from .common import ROOT, append


@contextmanager
def quiet_analysis(label):
    with (ROOT/'state/quiet_lane.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        begin = time.time()
        append(ROOT/'state/CPU_analysis_intervals.jsonl',
               dict(event='START', label=label, pid=os.getpid(), time=begin))
        try:
            yield
        finally:
            append(ROOT/'state/CPU_analysis_intervals.jsonl',
                   dict(event='END', label=label, pid=os.getpid(), time=time.time(), start=begin))
            fcntl.flock(handle, fcntl.LOCK_UN)
