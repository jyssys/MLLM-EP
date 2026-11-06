"""Finish the active GPU job, then serialize research controllers by priority."""
import fcntl
import os
import signal
import time
from .common import *

CONTROLLERS={'slack_confirmation','collector_diagnosis','collector_force_trials','submission_event_trials',
    'final_flush_confirmation','native_repeat_controls','mixed_qualification','quantum_calibration','drained_flush_trials'}


def controllers():
    values=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid!=os.getuid() or '-m' not in words:continue
            module=words[words.index('-m')+1]
            prefix='poc_flashvep.adaptive_idle_loop_continuation.'
            if module.startswith(prefix) and module[len(prefix):] in CONTROLLERS:
                values.append(dict(pid=int(proc.name),module=module))
        except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeError,IndexError):pass
    return values


def active_parent():
    try:
        value=json.loads((ROOT/'state/active.json').read_text());pid=value.get('pid')
        if not pid:return None
        for line in Path('/proc',str(pid),'status').read_text().splitlines():
            if line.startswith('PPid:'):return int(line.split()[1])
    except (FileNotFoundError,ProcessLookupError,json.JSONDecodeError):pass
    return None


def main():
    initialize();stopped=[];before=controllers()
    # A waiting controller has no GPU lease. Never terminate the active parent.
    for row in before:
        if row['pid']==active_parent():continue
        try:os.kill(row['pid'],signal.SIGTERM);stopped.append(row)
        except ProcessLookupError:pass
    with (ROOT/'state/quiet_lane.lock').open('a+') as lane:
        fcntl.flock(lane,fcntl.LOCK_EX)
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources();assert not owner.processes(),'Do not interrupt a live GPU owner'
        for row in controllers():
            try:os.kill(row['pid'],signal.SIGTERM);stopped.append(row)
            except ProcessLookupError:pass
        write(ROOT/'state/orchestration_takeover.json',dict(time=time.time(),controllers_before=before,
            stopped_queued_controllers=stopped,active_GPU_jobs_aborted=0,
            reason='Give slack measurement and CPU diagnosis priority over queued hashing controls.',
            scope='The active GPU job returned before final controller shutdown. Immutable receipts and cached qualification results remain available for idempotent continuation. No experimental failure is reclassified.'))
    print(json.dumps(dict(queued_controllers_stopped=len(stopped),GPU_jobs_aborted=0)))


if __name__=='__main__':main()
