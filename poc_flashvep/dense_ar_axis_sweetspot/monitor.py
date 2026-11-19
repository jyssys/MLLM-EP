"""Keep a resumable inventory and restore approved burners after controllers exit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,write,preserve,resources


def read(path,default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return default


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:
        return False


def snapshot():
    controllers={p.stem:dict(**read(p),alive=alive(read(p)['pid']))
        for p in (ROOT/'state').glob('*controller*.json') if isinstance(read(p),dict) and 'pid' in read(p)}
    receipts=[read(p) for p in (ROOT/'raw').glob('*/dispatcher_result.json')]
    return dict(time=time.time(),remaining_seconds=max(0,DEADLINE-time.time()),
        controllers=controllers,completed_dispatchers=sum(bool(r.get('complete')) for r in receipts),
        failed_dispatchers=sum(not r.get('complete') for r in receipts),
        initial=read(ROOT/'state/initial_progress_v2.json'),research=read(ROOT/'state/research_progress.json'),
        GPU_lease=read(ROOT/'state/GPU_lease_current.json'),
        initial_complete=(ROOT/'state/initial_complete.json').exists(),
        research_complete=(ROOT/'state/research_GPU_complete.json').exists(),
        diagnostics_complete=(ROOT/'state/diagnostic_complete.json').exists(),
        critical_stops={p.name:read(p) for p in (ROOT/'state').glob('*stopped*.json')
            if p.name not in ('initial_stopped.json',)},
        known_failure='state_Native_Last0_N8_B1_v1 has a preserved premeasurement configuration failure. Recovery uses immutable v2 names.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    previous=-1
    while True:
        preserve();value=snapshot()
        write(ROOT/'state/resume_progress_manifest.json',value)
        if value['completed_dispatchers']!=previous:
            import importlib
            from . import statistics as current_statistics
            importlib.reload(current_statistics).main();previous=value['completed_dispatchers']
            print('MONITOR',json.dumps({k:v for k,v in value.items() if k in
                ('time','completed_dispatchers','failed_dispatchers','GPU_lease','critical_stops')}),flush=True)
        active=any(v['alive'] and v['pid']!=os.getpid() for v in value['controllers'].values())
        if not active:
            with QUIET_LOCK.open('a+') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                owner=resources()
                if not owner.processes():
                    owner.restore(list(range(8)))
            write(ROOT/'state/all_controllers_returned.json',dict(time=time.time(),
                scope='Controllers exited. This is not a scientific success verdict or a completed publication. Verify all8 approved burner processes separately.'))
            return
        if args.once:
            return
        time.sleep(30)


if __name__=='__main__':
    main()
