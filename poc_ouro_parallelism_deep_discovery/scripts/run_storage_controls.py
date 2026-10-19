"""Bounded P4 physical-sharing counterfactuals. These are not candidate results."""
import datetime as dt
import os
import subprocess
import time
from poc_ouro_parallelism_deep_discovery.contract import *


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    target = ROOT / 'artifacts/storage_control_queue.json'
    assert not target.exists()
    state = dict(pid=os.getpid(), start_utc=stamp(), status='WAITING_FOR_OWN_PREREQUISITE',
                 waiting_is_substantive_research=False, jobs=[],
                 purpose='Physical sharing control. No new scheduler, new model, or quality-equivalent R1 claim.')
    write(target, state)
    prerequisite = ROOT / 'outputs/quality_recheck_base26_P1_v1/guard_status.json'
    deadline = time.monotonic() + 7200
    while not prerequisite.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(prerequisite))
        time.sleep(1)
    assert read(prerequisite).get('success'), prerequisite
    for size, arm in [('1.4B', 'alias'), ('1.4B', 'copies'), ('2.6B', 'copies'), ('2.6B', 'alias')]:
        short = size.replace('.', '').replace('B', '')
        name = f'storage_{arm}_base{short}_P4_v1'
        command = [REFERENCE_PYTHON, '-m', 'poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
                   '--gpus', '4,5,6,7', '--output', 'outputs/' + name, '--timeout', '3600', '--',
                   PYTHON, '-m', 'poc_ouro_parallelism_deep_discovery.scripts.native_baseline',
                   '--size', size, '--tp', '4', '--R', '4', '--async-scheduling',
                   '--weight-storage-control', arm]
        job = dict(name=name, command=command, start_utc=stamp(), status='RUNNING')
        state['jobs'].append(job)
        state['status'] = 'RUNNING'
        write(target, state)
        result = subprocess.run(command, cwd=PROJECT)
        job.update(exit_code=result.returncode, end_utc=stamp(), status='PASS' if result.returncode == 0 else 'FAIL')
        write(target, state)
        if result.returncode:
            state.update(status='STOPPED_AFTER_FAILURE', end_utc=stamp())
            write(target, state)
            raise SystemExit(result.returncode)
    state.update(status='COMPLETE', end_utc=stamp())
    write(target, state)


if __name__ == '__main__':
    main()
