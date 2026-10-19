"""Sequential guarded native context controls. Idle waiting is not research time."""
import datetime as dt
import argparse
import subprocess
import time

from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.resource_release import release_complete


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    target = ROOT / f'artifacts/native_long_context_queue_v{args.attempt}.json'
    if target.exists():
        raise FileExistsError(target)
    state = dict(status='WAITING_FOR_OWNED_PREREQUISITE', started_utc=stamp(), jobs=[],
                 waiting_is_substantive_research=False)
    write(target, state)
    prerequisites = [ROOT / 'outputs/quality_native_async_base26_P4_v1']
    if args.attempt == 3:
        prerequisites.append(ROOT / 'outputs/native_graph_argument_inventory_base14_P1_v1')
    for prerequisite in prerequisites:
        deadline = time.monotonic() + 7200
        while not release_complete(prerequisite):
            status = prerequisite / 'guard_status.json'
            if status.exists() and not read(status).get('success'):
                raise RuntimeError(f'Owned prerequisite failed: {prerequisite.name}')
            if time.monotonic() > deadline:
                raise TimeoutError(f'Owned job did not release its devices: {prerequisite.name}')
            time.sleep(1)
        assert release_complete(prerequisite)
    for size in MODELS:
        for tp, gpus in [(1, '4'), (2, '4,6'), (4, '4,5,6,7')]:
            short = size.replace('.', '').replace('B', '')
            name = f'native_longctx_base{short}_P{tp}_v{args.attempt}'
            assert not (ROOT / 'outputs' / name).exists()
            command = [REFERENCE_PYTHON, '-m', 'poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
                '--gpus', gpus, '--output', 'outputs/' + name, '--timeout', '3600', '--', PYTHON,
                '-m', 'poc_ouro_parallelism_deep_discovery.scripts.native_long_context_control',
                '--size', size, '--tp', str(tp)]
            job = dict(name=name, size=size, TP=tp, physical_gpus=list(map(int, gpus.split(','))),
                       command=command, started_utc=stamp(), status='RUNNING')
            state['jobs'].append(job)
            state['status'] = 'RUNNING'
            write(target, state)
            result = subprocess.run(command, cwd=PROJECT)
            job.update(ended_utc=stamp(), returncode=result.returncode,
                       status='COMPLETE' if result.returncode == 0 else 'FAILED')
            write(target, state)
            if result.returncode:
                state.update(status='STOPPED_AFTER_FAILURE', ended_utc=stamp())
                write(target, state)
                raise SystemExit(result.returncode)
    state.update(status='COMPLETE', ended_utc=stamp())
    write(target, state)


if __name__ == '__main__':
    main()
