"""Bounded independent screens after the owned context queue. No engine rewrite."""
import datetime as dt
import subprocess
import time

from poc_ouro_parallelism_deep_discovery.contract import ROOT, PROJECT, REFERENCE_PYTHON, PYTHON, read, write


def main():
    target = ROOT / 'artifacts/post_context_controls_queue_v1.json'
    if target.exists():
        raise FileExistsError(target)
    stamp = lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    state = dict(status='WAITING_FOR_OWNED_CONTEXT_QUEUE', started_utc=stamp(), jobs=[],
                 waiting_is_substantive_research=False, held_out='UNTOUCHED')
    write(target, state)
    queue = ROOT / 'artifacts/native_long_context_queue_v3.json'
    deadline = time.monotonic() + 7200
    while True:
        if queue.exists():
            status = read(queue)['status']
            if status == 'COMPLETE':
                break
            if status == 'STOPPED_AFTER_FAILURE':
                raise RuntimeError('The owned context queue failed. Inspect before another experiment.')
        if time.monotonic() > deadline:
            raise TimeoutError('The owned context queue did not finish.')
        time.sleep(1)
    fixtures = [
        ('c07_base14_tp4_fixture_v1', '4,5,6,7',
         ['--size', '1.4B', '--tp', '4', '--context', '1024']),
        ('c07_base14_tp1_row2_depth2_fixture_v1', '4',
         ['--size', '1.4B', '--tp', '1', '--context', '1024', '--row', '2', '--depth', '2'])]
    previous_fixture_passed = True
    for name, gpus, options in fixtures:
        if not previous_fixture_passed:
            state['jobs'].append(dict(name=name, status='NOT_RUN_AFTER_PREVIOUS_FIXTURE_FAILURE'))
            write(target, state)
            continue
        command = [REFERENCE_PYTHON, '-m', 'poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
            '--gpus', gpus, '--output', 'outputs/' + name, '--timeout', '1800', '--', PYTHON,
            '-m', 'poc_ouro_parallelism_deep_discovery.scripts.c07_one_pass_fixture', *options]
        row = dict(name=name, command=command, started_utc=stamp(), status='RUNNING')
        state['status'] = 'RUNNING'; state['jobs'].append(row); write(target, state)
        result = subprocess.run(command, cwd=PROJECT)
        previous_fixture_passed = result.returncode == 0
        row.update(ended_utc=stamp(), returncode=result.returncode,
                   status='COMPLETE' if previous_fixture_passed else 'FAILED_OR_UNRESOLVED')
        write(target, state)
    # These are separate native calibration baselines, not a rescue of a failed candidate.
    for size, short in [('1.4B', '14'), ('2.6B', '26')]:
        name = f'quality_native_async_base{short}_P2_v1'
        command = [REFERENCE_PYTHON, '-m', 'poc_ouro_parallelism_deep_discovery.scripts.gpu_session',
            '--gpus', '4,6', '--output', 'outputs/' + name, '--timeout', '3600', '--', PYTHON,
            '-m', 'poc_ouro_parallelism_deep_discovery.scripts.native_quality_pilot',
            '--size', size, '--tp', '2', '--async-scheduling', '--runtime-label', 'tp2baseline']
        row = dict(name=name, command=command, started_utc=stamp(), status='RUNNING')
        state['jobs'].append(row); write(target, state)
        result = subprocess.run(command, cwd=PROJECT)
        row.update(ended_utc=stamp(), returncode=result.returncode,
                   status='COMPLETE' if result.returncode == 0 else 'FAILED')
        write(target, state)
        if result.returncode:
            state.update(status='STOPPED_AFTER_NATIVE_FAILURE', ended_utc=stamp())
            write(target, state)
            raise SystemExit(result.returncode)
    state.update(status='COMPLETE', ended_utc=stamp(),
                 candidate_closed_loop='NOT_RUN', graph_fixture_headroom='UNVALIDATED')
    write(target, state)


if __name__ == '__main__':
    main()
