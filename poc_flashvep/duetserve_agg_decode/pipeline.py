"""Run only gated characterization phases and preserve phase failures."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time

from .common import PYTHON, QUIET_LOCK, REPO, ROOT, initialize, resources, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after', default='state')
    parser.add_argument('phases', nargs='*', default=['concurrency', 'paper_dp2', 'native', 'budget', 'tp2_qualification', 'tp2'])
    args = parser.parse_args()
    initialize()
    if args.after != 'none':
        ready = ROOT / 'state' / (args.after + '_phase_complete.json')
        until = time.monotonic() + 2400
        while not ready.exists():
            assert time.monotonic() < until, 'Predecessor phase did not complete'
            time.sleep(5)
    child = None
    try:
        for phase in args.phases:
            command = [str(PYTHON), '-u', '-m', 'poc_flashvep.duetserve_agg_decode.campaign', phase]
            print('PHASE_START', phase, flush=True)
            started = time.time()
            with (ROOT / 'logs' / ('pipeline_' + phase + '.log')).open('a') as stream:
                child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                write(ROOT / 'state/pipeline_active.json', dict(phase=phase, time=started,
                    command=command, pid=child.pid, status='RUNNING'))
                try:
                    code = child.wait(7200)
                except subprocess.TimeoutExpired:
                    child.send_signal(signal.SIGTERM)
                    try:
                        child.wait(45)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(20)
                    raise
            write(ROOT / 'state' / ('pipeline_' + phase + '_result.json'), dict(phase=phase,
                start_epoch=started, end_epoch=time.time(), returncode=code, command=command,
                passed=code == 0, scope='A failed phase stops the dependent pipeline. Inspect preserved evidence before selecting a new experiment.'))
            print('PHASE_END', phase, code, flush=True)
            assert code == 0, 'Phase failed: ' + phase
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(45)
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
