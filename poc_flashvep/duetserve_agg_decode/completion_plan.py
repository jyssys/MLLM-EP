"""Serialize confirmation and long controls after diagnostic collectors finish."""
import fcntl
import json
import subprocess
import time

from .common import PYTHON, QUIET_LOCK, REPO, ROOT, initialize, resources, write


def main():
    initialize()
    deadline = time.monotonic() + 3600
    prerequisites = [ROOT / 'state/budget_coverage_phase_complete.json',
                     ROOT / 'analysis/TP2_numerical_diagnosis.json']
    while not all(path.exists() for path in prerequisites):
        assert time.monotonic() < deadline, 'Diagnostic prerequisite did not finish'
        time.sleep(5)
    commands = [
        ('fresh_confirmation', ['confirmation', 'evaluate'], 14400),
        ('long_windows', ['long_window'], 21600),
        ('secondary', ['campaign', 'secondary'], 7200),
    ]
    try:
        for phase, args, timeout in commands:
            result = ROOT / 'state' / (phase + '_plan_result.json')
            if result.exists():
                assert json.loads(result.read_text())['returncode'] == 0
                continue
            cmd = [str(PYTHON), '-u', '-m', 'poc_flashvep.duetserve_agg_decode.' + args[0], *args[1:]]
            started = time.time()
            with (ROOT / 'logs' / ('completion_' + phase + '.log')).open('a') as stream:
                child = subprocess.Popen(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                write(ROOT / 'state/completion_plan_active.json', dict(phase=phase,
                    command=cmd, pid=child.pid, start_epoch=started))
                code = child.wait(timeout)
            write(result, dict(command=cmd, start_epoch=started, end_epoch=time.time(), returncode=code))
            assert code == 0, phase
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
