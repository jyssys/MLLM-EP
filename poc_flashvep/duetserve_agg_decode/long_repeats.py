"""Predeclare large-budget long-window replication before its results exist."""
import json
import subprocess
import time

from .common import PYTHON, REPO, ROOT, write


def main():
    write(ROOT / 'state/long_replication_predeclaration.json', dict(time=time.time(),
        seeds=[2026111632, 2026111633], QPS=[2., 6.], method='DP2_B8192_S1024',
        measurement_seconds=600, threshold=2.,
        scope='Run both independent seeds regardless of the first long-window ratio. Same execution and workload settings. No repeat-until-success or optional stopping at2x.'))
    until = time.monotonic() + 36000
    result = ROOT / 'state/long_windows_plan_result.json'
    while not result.exists():
        assert time.monotonic() < until
        time.sleep(5)
    assert json.loads(result.read_text())['returncode'] == 0
    command = [str(PYTHON), '-u', '-m', 'poc_flashvep.duetserve_agg_decode.long_window',
               '--methods', 'DP2_B8192_S1024', '--seeds', '2026111632', '2026111633']
    started = time.time()
    with (ROOT / 'logs/long_window_independent_repeats.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        write(ROOT / 'state/long_replication_running.json', dict(command=command, pid=child.pid, start_epoch=started))
        code = child.wait(14400)
    write(ROOT / 'state/long_replication_complete.json', dict(command=command, start_epoch=started,
        end_epoch=time.time(), returncode=code, passed=code == 0))
    assert code == 0


if __name__ == '__main__':
    main()
