"""Move the user-priority campaign at a completed GPU-job boundary."""
import fcntl
import json
import os
import signal
import time
from pathlib import Path
from .common import ROOT, QUIET_LOCK, preserve, resources, write
from .monitor import snapshot


LEGACY_CONTROLLERS = ('campaign', 'confirmation', 'baseline_change', 'monitor')


def targets():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for module in LEGACY_CONTROLLERS:
            expected = ('poc_flashvep.dense_ar_matched_axis_pd.'+module).encode()
            if expected in args:
                assert (path/'status').read_text().split('Uid:\t', 1)[1].split()[0] == str(os.getuid())
                found.append(dict(pid=int(path.name), module=module,
                                  command=[a.decode() for a in args if a]))
    return found


def main():
    path = ROOT/'state/user_batch_matched_scope_revision_20261118.json'
    if path.exists():
        assert json.loads(path.read_text())['status'] == 'TRANSITION_COMPLETE'
        print('BATCH_MATCHED_SCOPE_ALREADY_ACTIVE', flush=True)
        return
    print('WAIT_FOR_EXISTING_GPU_JOB_BOUNDARY', flush=True)
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        preserve()
        before = snapshot()
        assert not before['critical_failures'], before
        assert (before.get('GPU_lease') or {}).get('status') != 'GPU_JOB_RUNNING', before
        owner = resources()
        approved = owner.burners()
        actual = owner.processes()
        assert all(os.getsid(row['pid']) == approved.get(row['gpu']) for row in actual), actual
        stopped = targets()
        # The exclusive quiet lease prevents any controller from starting
        # another GPU job. Every active serving job already completed.
        for row in stopped:
            os.kill(row['pid'], signal.SIGTERM)
        until = time.monotonic()+10
        remaining = stopped
        while remaining and time.monotonic() < until:
            time.sleep(.1)
            remaining = [row for row in stopped if (Path('/proc')/str(row['pid'])).exists()]
        assert not remaining, remaining
        preserve()
        write(path, dict(epoch=time.time(), status='TRANSITION_COMPLETE',
            reason='The user prioritizes actual matched batches and an online small-cut versus large-cut crossover test. Stop only idle campaign controllers after the active GPU job completes.',
            stopped_idle_controllers=stopped, GPU_jobs_terminated=0,
            before=before, unfinished_original_matrices='DEFERRED_BY_USER_SCOPE_REVISION',
            original_manifests_preserved=True, frozen_inference_unchanged=True,
            original_SLO_unchanged=True, Last2_full_sweep_remains_cancelled=True,
            comparison_scope='A matched-batch causal control is not an optimized unrestricted Token or Layer baseline. A crossover is a hypothesis, not a required result.'))
        owner.restore(list(range(8)))
    print('BATCH_MATCHED_SCOPE_TRANSITION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
