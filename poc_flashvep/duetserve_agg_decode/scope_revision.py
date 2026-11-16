"""Apply an explicit user scope change without deleting previous measurements."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from .common import QUIET_LOCK, ROOT, digest, preserve, resources, write


CONTROLLERS = {
    683118: 'duetserve_agg_decode.completion_plan',
    733795: 'duetserve_agg_decode.confirmation',
    727064: 'duetserve_agg_decode.long_repeats',
    734339: 'duetserve_agg_decode.monitor',
    843979: 'duetserve_agg_decode.prepare_review',
}


def terminate(pid, expected):
    proc = Path(f'/proc/{pid}')
    if not proc.exists():
        return
    assert proc.stat().st_uid == os.getuid()
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert expected in command and os.getsid(pid) == pid, (pid, command)
    os.kill(pid, signal.SIGTERM)


def main():
    path = ROOT / 'state/user_primary_scope_revision_v2.json'
    assert not path.exists()
    requested = time.time()
    lease_state = json.loads((ROOT / 'state/GPU_lease_current.json').read_text())
    interrupted = None
    if lease_state.get('status') == 'GPU_JOB_RUNNING':
        cfg = json.loads((ROOT / 'configs' / (lease_state['name'] + '.json')).read_text())
        if cfg['arm'] == 'DP2' and cfg['token_budget'] == 512:
            interrupted = lease_state['name']
            terminate(733795, CONTROLLERS[733795])
    # A Native or large-budget job completes normally before this lease is
    # obtained. No new server can start while the controller handoff occurs.
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        approved = owner.burners()
        assert all(os.getsid(r['pid']) == approved.get(r['gpu']) for r in owner.processes())
        for pid, expected in CONTROLLERS.items():
            terminate(pid, expected)
        preserve()
        completed = [p.parent.name for p in (ROOT / 'raw').glob('*/dispatcher_result.json')
                     if json.loads(p.read_text())['complete']]
        write(path, dict(requested_epoch=requested, applied_epoch=time.time(),
            reason='EXPLICIT_USER_REQUEST: Use B8192/S1024 as the Primary Aggregated setting. Stop further B512/S16 GPU experiments. Preserve existing results.',
            primary_methods=['Agg-DP2 B8192/S1024', 'Native PD 1P1D'],
            model_precision_prompt_arrivals_SLO_unchanged=True,
            original_freeze_SHA256=digest(ROOT / 'state/confirmation_freeze.json'),
            preserved_completed_starts=completed, interrupted_small_budget_run=interrupted,
            primary_confirmation_planned=54, long_window_planned=8, secondary_planned=12,
            prohibited_future_GPU_family='Agg-DP2 B512/S16',
            scope='A prospective user-directed scope revision, not a performance-driven deletion or a new claim of global baseline optimality. Old configurations, receipts, and figures remain intact.'))
        if not owner.processes():
            owner.restore(list(range(8)))
        print('USER_SCOPE_REVISION_APPLIED', len(completed), flush=True)


if __name__ == '__main__':
    main()
