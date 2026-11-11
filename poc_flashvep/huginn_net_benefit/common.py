"""Route reused execution into an independent evidence namespace."""
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from ..completion_aware_loop_pd import common as previous

REPO = previous.REPO
SOURCE = Path(__file__).parent
PRIOR_ROOT = previous.ROOT
ROOT = REPO / 'poc_flashvep/reports/huginn_autonomous_adaptive_loop_pd_net_benefit'
BASE = '6cadf1011a0b596efe39ccccefc9898dd8ef2925'
BRANCH = 'flashvep/huginn-adaptive-loop-net-benefit'
START = datetime(2026, 11, 10, 13, 25, 48, tzinfo=timezone.utc).timestamp()
# Review at ten hours. This safety bound is not a requirement to consume time.
DEADLINE = START + 16 * 3600
GPU_DEADLINE = DEADLINE - 90 * 60
LONG_START_CUTOFF = GPU_DEADLINE - 30 * 60
write, digest = previous.write, previous.digest
KEYS = ('ROOT', 'SOURCE', 'BASE', 'BRANCH', 'START', 'DEADLINE',
        'GPU_DEADLINE', 'LONG_START_CUTOFF')


def configure():
    for key in KEYS:
        setattr(previous, key, globals()[key])
    result = previous.configure()
    from ..looped_pd_autonomous_headroom.common import resources
    resources().APPROVED_LOGS.add(PRIOR_ROOT / 'burner.log')
    return result


def initialize():
    configure()
    from ..looped_pd_autonomous_headroom import common as base
    base.initialize()
    write(ROOT / 'state/budget.json', dict(start_epoch=START,
        review_epoch=START + 10 * 3600, hard_safety_stop_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE, long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        scope='New autonomous campaign. Ten-hour review is not a stop. Stop earlier only after a supported decision. Extend the safety bound explicitly if useful.'))


def save(cfg):
    path = ROOT / 'configs' / (cfg['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg, 'Immutable configuration changed'
    else:
        write(path, cfg)


def analysis():
    configure()
    from ..completion_aware_loop_pd import analyze as parent
    parent.ROOT = ROOT
    parent.configure = configure
    from . import analysis_view
    return analysis_view


if __name__ == '__main__':
    initialize()
    print(json.dumps(dict(root=str(ROOT), branch=BRANCH, start=START)))
