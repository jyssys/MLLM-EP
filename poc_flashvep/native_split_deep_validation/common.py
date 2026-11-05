"""Keep the new budget, evidence, branch, and process-local hooks separate."""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ..native_loop_2d_oracle import common as prior

REPO = prior.REPO
SOURCE = Path(__file__).parent
ROOT = REPO / 'poc_flashvep/reports/ouro_huginn_loop_specific_native_split_deep_validation'
PREVIOUS = REPO / 'poc_flashvep/reports/ouro_huginn_native_loop_continuation_2d_oracle'
BASE = 'a1c9845ba4f7a56b4e990b16e4f25f96311311de'
BRANCH = 'flashvep/loop-specific-native-split-deep-validation'
START = datetime(2026, 11, 5, 12, 38, 46, tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE = START + 11 * 3600
GPU_DEADLINE = START + 9.75 * 3600
LONG_START_CUTOFF = GPU_DEADLINE - 20 * 60
write, digest = prior.write, prior.digest


def configure():
    for key in ('ROOT', 'SOURCE', 'BASE', 'BRANCH', 'START', 'DEADLINE', 'GPU_DEADLINE', 'LONG_START_CUTOFF'):
        setattr(prior, key, globals()[key])
    prior.PREVIOUS = PREVIOUS
    return prior.configure()


def initialize():
    configure()
    prior.initialize()
    write(ROOT / 'state/budget.json', dict(start_epoch=START, hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE, long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=11, timezone='Asia/Seoul',
        scope='New user-authorized campaign. Audit, implementation, GPU jobs, analysis, publication, and burner restoration share this fixed budget.'))


def save(cfg):
    path = ROOT / 'configs' / (cfg['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg, 'Immutable configuration differs'
    else:
        write(path, cfg)


if __name__ == '__main__':
    initialize()
    print(json.dumps(dict(root=str(ROOT), branch=BRANCH,
        GPU_stop=datetime.fromtimestamp(GPU_DEADLINE, ZoneInfo('Asia/Seoul')).isoformat(),
        deadline=datetime.fromtimestamp(DEADLINE, ZoneInfo('Asia/Seoul')).isoformat())))
