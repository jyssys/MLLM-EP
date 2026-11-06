"""Preserve the checkout and private runtime; use a separate publication ref."""
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ..native_split_deep_validation import common as prior

REPO = prior.REPO
SOURCE = Path(__file__).parent
ROOT = REPO / 'poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching'
PREVIOUS = prior.ROOT
BASE = '034b4ef9d2a8f0fe1a9d993370693d349cd03838'
BRANCH = 'flashvep/d-local-cross-depth-batching-poc'
START = datetime(2026, 11, 6, 8, 21, 36, tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE = START + 6 * 3600
GPU_DEADLINE = START + 5.25 * 3600
LONG_START_CUTOFF = GPU_DEADLINE - 20 * 60
write, digest = prior.write, prior.digest


def configure():
    for key in ('ROOT', 'SOURCE', 'BASE', 'BRANCH', 'START', 'DEADLINE', 'GPU_DEADLINE', 'LONG_START_CUTOFF'):
        setattr(prior, key, globals()[key])
    result = prior.configure()
    from ..looped_pd_autonomous_headroom.common import resources
    resources().APPROVED_LOGS.add(PREVIOUS / 'burner.log')
    return result


def initialize():
    configure()
    prior.initialize()
    from ..looped_pd_autonomous_headroom.common import resources
    owner = resources()
    verified = owner.burners()
    for path in (PREVIOUS / 'state').glob('burner_*.json'):
        value = json.loads(path.read_text())
        pid = value.get('supervisor')
        try:
            proc = Path('/proc', str(pid))
            words = (proc / 'cmdline').read_bytes().decode().split('\0')
            if (set(value.get('GPUs', [])) <= set(verified)
                    and proc.stat().st_uid == os.getuid() and os.getsid(pid) == pid
                    and owner.SCRIPT in words):
                target = ROOT / 'state' / path.name
                if not target.exists():
                    write(target, dict(value, adopted_from=str(path)))
        except (FileNotFoundError, ProcessLookupError):
            pass
    write(ROOT / 'state/budget.json', dict(start_epoch=START, hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE, long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=6, timezone='Asia/Seoul',
        scope='Audit, implementation, GPU tests, analysis, publication, and resource restoration share this fixed budget.'))


def save(cfg):
    path = ROOT / 'configs' / (cfg['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg, 'Immutable configuration differs'
    else:
        write(path, cfg)


if __name__ == '__main__':
    initialize()
    print(json.dumps(dict(root=str(ROOT), branch=BRANCH, deadline=DEADLINE)))
