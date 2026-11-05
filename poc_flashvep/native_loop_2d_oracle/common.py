"""Reuse the qualified guard without changing earlier artifacts or the real index."""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ..dynamic_loop_split_serving import common as parent

REPO = parent.REPO
SOURCE = Path(__file__).parent
ROOT = REPO / 'poc_flashvep/reports/ouro_huginn_native_loop_continuation_2d_oracle'
PREVIOUS = parent.ROOT
BASE = '5a4ff4a8a3f13119951b4e2ff202fb3541ac1d0a'
BRANCH = 'flashvep/native-loop-continuation-2d-oracle'
START = datetime(2026, 11, 5, 0, 47, 27, tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE = START + 6 * 3600
GPU_DEADLINE = START + 5 * 3600
LONG_START_CUTOFF = GPU_DEADLINE - 180
write, digest = parent.write, parent.digest


def configure():
    for key in ('ROOT', 'SOURCE', 'BASE', 'BRANCH', 'START', 'DEADLINE', 'GPU_DEADLINE', 'LONG_START_CUTOFF'):
        setattr(parent, key, globals()[key])
    result = parent.configure()
    parent.prior.resources().APPROVED_LOGS.add(PREVIOUS / 'burner.log')
    return result


def initialize():
    configure()
    parent.prior.initialize()
    owner = parent.prior.resources()
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
        GPU_stop_epoch=GPU_DEADLINE, maximum_wall_hours=6, timezone='Asia/Seoul',
        scope='Audit, GPU experiments, analysis, publication, and burner restoration. The clock does not restart after a pause.'))


def save(cfg):
    path = ROOT / 'configs' / (cfg['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg, 'Immutable configuration differs'
    else:
        write(path, cfg)


if __name__ == '__main__':
    initialize()
    print(json.dumps(dict(root=str(ROOT), branch=BRANCH, deadline=DEADLINE)))
