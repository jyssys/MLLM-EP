"""A new bounded campaign. Preserve all earlier evidence and the real index."""
import json
import hashlib
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ..looped_pd_autonomous_headroom import common as prior

REPO = prior.REPO
SOURCE = Path(__file__).parent
ROOT = REPO / 'poc_flashvep/reports/ouro_huginn_fluidtoken_dynamic_loop_split_serving'
PREVIOUS = REPO / 'poc_flashvep/reports/looped_pd_fluidtoken_vs_dynamic_loop_split'
PARENT_SOURCE = REPO / 'poc_flashvep/fluidtoken_loop_split'
BASE = 'a4c9e4a69386b0cd9b50d8b3e5be3af05aa0b751'
BRANCH = 'flashvep/full-dynamic-loop-split-serving'
START = datetime(2026, 11, 4, 14, 5, 34, tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE = START + 11 * 3600
GPU_DEADLINE = START + 10 * 3600
LONG_START_CUTOFF = START + 9.5 * 3600
write, digest = prior.write, prior.digest


def write_derived(path, value):
    """Keep the prior derived bytes before a changed analysis replaces them."""
    path = Path(path)
    if path.exists():
        previous = path.read_bytes()
        if json.loads(previous) == value:
            return
        sha = hashlib.sha256(previous).hexdigest()
        history = path.with_name(path.stem + '_history_' + sha + path.suffix)
        if history.exists():
            assert history.read_bytes() == previous
        else:
            history.write_bytes(previous)
    write(path, value)


def configure():
    for key in ('ROOT', 'BASE', 'BRANCH', 'START', 'DEADLINE', 'GPU_DEADLINE', 'LONG_START_CUTOFF'):
        setattr(prior, key, globals()[key])
    prior.resources().APPROVED_LOGS.add(PREVIOUS / 'burner.log')
    result = prior.activate()
    from ..fluidtoken_loop_split import common as parent
    for key in ('ROOT', 'SOURCE', 'BASE', 'BRANCH', 'START', 'DEADLINE', 'GPU_DEADLINE', 'LONG_START_CUTOFF'):
        setattr(parent, key, globals()[key])
    parent.PREVIOUS = REPO / 'poc_flashvep/reports/looped_pd_autonomous_prefill_compute_decode_headroom'
    parent.configure, parent.initialize = configure, initialize
    return result


def initialize():
    configure()
    prior.initialize()
    owner = prior.resources()
    verified = owner.burners()
    for path in (PREVIOUS / 'state').glob('burner_*.json'):
        value = json.loads(path.read_text())
        pid = value.get('supervisor')
        try:
            proc = Path('/proc', str(pid))
            words = (proc / 'cmdline').read_bytes().decode().split('\0')
            if (set(value.get('GPUs', [])) <= set(verified) and proc.stat().st_uid == os.getuid()
                    and os.getsid(pid) == pid and owner.SCRIPT in words):
                target = ROOT / 'state' / path.name
                if not target.exists():
                    write(target, dict(value, adopted_from=str(path)))
        except (FileNotFoundError, ProcessLookupError):
            pass
    write(ROOT / 'state/budget.json', dict(start_epoch=START, hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE, long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=11, timezone='Asia/Seoul',
        scope='New user-authorized campaign. Includes audit, implementation, GPU work, analysis, publication, and resource restoration.'))


def save(cfg):
    path = ROOT / 'configs' / (cfg['name'] + '.json')
    if path.exists():
        assert json.loads(path.read_text()) == cfg, 'Do not overwrite an immutable run configuration'
    else:
        write(path, cfg)


if __name__ == '__main__':
    initialize()
    print(json.dumps(dict(root=str(ROOT), branch=BRANCH, start=START, deadline=DEADLINE)))
