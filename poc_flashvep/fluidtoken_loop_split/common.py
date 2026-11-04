"""Run a new campaign without changing prior evidence or the real Git index."""
import json
import os
import subprocess
from pathlib import Path
from ..looped_pd_autonomous_headroom import common as prior

REPO = prior.REPO
SOURCE = Path(__file__).parent
ROOT = REPO / 'poc_flashvep/reports/looped_pd_fluidtoken_vs_dynamic_loop_split'
PREVIOUS = prior.ROOT
BASE = 'ef3efe173ae48c00925ec616bdb5cc4afddf86a2'
BRANCH = 'flashvep/huginn-recovery-fluidtoken-loop-split'
START = 1793715523
DEADLINE = START + 11 * 3600
GPU_DEADLINE = START + 10 * 3600
LONG_START_CUTOFF = START + 9.5 * 3600
write, digest = prior.write, prior.digest


def configure():
    prior.ROOT = ROOT
    prior.BASE = BASE
    prior.BRANCH = BRANCH
    prior.START = START
    prior.DEADLINE = DEADLINE
    prior.GPU_DEADLINE = GPU_DEADLINE
    prior.LONG_START_CUTOFF = LONG_START_CUTOFF
    prior.resources().APPROVED_LOGS.add(PREVIOUS / 'burner.log')
    return prior.activate()


def initialize():
    configure()
    prior.initialize()
    # Adopt verified supervisors from the last campaign without editing it.
    owner = prior.resources()
    verified = owner.burners()
    for path in (PREVIOUS / 'state').glob('burner_*.json'):
        value = json.loads(path.read_text())
        pid = value.get('supervisor')
        try:
            proc = Path('/proc', str(pid))
            command = (proc / 'cmdline').read_bytes().decode().split('\0')
            if (set(value.get('GPUs', [])) <= set(verified)
                    and proc.stat().st_uid == os.getuid()
                    and os.getsid(pid) == pid and owner.SCRIPT in command):
                target = ROOT / 'state' / path.name
                if not target.exists():
                    write(target, dict(value, adopted_from=str(path)))
        except (FileNotFoundError, ProcessLookupError):
            pass
    write(ROOT / 'state/budget.json', dict(start_epoch=START,
        hard_deadline_epoch=DEADLINE, GPU_stop_epoch=GPU_DEADLINE,
        long_job_start_cutoff_epoch=LONG_START_CUTOFF, maximum_wall_hours=11,
        timezone='Asia/Seoul', scope='Includes audit, GPU work, reports, push, and resource restoration.'))


if __name__ == '__main__':
    initialize()
    print(json.dumps(dict(root=str(ROOT), branch=BRANCH, deadline=DEADLINE)))
