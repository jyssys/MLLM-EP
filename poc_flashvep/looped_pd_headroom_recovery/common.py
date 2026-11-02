"""Keep this campaign separate from the completed atomic-prefill campaign."""
import hashlib
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path('/home/esjung/MLLM-EP-github')
ROOT = REPO / 'poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom'
PRIOR = REPO / 'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'
SOURCE = Path(__file__).parent
PYTHON = Path('/home/esjung/.venvs/looped-pd-abc/bin/python')
RLT = Path('/home/esjung/external/looped-pd-abc-rlt')
BRANCH = 'flashvep/looped-pd-kv-profile-headroom-recovery'


def setup():
    for name in ('state', 'configs', 'data', 'raw', 'logs', 'figures'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configure():
    """Redirect reused harness modules before their process-local imports."""
    from ..looped_pd_atomic_prefill import common as old
    old.ROOT = ROOT
    old.setup = setup
    # Keep the old source directory: its execution hooks are reused unchanged.
    old.redirect()
    return old


def environment(gpus):
    old = configure()
    env = old.environment(gpus)
    env.pop('ATOMIC_PD_WORKER', None)
    env.pop('BC_WORKER', None)
    env.pop('LTP_WORKER', None)
    env['HEADROOM_RECOVERY_WORKER'] = '1'
    env['PYTHONPATH'] = str(SOURCE / 'bootstrap') + os.pathsep + env['PYTHONPATH']
    return env


def resources():
    from ..ouro_budget_joint_cg import resources as owner
    owner.ROOT = ROOT
    owner.PRIOR = PRIOR
    owner.APPROVED_LOGS.update((ROOT / 'burner.log', PRIOR / 'burner.log'))
    return owner


def restore_if_idle():
    """Restore owned burners only while no experiment owns the quiet lane."""
    import fcntl
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        owner=resources()
        if owner.processes():return False
        owner.restore(list(range(8)))
        return True


def index_signature():
    # The real index remains untouched. Publication must use a temporary index.
    return hashlib.sha256(subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=REPO)).hexdigest()


def preserve():
    value = json.loads((ROOT / 'state/preservation.json').read_text())
    assert index_signature() == value['index_entries_SHA256'], 'Unrelated index changed. Do not publish automatically.'
    current=subprocess.check_output(['git', 'diff', '--binary'], cwd=RLT)
    assert hashlib.sha256(current).hexdigest()==value['private_diff_SHA256'], 'Private runtime changed'
