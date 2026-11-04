"""Reload one completed controller process without stopping a GPU worker."""
import subprocess
import time
from .common import *


def alive(pid, module):
    proc = Path('/proc', str(pid))
    try:
        words = (proc / 'cmdline').read_bytes().split(b'\0')
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        return proc.stat().st_uid == os.getuid() and module.encode() in words and state != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def main(pid, stage='static_campaign'):
    initialize()
    assert stage in ('static_campaign', 'primary_screen')
    module = 'poc_flashvep.dynamic_loop_split_serving.' + stage
    assert alive(pid, module), 'Only the named owned controller can be supervised'
    prefix = 'static' if stage == 'static_campaign' else 'primary'
    target = ROOT / 'state' / (prefix + '_controller_reload.json')
    write(target, dict(status='WAITING_FOR_ORIGINAL_CONTROLLER_EXIT', original_pid=pid,
        reason='The loaded analyzer predates the measured-only device-row filter. Reload only after the original controller exits.',
        scope='Do not signal or stop any GPU worker. Preserve completed namespaces and the original budget.'))
    while alive(pid, module) and time.time() < LONG_START_CUTOFF:
        time.sleep(5)
    status = json.loads((ROOT / 'state' / (stage + '.json')).read_text())
    terminal = 'STATIC_CAMPAIGN_FINISHED' if stage == 'static_campaign' else 'PRIMARY_SCREEN_FINISHED'
    if status['status'] == terminal or time.time() >= LONG_START_CUTOFF:
        write(target, dict(status='NO_RELOAD_NEEDED_OR_TIME_BUDGET', original_pid=pid))
        return
    active = json.loads((ROOT / 'state/active.json').read_text())
    if active.get('pid') and alive(active['pid'], 'poc_flashvep.looped_pd_headroom_recovery.worker'):
        # Another independent controller can own the quiet lane. Its worker
        # remains untouched. The replacement waits on the existing same lock.
        owner_note = 'Another registered job can continue under the same quiet-lane lock.'
    else:
        owner_note = 'No registered GPU worker is stopped by this reload.'
    log = ROOT / 'logs' / (prefix + '_controller_reloaded_v1.log')
    with log.open('x') as handle:
        command = [str(prior.PYTHON), '-m', module]
        child = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        write(target, dict(status='RELOADED_ONCE', original_pid=pid, replacement_pid=child.pid,
            command=command, log=str(log), time=time.time(), scope=owner_note,
            preserved_namespaces=True, budget_start_unchanged=START))
        child.wait()
        write(target, dict(status='REPLACEMENT_EXITED', original_pid=pid, replacement_pid=child.pid,
            returncode=child.returncode, time=time.time(), log=str(log),
            scope='A process exit is not a correctness or performance verdict. Read the stage ledgers. No automatic second retry.'))


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'static_campaign')
