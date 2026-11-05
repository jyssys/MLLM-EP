"""Close idle experiment owners and restore only approved burner sessions."""
import argparse
import signal
import subprocess
from .common import *


def command(pid):
    try:
        return Path('/proc', str(pid), 'cmdline').read_bytes().decode().replace('\0', ' ').strip()
    except FileNotFoundError:
        return ''


def main(controller, guard):
    initialize()
    assert not command(controller), 'Do not close a running controller or a reused PID'
    active = json.loads((ROOT / 'state/active.json').read_text())
    assert active.get('status') == 'IDLE', 'The experiment still owns an active job'
    owner = prior.parent.prior.resources()
    owned = owner.burners()
    for row in owner.processes():
        assert Path('/proc', str(row['pid'])).stat().st_uid == os.getuid()
        assert os.getsid(row['pid']) == owned.get(row['gpu']), 'Leave an unrelated GPU process untouched'
    guard_command = command(guard)
    if guard_command:
        assert Path('/proc', str(guard)).stat().st_uid == os.getuid()
        assert 'poc_flashvep.native_split_deep_validation.deadline_guard' in guard_command
        os.kill(guard, signal.SIGTERM)
        until = time.monotonic() + 5
        while command(guard) and time.monotonic() < until:
            time.sleep(.1)
        assert not command(guard), 'The deadline guard did not retire'
    from ..looped_pd_headroom_recovery.common import restore_if_idle
    restore_if_idle()
    until = time.monotonic() + 30
    while time.monotonic() < until:
        owned, processes = owner.burners(), owner.processes()
        if set(owned) == set(range(8)) and {row['gpu'] for row in processes} == set(range(8)):
            break
        time.sleep(.5)
    assert set(owned) == set(range(8)), 'Approved burner workers are missing'
    assert {row['gpu'] for row in processes} == set(range(8)), 'A restored GPU has no compute process'
    for row in processes:
        assert Path('/proc', str(row['pid'])).stat().st_uid == os.getuid()
        assert os.getsid(row['pid']) == owned[row['gpu']], 'The GPU process does not belong to its approved burner session'
    path = ROOT / 'state/serving_campaign.json'
    ledger = json.loads(path.read_text())
    original = ROOT / 'state/serving_campaign_before_closeout.json'
    if not original.exists():
        write(original, ledger)
    completed = [json.loads(path.read_text())['time']
                 for path in (ROOT / 'raw').glob('*/dispatcher_result.json')]
    receipt = dict(time=time.time(), status='GPU_MEASUREMENTS_CLOSED',
        last_dispatch_result_epoch=max(completed), controller_PID=controller,
        controller_running=False, deadline_guard_PID=guard, deadline_guard_running=False,
        approved_burner_workers=owned, GPU_compute_processes=processes,
        GPU_status=subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,memory.used,utilization.gpu', '--format=csv'], text=True),
        scope='Scientific GPU jobs ended. Approved burners continue on GPUs 0 through 7. Analysis and publication remain separate.')
    write(ROOT / 'state/gpu_campaign_closed.json', receipt)
    ledger.update(prior_progress_status=ledger['status'], status='GPU_MEASUREMENTS_CLOSED',
                  GPU_close_epoch=receipt['time'])
    write(path, ledger)
    prior.parent.prior.preserve()
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=int, required=True)
    parser.add_argument('--guard', type=int, required=True)
    args = parser.parse_args()
    main(args.controller, args.guard)
