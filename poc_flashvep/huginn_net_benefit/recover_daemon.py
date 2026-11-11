"""Recover a completed qualification without repeating its GPU work."""
from .common import *


def main():
    initialize()
    name = 'connections5_sm16_zero_state8'
    folder = ROOT/'raw'/name
    terminal = json.loads((folder/'dispatcher_result.json').read_text())
    assert terminal['complete']
    exits = [json.loads(path.read_text()) for path in folder.glob('native_worker_exit_*.json')]
    assert any(record['passed'] and {row['role'] for row in record['records']
        if row['exitcode'] == 0 and not row['forced_cleanup']} == {'prefill', 'decode'}
        for record in exits)
    qualification = analysis().qualify_frozen_reference(name)
    assert qualification['passed']
    qualification['campaign_gate_passed'] = True
    qualification['recovered_after_daemon_restart'] = True
    write(folder/'qualification_with_frozen_reference.json', qualification)
    write(ROOT/'state/daemon_restart_recovery.json', dict(time=time.time(),
        interrupted_controller_pid=1648649, interrupted_waiter_pid=1656428,
        recovered_exactness_namespace=name, worker_exit_codes=[0,0],
        preserved_launcher_returncode=120,
        observation='The API server restarted. The completed cell retained exact outputs, states, and zero worker exits. The launcher recorded exit code 120. The controller and waiting process then disappeared. All eight approved Burners were resident.',
        failure_attribution='A lost output pipe is the current infrastructure hypothesis. The complete failing launcher stack is NOT_RECORDED. No unsuccessful GPU worker exit appears in this recovered cell.',
        action='Do not repeat completed GPU work. Use a file-backed controller for remaining cells. Preserve the original terminal receipts and failed flag-check namespace.'))
    print('COMPLETED_GPU_QUALIFICATION_RECOVERED')


if __name__ == '__main__':
    main()
