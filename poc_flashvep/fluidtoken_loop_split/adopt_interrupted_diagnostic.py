"""Guard an identified diagnostic after its controller was interrupted."""
import fcntl
import signal
import time
from .common import *


def main():
    configure()
    pid = 1626595
    name = 'static_gate_v3_huginn_atomic_native8'
    folder = ROOT / 'raw' / name
    process = Path('/proc', str(pid))
    command = (process / 'cmdline').read_bytes().split(b'\0')
    assert process.stat().st_uid == os.getuid() and os.getsid(pid) == pid
    assert str(ROOT / 'configs' / (name + '.json')).encode() in command
    with (ROOT / 'state/quiet_lane.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write(folder / 'controller_adoption.json', dict(pid=pid, time=time.time(),
            reason='The controller acquired the lane and launched this diagnostic before its cancellation reached it.',
            action='Retain the live owned worker and restore exclusive monitoring. Do not infer its unavailable parent exit status.'))
        deadline = time.monotonic() + 900
        while process.exists():
            if time.monotonic() >= deadline:
                assert process.stat().st_uid == os.getuid() and os.getsid(pid) == pid
                os.killpg(pid, signal.SIGTERM)
                break
            try:
                state = (process / 'stat').read_text().split(') ', 1)[1].split()[0]
                if state == 'Z':
                    break
            except FileNotFoundError:
                break
            time.sleep(.5)
        owner = prior.resources()
        while any(os.getsid(row['pid']) == pid for row in owner.processes()):
            time.sleep(.5)
        native = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
        write(folder / 'dispatcher_result.json', dict(returncode=None, complete=False,
            status='CONTROLLER_INTERRUPTED_DIAGNOSTIC_ONLY', native_status=native,
            reason='Parent exit status is unavailable. Retain evidence and repeat formal qualification in a new namespace.'))
        write(ROOT / 'state/active.json', dict(status='IDLE', time=time.time()))


if __name__ == '__main__':
    main()
