"""Observe the bounded campaign. Do not change acceptance gates or jobs."""
import fcntl
import os
import time
from .common import *


def main():
    initialize()
    stop_file = ROOT / 'state/CAMPAIGN_CLOSED.json'
    previous = None
    idle_since = None
    while time.time() < DEADLINE and not stop_file.exists():
        now = time.time()
        active_path = ROOT / 'state/active.json'
        active = json.loads(active_path.read_text()) if active_path.exists() else {}
        pid = active.get('pid')
        valid = False
        if pid:
            proc = Path('/proc', str(pid))
            try:
                words = (proc / 'cmdline').read_bytes().split(b'\0')
                valid = (proc.stat().st_uid == os.getuid()
                    and b'poc_flashvep.looped_pd_headroom_recovery.worker' in words
                    and any(str(ROOT / 'configs').encode() in word for word in words))
            except (FileNotFoundError, ProcessLookupError):
                pass
        name = active.get('name')
        row = dict(epoch=now, elapsed_hours=(now - START) / 3600,
            GPU_hours_remaining=max(0., GPU_DEADLINE - now) / 3600,
            total_hours_remaining=max(0., DEADLINE - now) / 3600,
            registered_name=name, registered_pid=pid, registered_process_verified=valid)
        if name:
            folder = ROOT / 'raw' / name
            path = folder / 'requests.jsonl'
            if path.exists():
                # File metadata does not parse token payloads during clean timing.
                row.update(result_bytes=path.stat().st_size,
                           seconds_since_result_write=now - path.stat().st_mtime)
        prior.append(ROOT / 'state/monitor_snapshots.jsonl', row)
        write(ROOT / 'state/monitor_last.json', row)
        signature = (name, valid)
        if signature != previous:
            print(json.dumps(row), flush=True)
            previous = signature
        if valid:
            idle_since = None
        else:
            idle_since = idle_since or now
            if now - idle_since > 300 and now < GPU_DEADLINE:
                # Use the same quiet-lane lock. Never start burners during
                # model startup, an experiment, or CPU analysis under that lock.
                with (ROOT / 'state/quiet_lane.lock').open('a+') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        owner = prior.resources()
                        if not owner.processes() and not owner.burners():
                            owner.restore(list(range(8)))
                            print('Approved idle-GPU burners restored', flush=True)
                    except BlockingIOError:
                        pass
        time.sleep(min(30., max(.1, DEADLINE - time.time())))


if __name__ == '__main__':
    main()
