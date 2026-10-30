"""Read-only GPU process samples. Never signal unrelated processes."""
import json
import os
from pathlib import Path
import subprocess
import time

from .common import ROOT, GPU_STOP, write


def main():
    write(ROOT / 'state/resource_observer.json', dict(pid=os.getpid(), started=time.time(), period_s=10))
    path = ROOT / 'logs/all_GPU_process_observations.jsonl'
    while time.time() < GPU_STOP and not (ROOT / 'state/stop_resource_observer.json').exists():
        owned_sessions = set()
        for state in (ROOT / 'state').glob('*.json'):
            try:
                pid = json.loads(state.read_text()).get('pid')
                proc = Path(f'/proc/{pid}')
                if pid and proc.stat().st_uid == os.getuid() and b'poc_flashvep.looped_pd_b_slo_serving.worker' in (proc / 'cmdline').read_bytes():
                    owned_sessions.add(pid)
            except (ValueError, FileNotFoundError, ProcessLookupError):
                continue
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10)
        processes = []
        for line in result.stdout.splitlines():
            fields = [value.strip() for value in line.split(',')]
            if len(fields) != 3 or not fields[1].isdigit():
                continue
            pid = int(fields[1])
            try:
                session, uid = os.getsid(pid), Path(f'/proc/{pid}').stat().st_uid
                owned = uid == os.getuid() and session in owned_sessions
            except (FileNotFoundError, ProcessLookupError):
                session, uid, owned = None, None, False
            processes.append(dict(GPU_UUID=fields[0], pid=pid, session=session, uid=uid,
                                  HBM_MiB=fields[2], registered_experiment_session=owned))
        with path.open('a') as stream:
            stream.write(json.dumps(dict(epoch=time.time(), monotonic=time.monotonic(), processes=processes,
                                         query_returncode=result.returncode)) + '\n')
        for _ in range(10):
            if (ROOT / 'state/stop_resource_observer.json').exists():
                break
            time.sleep(1)
    write(ROOT / 'state/resource_observer_done.json', dict(pid=os.getpid(), finished=time.time()))


if __name__ == '__main__':
    main()
