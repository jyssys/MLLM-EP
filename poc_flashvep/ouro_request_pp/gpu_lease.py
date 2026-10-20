"""Stop only identified project burners. Never signal an unrelated process."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from .contract import ROOT, GPUS, UUIDS, write


def burners():
    found = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            words = (path/'cmdline').read_bytes().decode().split('\0')
            if '/home/esjung/vllm-ep/utilize.py' not in words or '--gpu' not in words:
                continue
            gpu = int(words[words.index('--gpu')+1])
            pid = int(path.name)
            if gpu in GPUS and os.getsid(pid) == pid:
                found[gpu] = pid
        except (ProcessLookupError, FileNotFoundError, PermissionError, ValueError):
            pass
    return found


def active():
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    result = []
    for line in raw.splitlines():
        uuid,pid = [v.strip() for v in line.split(',')]
        if uuid.removeprefix('GPU-') in UUIDS:
            result.append(dict(gpu=4+UUIDS.index(uuid.removeprefix('GPU-')), pid=int(pid)))
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stop','restore','status']); args = p.parse_args()
    b = burners()
    if args.action == 'stop':
        for row in active():
            sid = os.getsid(row['pid'])
            if sid != b.get(row['gpu']):
                raise RuntimeError(f'GPU unavailable. Unrelated process left untouched: {row}')
        write(ROOT/'burners_before_stop.json', dict(session_leaders=b, active=active()))
        for gpu,pid in b.items():
            os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic()+30
        while active() and time.monotonic() < deadline:
            time.sleep(.2)
        assert not active(), active()
        print('ALLOWED_GPUS_FREE', GPUS)
    elif args.action == 'restore':
        missing = [g for g in GPUS if g not in b]
        busy = {r['gpu'] for r in active()}
        assert not (set(missing)&busy), 'Do not start a burner on an occupied GPU.'
        if missing:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
            env['VLLM_UTILIZE_GPUS'] = ','.join(map(str,missing))
            log = (ROOT/'burner_restore.log').open('a')
            proc = subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'],cwd='/home/esjung/vllm-ep',
                                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(ROOT/'burner_restore.json', dict(supervisor=proc.pid, GPUs=missing))
            print('RESTORING', missing, proc.pid)
    else:
        print(dict(burners=b, active=active()))


if __name__ == '__main__':
    main()
