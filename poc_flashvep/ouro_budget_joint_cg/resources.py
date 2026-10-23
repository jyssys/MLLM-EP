"""Only signal verified owned burner sessions. Never signal other jobs."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT, PRIOR, UUIDS, write, append

SCRIPT = '/home/esjung/vllm-ep/utilize.py'
APPROVED_LOGS = {ROOT/'burner.log', PRIOR/'burner.log', Path('/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/burner_restore.log')}

def processes():
    rows = []
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    for line in output.splitlines():
        uuid, pid = (s.strip() for s in line.split(','))
        gpu = next((g for g, u in UUIDS.items() if u == uuid), None)
        if gpu is not None:
            rows.append(dict(gpu=gpu, pid=int(pid)))
    return rows

def burners():
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            words = (proc/'cmdline').read_bytes().decode().split('\0')
            if (proc.stat().st_uid == os.getuid() and SCRIPT in words and '--worker' in words
                    and (proc/'fd/1').resolve() in APPROVED_LOGS):
                gpu = int(words[words.index('--gpu')+1])
                pid = int(proc.name)
                if os.getsid(pid) == pid and gpu in UUIDS:
                    found[gpu] = pid
        except (OSError, ValueError, UnicodeError):
            continue
    return found

def stop(gpus):
    owned = burners()
    before = [r for r in processes() if r['gpu'] in gpus]
    for row in before:
        proc = Path('/proc')/str(row['pid'])
        assert proc.stat().st_uid == os.getuid() and os.getsid(row['pid']) == owned.get(row['gpu']), ('UNRELATED_JOB_LEFT_UNTOUCHED', row)
    append(ROOT/'resource_ledger.jsonl', dict(action='STOP_OWNED_BURNERS', time=time.time(), GPUs=gpus, owned=owned, active=before))
    # An owned supervisor can spawn workers after an initially empty GPU query.
    # Terminate it first only when this lease contains its entire device set.
    for file in list((ROOT/'state').glob('burner_*.json'))+list((PRIOR/'state').glob('burner_*.json')):
        record=__import__('json').loads(file.read_text())
        if not set(record['GPUs'])<=set(gpus):
            continue
        pid=record['supervisor']
        proc=Path('/proc')/str(pid)
        try:
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid and SCRIPT in words and '--worker' not in words:
                os.kill(pid,signal.SIGTERM)
                time.sleep(.2)
                if proc.exists() and (proc/'cmdline').read_bytes():
                    os.kill(pid,signal.SIGKILL)
        except (FileNotFoundError,ProcessLookupError):
            pass
    owned=burners()
    for gpu, pid in owned.items():
        if gpu in gpus:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    until = time.monotonic()+45
    while (any(r['gpu'] in gpus for r in processes()) or set(burners())&set(gpus)) and time.monotonic() < until:
        time.sleep(.5)
    assert not [r for r in processes() if r['gpu'] in gpus], 'A device remains occupied. No escalation.'
    assert not set(burners())&set(gpus),'An owned burner worker has not exited'

def restore(gpus):
    occupied = {r['gpu'] for r in processes()}
    owned = burners()
    missing = [g for g in gpus if g not in occupied and g not in owned]
    if not missing:
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,missing)), VLLM_UTILIZE_GPUS=','.join(map(str,missing)),
        VLLM_UTILIZE_PYTHON='/home/esjung/anaconda3/envs/flashvep-poc/bin/python')
    with (ROOT/'burner.log').open('a') as log:
        child = subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'], cwd='/home/esjung/vllm-ep', env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT/'state'/f'burner_{child.pid}.json', dict(GPUs=missing, supervisor=child.pid, time=time.time()))
    until=time.monotonic()+30
    while not set(missing)<=set(burners()) and child.poll() is None and time.monotonic()<until:
        time.sleep(.1)
    assert set(missing)<=set(burners()),'Burner supervisor did not expose its owned worker sessions'
    print('BURNER_STARTED', missing, child.pid, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['status','stop','restore'])
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    gpus = list(map(int,a.gpus.split(',')))
    assert set(gpus) <= set(UUIDS)
    if a.action == 'status':
        print(dict(owned_burners=burners(), active=processes()))
    else:
        globals()[a.action](gpus)
