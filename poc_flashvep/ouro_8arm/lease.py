"""Signal only this user's recognized burners on the requested physical devices."""
import argparse
import os
import signal
import subprocess
import time
from .contract import ROOT,UUIDS,write,read
from poc_flashvep.ouro_request_pp.gpu_lease import burners

def active(gpus):
    rows=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
        uuid,pid=[v.strip() for v in line.split(',')]
        if uuid.removeprefix('GPU-') in UUIDS:
            gpu=4+UUIDS.index(uuid.removeprefix('GPU-'))
            if gpu in gpus:rows.append(dict(gpu=gpu,pid=int(pid)))
    return rows

def stop(gpus):
    found=burners()
    initial=active(gpus)
    for row in initial:
        proc=__import__('pathlib').Path('/proc')/str(row['pid'])
        logged_here=proc.stat().st_uid==os.getuid() and (proc/'fd/1').resolve()==ROOT/'burner_restore.log'
        assert os.getsid(row['pid'])==found.get(row['gpu']) or logged_here,f'Unrelated job left untouched: {row}'
    write(ROOT/'state'/f'lease_stop_{time.time_ns()}.json',dict(requested=gpus,workers=found,active=initial))
    # A newly started supervisor can create workers after an empty nvidia-smi result.
    # Stop only recorded supervisors whose entire GPU set is in this lease.
    for file in (ROOT/'state').glob('burner_restore_*.json'):
        record=read(file);pid=record['supervisor']
        if not set(record['GPUs'])<=set(gpus):continue
        try:
            proc=__import__('pathlib').Path('/proc')/str(pid)
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid and '/home/esjung/vllm-ep/utilize.py' in words:
                os.killpg(pid,signal.SIGTERM)
        except (FileNotFoundError,ProcessLookupError):pass
    for gpu,pid in found.items():
        if gpu in gpus:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    for row in initial:
        if row['gpu'] not in found:
            try:os.kill(row['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
    end=time.monotonic()+40
    while active(gpus) and time.monotonic()<end:time.sleep(.2)
    for row in active(gpus):
        assert row in initial,'A new GPU process appeared. Do not signal it.'
        proc=__import__('pathlib').Path('/proc')/str(row['pid'])
        assert proc.stat().st_uid==os.getuid() and ((proc/'fd/1').resolve()==ROOT/'burner_restore.log' or os.getsid(row['pid'])==found.get(row['gpu']))
        os.kill(row['pid'],signal.SIGKILL)
    end=time.monotonic()+10
    while active(gpus) and time.monotonic()<end:time.sleep(.2)
    assert not active(gpus),active(gpus)

def restore(gpus):
    existing=burners();missing=[g for g in gpus if g not in existing]
    assert not active(missing),'Do not start a burner on occupied devices.'
    if not missing:return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',VLLM_UTILIZE_GPUS=','.join(map(str,missing)))
    with (ROOT/'burner_restore.log').open('a') as log:
        p=subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'],cwd='/home/esjung/vllm-ep',env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state'/f'burner_restore_{p.pid}.json',dict(supervisor=p.pid,GPUs=missing))
    end=time.monotonic()+40
    while not set(missing)<=set(burners()) and time.monotonic()<end:time.sleep(.1)
    assert set(missing)<=set(burners()),'Burner startup did not expose its worker sessions.'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stop','restore','status']);p.add_argument('--gpus',default='4,5,6,7')
    a=p.parse_args();gpus=list(map(int,a.gpus.split(',')));assert set(gpus)<={4,5,6,7}
    if a.action=='status':print(dict(burners=burners(),active=active(gpus)))
    else:globals()[a.action](gpus)
