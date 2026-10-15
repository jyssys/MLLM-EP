"""Restore the user-requested scoped launcher. Never signal an unidentified job."""
import datetime
import os
import signal
import subprocess
import time
from poc_looped_multigpu_kv_placement.contract import *
from poc_looped_multigpu_kv_placement.scripts.gpu_session import contexts
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor


def main():
    target=ROOT/'results/final_burner_restore.json';assert not target.exists()
    initial=contexts(GPUS);workers={}
    for pid in initial:
        worker,_=burner_worker_ancestor(pid)
        proc=Path('/proc')/str(worker);assert proc.stat().st_uid==os.getuid()
        argv=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        assert int(argv[argv.index('--gpu')+1]) in GPUS
        assert os.getpgid(worker)==worker
        workers[worker]=argv
    for worker in workers:
        try:os.killpg(worker,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+45
    while contexts(GPUS) and time.monotonic()<deadline:time.sleep(.5)
    if contexts(GPUS):
        for pid in contexts(GPUS):
            worker,_=burner_worker_ancestor(pid);assert worker in workers
            try:os.killpg(worker,signal.SIGKILL)
            except ProcessLookupError:pass
        deadline=time.monotonic()+25
        while contexts(GPUS) and time.monotonic()<deadline:time.sleep(.5)
    assert not contexts(GPUS),'Do not replace a competing or unreleased context'
    live=ASSETS/'burner_logs';live.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK,VLLM_UTILIZE_GPUS=MASK)
    env.pop('PYTHONPATH',None);env.pop('LOOPED_KV_OUTPUT',None)
    with (live/'final_scoped_launcher.log').open('a') as stream:
        launcher=subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'],cwd='/home/esjung/vllm-ep',
            env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mask=MASK,
        launcher_PID=launcher.pid,command=['bash','/home/esjung/vllm-ep/run_utilize.sh'],
        override={'VLLM_UTILIZE_GPUS':MASK,'CUDA_VISIBLE_DEVICES':MASK},
        initial_contexts=initial,replaced_only_verified_own_burner_workers=workers,
        foreign_processes_or_GPUs_signalled=False,verification='PENDING')
    write(target,record)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        assert launcher.poll() is None,'Scoped launcher exited before validation'
        sample=subprocess.check_output(['nvidia-smi','--id=4,5,6,7',
            '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        rows=[line.split(',') for line in sample.splitlines()]
        if len(rows)==4 and all(int(row[2])>65000 and int(row[3])>=20 for row in rows):
            evidence=[]
            for pid in contexts(GPUS):
                worker,argv=burner_worker_ancestor(pid)
                evidence.append(dict(context_pid=pid,worker_pid=worker,worker_argv=argv))
            record.update(verification='RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs',sample=sample,contexts=evidence)
            write(target,record);print(record['verification'],flush=True);return
        time.sleep(3)
    record.update(verification='LAUNCHED_BUT_UTILIZATION_CHECK_UNRESOLVED',sample=sample)
    write(target,record);raise RuntimeError(record['verification'])


if __name__=='__main__':main()
