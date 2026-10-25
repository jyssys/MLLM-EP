"""Resumable queue of independent single-GPU jobs. Never shard a model."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

from .common import ROOT, PYTHON, REPO, record, write
from . import resources

def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    a=p.parse_args()
    plan=json.loads(Path(a.manifest).read_text())
    phase=plan.get('phase')
    if phase=='p1':
        for model,gpu in [('ifm_ouro',0),('ifm_huginn',3),('foil1',6)]:
            assert (ROOT/'raw'/f'{model}_p0_gpu{gpu}_native_v5'/'complete.json').exists(), 'P0 prerequisite missing'
    prior={'calibration':'p1','heldout':'calibration'}.get(phase)
    if prior:
        previous=json.loads((ROOT/f'plan_{prior}.json').read_text())
        assert all((ROOT/'raw'/j['id']/'complete.json').exists() for j in previous['jobs']),f'Incomplete prerequisite {prior}'
    gpus=list(map(int,a.gpus.split(',')))
    assert set(gpus)<=set(range(8))
    (ROOT/'state').mkdir(exist_ok=True,parents=True)
    lock=(ROOT/'state/dispatch.lock').open('a')
    # The last worker can finish before its burner restoration completes.
    # Serialize the next campaign instead of treating this interval as a failed experiment.
    fcntl.flock(lock,fcntl.LOCK_EX)
    pending=queue.Queue()
    for job in plan['jobs']:
        target=ROOT/'raw'/job['id']
        if (target/'complete.json').exists():
            continue
        assert not (target/'failure.json').exists(),('Review failed job and use a new ID',job['id'])
        config=ROOT/'state'/f"{job['id']}.json"
        if config.exists():
            assert json.loads(config.read_text())==job
        else:
            write(config,job)
        pending.put(job)
    if pending.empty():
        print('NO_PENDING_JOBS')
        return
    resources.stop(gpus)
    load_lock=threading.Lock()
    failures=[]
    def worker(gpu):
        try:
            while True:
                try:
                    job=pending.get_nowait()
                except queue.Empty:
                    break
                eligible=job.get('gpus',gpus)
                if gpu not in eligible:
                    pending.put(job)
                    pending.task_done()
                    time.sleep(.2)
                    continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                    ENABLE_FLASH_ATTENTION_4='true',ENABLE_FLASH_ATTENTION_3='false',
                    CUDA_CACHE_PATH=str(ROOT/'state'/f'cuda_gpu{gpu}'),TRITON_CACHE_DIR=str(ROOT/'state'/f'triton_gpu{gpu}'))
                command=['taskset','-c',f'{gpu*16}-{gpu*16+15}',str(PYTHON),'-m','poc_flashvep.looped_moe_headroom.worker',
                    '--model',job['model'],'--phase',job['phase'],'--run-id',job['id'],
                    '--job',str(ROOT/'state'/f"{job['id']}.json")]
                logpath=ROOT/'logs'/f"{job['id']}.log"
                with logpath.open('a') as log:
                    with load_lock:
                        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        record('WORKER_START',gpu=gpu,job=job,pid=child.pid)
                        time.sleep(3)
                    try:
                        rc=child.wait(timeout=job.get('timeout_s',14400))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        rc=child.wait(timeout=30)
                        write(ROOT/'raw'/job['id']/'launcher_failure.json',dict(reason='TIMEOUT',returncode=rc))
                record('WORKER_END',gpu=gpu,job_id=job['id'],returncode=rc)
                if rc or not (ROOT/'raw'/job['id']/'complete.json').exists():
                    failures.append(job['id'])
                pending.task_done()
        finally:
            resources.restore([gpu])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in gpus]
        for future in futures:
            future.result()
    assert not failures,('FAILED_JOBS',failures)

if __name__=='__main__':
    main()
