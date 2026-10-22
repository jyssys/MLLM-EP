"""Independent, affinity-aware jobs with warm barriers and hash validation."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from .common import ROOT, REPO, FORK, PYTHON, digest, write, append
from .resources import processes, stop, restore

def completed(job):
    path=ROOT/'raw'/job['id']/'completed.json'
    if not path.exists():
        return False
    data=json.loads(path.read_text())
    for item in data['artifacts']:
        artifact=path.parent/item['path']
        assert artifact.exists() and digest(artifact)==item['sha256'],('CORRUPTED_COMPLETE',artifact)
    return True

def launch(job,gpu):
    if completed(job):
        return dict(job=job['id'],GPU=gpu,status='SKIPPED_COMPLETE')
    target=ROOT/'raw'/job['id']
    assert not (target/'failure.json').exists(), 'Preserve failure. Register a new attempt ID.'
    assert not (target/'identity.json').exists(), 'Interrupted attempt is immutable. Register a reviewed retry ID.'
    assert not [r for r in processes() if r['gpu']==gpu], f'GPU{gpu} occupied'
    path=ROOT/'jobs'/f"{job['id']}.json"
    if path.exists():
        assert json.loads(path.read_text())==job,'Immutable job changed'
    else:
        write(path,job)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(FORK)+':'+str(REPO),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUDA_CACHE_PATH=f'/home/esjung/.cache/ouro-deadline-{gpu}/cuda',
        TRITON_CACHE_DIR=f'/home/esjung/.cache/ouro-deadline-{gpu}/triton',
        FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='1',
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=f'/home/esjung/.cache/ouro-deadline-{gpu}/fa4')
    (ROOT/'logs').mkdir(exist_ok=True,parents=True)
    with (ROOT/'logs'/f"{job['id']}_gpu{gpu}.log").open('a') as log:
        child=subprocess.Popen([PYTHON,'-u','-m','poc_flashvep.ouro_adaptive_chunk_loop.worker','--job',str(path)],
            cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(job=job['id'],GPU=gpu,pid=child.pid,status='RUNNING',wall_start=time.time())
        append(ROOT/'run_manifest.jsonl',record)
        write(ROOT/'state'/f'gpu{gpu}.json',record)
        print('START',job['id'],'GPU',gpu,'PID',child.pid,flush=True)
        with (target/'telemetry.jsonl').open('a') if target.exists() else __import__('contextlib').nullcontext() as unused:
            pass
        next_record=0
        timeout=job.get('timeout_s',5400)
        while child.poll() is None:
            now=time.time()
            if now-record['wall_start']>timeout:
                os.killpg(child.pid,__import__('signal').SIGTERM)
                break
            if now>=next_record:
                target.mkdir(parents=True,exist_ok=True)
                try:
                    gpu_stats=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=8).strip()
                    append(target/'telemetry.jsonl',dict(wall_s=now,gpu=gpu,stats=gpu_stats,load=os.getloadavg(),meminfo=open('/proc/meminfo').read(),io=open('/proc/diskstats').read()))
                except (OSError,subprocess.SubprocessError) as error:
                    append(target/'telemetry.jsonl',dict(wall_s=now,error=repr(error)))
                next_record=now+5
            time.sleep(.5)
        try:
            code=child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,__import__('signal').SIGKILL)
            code=child.wait()
    result=dict(record,status='COMPLETED' if code==0 and completed(job) else 'FAILED',exit_code=code,wall_end=time.time())
    append(ROOT/'run_manifest.jsonl',result)
    write(ROOT/'state'/f'gpu{gpu}.json',result)
    print('END',job['id'],gpu,result['status'],flush=True)
    return result

def run_group(jobs,barrier=None):
    # All jobs in a calibration group have explicit physical affinity.
    write(ROOT/'state'/'active_group.json',dict(jobs=[j['id'] for j in jobs],barrier=barrier))
    if barrier:
        release=ROOT/'state'/f'release_{barrier}.json'
        assert not release.exists(),'A barrier ID cannot be reused'
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(launch,j,j['gpu']) for j in jobs]
        if barrier:
            until=time.monotonic()+1800
            while not all((ROOT/'raw'/j['id']/'READY.json').exists() for j in jobs):
                if any(f.done() and (f.exception() or f.result()['status']=='FAILED') for f in futures):
                    raise RuntimeError('Warm barrier participant failed')
                assert time.monotonic()<until,'Warmup group timeout'
                time.sleep(1)
            write(release,dict(start_wall_s=time.time()+2,jobs=[j['id'] for j in jobs]))
        results=[f.result() for f in futures]
    assert all(r['status'] in ('COMPLETED','SKIPPED_COMPLETE') for r in results),results
    return results

def main():
    import fcntl
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--manage-burners',action='store_true')
    a=p.parse_args()
    (ROOT/'state').mkdir(parents=True,exist_ok=True)
    lease_file=(ROOT/'state'/'dispatcher.lock').open('a')
    fcntl.flock(lease_file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(open(a.manifest).read())
    devices=plan.get('gpus',list(range(8)))
    assert set(devices)<=set(range(8))
    if a.manage_burners:
        stop(devices)
    try:
        if 'groups' in plan:
            for group in plan['groups']:
                jobs=[j for j in group['jobs'] if not completed(j)]
                if jobs:
                    run_group(jobs,group.get('barrier'))
        else:
            # Prefer registered crossover devices. Movable blocks retain all
            # paired policies inside one physical GPU/process-start job.
            pending=list(plan['jobs'])
            claim_lock=threading.Lock()
            def worker(gpu):
                try:
                    while True:
                        with claim_lock:
                            index=next((i for i,j in enumerate(pending) if j['gpu']==gpu),None)
                            if index is None:
                                index=next((i for i,j in enumerate(pending) if j.get('allow_any_gpu')),None)
                            job=pending.pop(index) if index is not None else None
                        if job is None:
                            break
                        result=launch(job,gpu)
                        if result['status']=='FAILED':
                            raise RuntimeError(result)
                finally:
                    if a.manage_burners:
                        restore([gpu])
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                list(pool.map(worker,devices))
    finally:
        if a.manage_burners:
            restore(devices)

if __name__=='__main__':
    main()
