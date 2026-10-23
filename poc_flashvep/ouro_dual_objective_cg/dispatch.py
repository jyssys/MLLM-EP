"""Reuse resume locks, queues, and warm barriers with a scoped worker entry point."""
import json
import os
import signal
import subprocess
import sys
import time
from .common import ROOT, REPO, FORK, PYTHON, GPUS, write, append
from .resources import processes, stop, restore
from ..ouro_budget_joint_cg import dispatch as inherited

inherited.ROOT=ROOT

def launch(job,gpu):
    assert gpu in GPUS and job['gpu'] in GPUS
    if inherited.completed(job):
        return dict(job=job['id'],GPU=gpu,status='SKIPPED_COMPLETE')
    target=ROOT/'raw'/job['id']
    assert not (target/'failure.json').exists(), 'Preserve the failed attempt. Register a new ID.'
    assert not (target/'identity.json').exists(), 'Preserve the interrupted attempt. Register a reviewed new ID.'
    assert not [r for r in processes() if r['gpu']==gpu], ('Occupied device',gpu)
    path=ROOT/'jobs'/f"{job['id']}.json"
    if path.exists(): assert json.loads(path.read_text())==job
    else: write(path,job)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(FORK)+':'+str(REPO),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUDA_CACHE_PATH=f'/home/esjung/.cache/ouro-dual-{gpu}/cuda',
        TRITON_CACHE_DIR=f'/home/esjung/.cache/ouro-dual-{gpu}/triton',
        FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='1',
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=f'/home/esjung/.cache/ouro-dual-{gpu}/fa4')
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    with (ROOT/'logs'/f"{job['id']}_gpu{gpu}.log").open('a') as log:
        child=subprocess.Popen(['taskset','-c',f'{gpu*16}-{gpu*16+15}',PYTHON,'-u','-m',
            'poc_flashvep.ouro_dual_objective_cg.worker','--job',str(path)],cwd=REPO,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(job=job['id'],GPU=gpu,pid=child.pid,status='RUNNING',wall_start=time.time())
        append(ROOT/'run_manifest.jsonl',record);write(ROOT/'state'/f'gpu{gpu}.json',record)
        print('START',job['id'],gpu,child.pid,flush=True)
        next_sample=0
        while child.poll() is None:
            now=time.time()
            if now-record['wall_start']>job.get('timeout_s',5400):
                os.killpg(child.pid,signal.SIGTERM);break
            if now>=next_sample:
                target.mkdir(parents=True,exist_ok=True)
                try:
                    stats=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                        '--query-gpu=uuid,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu',
                        '--format=csv,noheader,nounits'],text=True,timeout=8).strip()
                    append(target/'telemetry.jsonl',dict(wall_s=now,GPU=gpu,stats=stats,load=os.getloadavg(),
                        meminfo=open('/proc/meminfo').read(),diskstats=open('/proc/diskstats').read()))
                except (OSError,subprocess.SubprocessError) as e:
                    append(target/'telemetry.jsonl',dict(wall_s=now,error=repr(e)))
                next_sample=now+5
            time.sleep(.5)
        try: code=child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    result=dict(record,status='COMPLETED' if code==0 and inherited.completed(job) else 'FAILED',
        exit_code=code,wall_end=time.time())
    append(ROOT/'run_manifest.jsonl',result);write(ROOT/'state'/f'gpu{gpu}.json',result)
    print('END',job['id'],gpu,result['status'],flush=True)
    return result

def main():
    path=sys.argv[sys.argv.index('--manifest')+1]
    plan=json.loads(open(path).read())
    assert 'gpus' in plan and set(plan['gpus'])<=set(GPUS)
    jobs=plan.get('jobs',[j for group in plan.get('groups',[]) for j in group['jobs']])
    assert all(j['gpu'] in GPUS for j in jobs)
    inherited.launch=launch;inherited.stop=stop;inherited.restore=restore
    inherited.main()

if __name__=='__main__': main()
