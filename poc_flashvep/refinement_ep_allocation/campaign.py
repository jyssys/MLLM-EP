"""Durable baseline queue. Resume skips complete run artifacts."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from .common import ROOT, REPO, HF_PYTHON, EP_PYTHON, DINFER, LEGACY_EP, setup, write, append
from . import resources

def environment(gpus):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('LLADA_EP_TRACE','LLADA_BLOCK_TRACE',
        'LLADA_DENOISE_TRACE','FLASHVEP_','SD_EP_','AGENT_EP_','LLADA_MOE_'))}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)), PYTHONUNBUFFERED='1',
        PYTHONPATH=f'{REPO}:{DINFER}/python:{LEGACY_EP}', OMP_NUM_THREADS='4',
        TORCHDYNAMO_DISABLE='1', SGL_ENABLE_JIT_DEEPGEMM='false', SGLANG_ENABLE_JIT_DEEPGEMM='false',
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='1024')
    return env

def launch(name, cmd, gpus):
    log = ROOT/'logs'/f'{name}.log'
    with log.open('a') as stream:
        child = subprocess.Popen(cmd, cwd=REPO, env=environment(gpus), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    append(ROOT/'commands.jsonl', dict(name=name, command=cmd, gpus=gpus, pid=child.pid,
        time=time.time(), log=str(log)))
    return child

def run():
    setup()
    lock = (ROOT/'state/campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'state/campaign.json', dict(status='RUNNING', pid=os.getpid(), stage='BASELINE_SMOKE', time=time.time()))
    resources.base.stop(list(range(8)))
    resources.base.restore([5,6,7])
    base = ['-m','poc_flashvep.refinement_ep_allocation.']
    manifest = ROOT/'manifests'
    hf = launch('hf_sanity', [HF_PYTHON,'-m',base[1]+'hf_worker','--manifest',str(manifest/'sanity.jsonl'),
        '--gpu','4','--run-id','sanity_reference'], [4])
    ep = launch('ep4_A_sanity', [EP_PYTHON,'-m',base[1]+'ep4_worker','--manifest',str(manifest/'sanity.jsonl'),
        '--gpus','0,1,2,3','--run-id','A_sanity','--parity','--trace'], [0,1,2,3])
    jobs = {'hf_smoke':hf, 'ep_smoke':ep}
    hf_queues = {}
    smoke_done = set()
    full_ep = None
    try:
        while jobs:
            for name, child in list(jobs.items()):
                code = child.poll()
                if code is None:
                    continue
                append(ROOT/'job_events.jsonl', dict(name=name,pid=child.pid,exit_code=code,time=time.time()))
                del jobs[name]
                if name == 'hf_smoke':
                    smoke_done.add(name)
                    if code:
                        resources.base.restore([4])
                    else:
                        resources.base.stop([5,6,7])
                        for shard, gpu in enumerate(range(4,8)):
                            hf_queues[gpu] = [
                                [HF_PYTHON,'-m',base[1]+'hf_worker','--manifest',str(manifest/f'{task}_test.jsonl'),
                                 '--gpu',str(gpu),'--shard',str(shard),'--shards','4','--run-id',f'full_{task}']
                                for task in ('gsm8k','humaneval')]
                            jobs[f'hf_gpu{gpu}'] = launch(f'hf_gpu{gpu}_gsm8k',hf_queues[gpu].pop(0),[gpu])
                elif name == 'ep_smoke':
                    smoke_done.add(name)
                    if code:
                        resources.base.restore([0,1,2,3])
                    else:
                        # Quality only. Concurrent reference jobs invalidate isolated latency claims.
                        full_ep = [('gsm8k','A_full_gsm8k'), ('humaneval','A_full_humaneval')]
                        task, runid = full_ep.pop(0)
                        jobs['ep_full'] = launch(runid, [EP_PYTHON,'-m',base[1]+'ep4_worker','--manifest',
                            str(manifest/f'{task}_test.jsonl'),'--gpus','0,1,2,3','--run-id',runid], [0,1,2,3])
                elif name.startswith('hf_gpu'):
                    gpu = int(name.removeprefix('hf_gpu'))
                    if not code and hf_queues[gpu]:
                        jobs[name] = launch(f'hf_gpu{gpu}_humaneval',hf_queues[gpu].pop(0),[gpu])
                    else:
                        resources.base.restore([gpu])
                elif name == 'ep_full':
                    if not code and full_ep:
                        task, runid = full_ep.pop(0)
                        jobs[name] = launch(runid, [EP_PYTHON,'-m',base[1]+'ep4_worker','--manifest',
                            str(manifest/f'{task}_test.jsonl'),'--gpus','0,1,2,3','--run-id',runid], [0,1,2,3])
                    else:
                        resources.base.restore([0,1,2,3])
            write(ROOT/'state/campaign.json', dict(status='RUNNING', pid=os.getpid(), stage='BASELINE_QUALITY',
                active={n:p.pid for n,p in jobs.items()}, time=time.time()))
            if jobs:
                time.sleep(5)
        write(ROOT/'state/campaign.json', dict(status='BASELINE_QUEUE_ENDED_REVIEW_REQUIRED', pid=os.getpid(), time=time.time(),
            next='Review fresh quality and parity. Phases 2-6 are not automatically approved.'))
    finally:
        # Do not terminate children on an agent disconnect. This supervisor owns its durable queue.
        if not jobs:
            resources.base.restore(list(range(8)))

def main():
    setup()
    if (ROOT/'pilot/scope_amendment.json').exists():
        raise RuntimeError('The user deferred full evaluation. Use the pilot queue. Do not restart this full queue.')
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('start','run'))
    a = p.parse_args()
    if a.action == 'run':
        run()
        return
    with (ROOT/'logs/supervisor.log').open('a') as log:
        child = subprocess.Popen([sys.executable,'-m','poc_flashvep.refinement_ep_allocation.campaign','run'],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT/'state/supervisor.json', dict(pid=child.pid,time=time.time()))
    print('STARTED',child.pid)

if __name__ == '__main__':
    main()
