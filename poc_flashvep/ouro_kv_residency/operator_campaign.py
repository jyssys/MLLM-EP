"""Lease real device groups. Restore verified burners after each campaign."""
import argparse
import json
import os
import subprocess
import time
import fcntl
from .common import ROOT,REPO,FORK,PYTHON,write,append,digest
from .resources import stop,restore

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--worlds',default='2,4,8')
    p.add_argument('--full',action='store_true');p.add_argument('--batches',default='1,2,4,8')
    p.add_argument('--repeats',type=int,default=25);p.add_argument('--starts',type=int,default=1)
    p.add_argument('--edge',action='store_true')
    p.add_argument('--contexts',default='2048,8192,32768,65408')
    p.add_argument('--representative',action='store_true');p.add_argument('--cache-pressure',action='store_true')
    p.add_argument('--after-job')
    a=p.parse_args();(ROOT/'state').mkdir(exist_ok=True)
    if a.after_job:
        path=ROOT/'raw'/a.after_job
        until=time.monotonic()+10800
        while not (path/'completed.json').exists():
            assert not (path/'failure.json').exists(),'Required fixture job failed.'
            assert time.monotonic()<until
            time.sleep(2)
    lock=(ROOT/'state'/'dispatcher.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    if a.full:
        selected=[]
        for p0 in map(int,a.contexts.split(',')):
            selected.extend(json.loads((ROOT/'raw'/f'fixture_P{p0}'/'fixtures.json').read_text()))
    else:
        selected=json.loads((ROOT/'raw'/'fixture_P2048'/'fixtures.json').read_text())
        selected=[r for r in selected if r['depth']==1 and r['layer']==12]
    if a.edge:
        selected=[dict(selected[0],length_override=length) for length in (1,15,17,31)]
    if a.representative:selected=[r for r in selected if r['depth']==1 and r['layer']==12]
    path=ROOT/'plans'/f'operator_{a.tag}.json';assert not path.exists()
    write(path,selected)
    # Quiet lane reserves every device. No unrelated process can pass ownership checks.
    stop(list(range(8)))
    try:
        for start in range(a.starts):
            for world in map(int,a.worlds.split(',')):
                name=f'{a.tag}_P{world}_s{start}';target=ROOT/'distributed'/name
                if all((target/f'rank{r}_completed.json').exists() for r in range(world)):
                    continue
                assert not target.exists(),'Preserve interrupted group. Use a new tag.'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))),
                    PYTHONPATH=str(FORK)+':'+str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
                command=[PYTHON,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
                    '-m','poc_flashvep.ouro_kv_residency.distributed','--run',name,'--fixtures',str(path),
                    '--batches',a.batches,'--repeats',str(a.repeats),'--skew']
                if a.cache_pressure:command.append('--cache-pressure')
                (ROOT/'logs').mkdir(exist_ok=True)
                append(ROOT/'run_manifest.jsonl',dict(id=name,kind='DISTRIBUTED_OPERATOR',world=world,command=command,start=time.time()))
                with (ROOT/'logs'/f'{name}.log').open('a') as log:
                    subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                append(ROOT/'run_manifest.jsonl',dict(id=name,status='COMPLETED',end=time.time()))
                print('OPERATOR_COMPLETE',name,flush=True)
    finally:restore(list(range(8)))

if __name__=='__main__':main()
