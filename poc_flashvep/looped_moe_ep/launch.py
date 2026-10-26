"""One rank per physical device. Only owned burner sessions can be stopped."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from .common import *
from . import resources

def launch(config):
    setup()
    gpus=config['gpus']
    assert len(set(gpus))==len(gpus) and set(gpus)<=set(range(8))
    target=ROOT/'raw'/config['id']
    if all((target/f'complete.rank{r}.json').exists() for r in range(len(gpus))):
        return dict(status='ALREADY_COMPLETE',id=config['id'])
    assert not target.exists(),f'Use a new attempt ID: {target}'
    path=ROOT/'configs'/f"{config['id']}.json"
    write(path,config)
    resources.stop(gpus)
    children=[]
    files=[]
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        for rank,gpu in enumerate(gpus):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                ENABLE_FLASH_ATTENTION_4='true',ENABLE_FLASH_ATTENTION_3='false',
                TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_DEBUG='WARN',NCCL_SOCKET_IFNAME='lo',
                CUDA_CACHE_PATH=str(ROOT/'state'/f'cuda_gpu{gpu}'),TRITON_CACHE_DIR=str(ROOT/'state'/f'triton_gpu{gpu}'))
            log=(ROOT/'logs'/f"{config['id']}.rank{rank}.log").open('w')
            files.append(log)
            cmd=['taskset','-c',f'{gpu*16}-{gpu*16+15}',str(PYTHON),'-m','poc_flashvep.looped_moe_ep.worker',str(path),
                '--rank',str(rank),'--port',str(port)]
            children.append(subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            time.sleep(2)
        until=time.monotonic()+config.get('timeout_s',1800)
        while not all(c.poll() is not None for c in children):
            if any(c.poll() not in (None,0) for c in children):
                raise RuntimeError('One EP rank failed. See preserved per-rank logs.')
            assert time.monotonic()<until,'Bounded EP run timed out'
            time.sleep(1)
        assert all(c.returncode==0 for c in children)
        result=dict(status='PASS',id=config['id'],world=len(gpus),physical_gpus=gpus)
        write(target/'complete.json',result)
        return result
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid,signal.SIGTERM)
        for c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid,signal.SIGKILL)
                c.wait()
        for log in files:log.close()
        resources.restore(gpus)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=list(MODELS),required=True)
    p.add_argument('--gpus',required=True)
    p.add_argument('--id',required=True)
    p.add_argument('--phase',default='p0')
    args=p.parse_args()
    print(launch(dict(model=args.model,gpus=list(map(int,args.gpus.split(','))),id=args.id,phase=args.phase)),flush=True)

if __name__=='__main__':main()
