"""Bounded, ownership-checked launches. One CUDA device per rank."""
import os
import signal
import socket
import subprocess
import time
from .common import *
from . import resources

def launch(config):
    setup()
    gpus=config['gpus'];world=len(gpus)
    assert len(set(gpus))==world and set(gpus)<=set(range(8))
    target=ROOT/'raw'/config['id']
    if (target/'complete.json').exists():return dict(id=config['id'],status='ALREADY_COMPLETE')
    assert not target.exists(),f'Preserve every attempt. Use a new ID: {target}'
    path=ROOT/'runs/configs'/f'{config["id"]}.json'
    write(path,config)
    resources.stop(gpus)
    children=[];handles=[]
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        for rank,gpu in enumerate(gpus):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                ENABLE_FLASH_ATTENTION_3='true',ENABLE_FLASH_ATTENTION_4='false',
                PYTHONPATH=str(DEPS)+':'+str(REPO)+':'+str(XLLM),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false',HF_DATASETS_TRUST_REMOTE_CODE='1',
                TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_DEBUG='WARN',NCCL_SOCKET_IFNAME='lo',
                CUDA_CACHE_PATH=str(ROOT/'state'/f'cuda_gpu{gpu}'),TRITON_CACHE_DIR=str(ROOT/'state'/f'triton_gpu{gpu}'))
            if config.get('allocator_conf'):
                env['PYTORCH_CUDA_ALLOC_CONF']=config['allocator_conf']
            log=(ROOT/'runs/logs'/f'{config["id"]}.rank{rank}.log').open('w');handles.append(log)
            python=config.get('python',str(PYTHON))
            command=['taskset','-c',f'{16*gpu}-{16*gpu+15}',python,'-m','poc_flashvep.looped_moe_affinity.worker',str(path),'--rank',str(rank),'--port',str(port)]
            append(ROOT/'runs/commands.jsonl',dict(config=config['id'],rank=rank,physical_gpu=gpu,command=command,
                source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),time=time.time()))
            children.append(subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            time.sleep(1)
        deadline=time.monotonic()+config.get('timeout_s',3600)
        while not all(p.poll() is not None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('A worker failed. Preserve its rank log.')
            assert time.monotonic()<deadline,'Bounded run timeout'
            time.sleep(1)
        codes=[p.returncode for p in children]
        if any(code!=0 for code in codes):
            write(target/'launch_failure.json',dict(returncodes=codes,rank_logs=[f'{config["id"]}.rank{r}.log' for r in range(world)]))
            raise RuntimeError(f'Worker exit codes {codes}. Read preserved rank failures for {config["id"]}.')
        result=dict(id=config['id'],status='PASS',gpus=gpus)
        write(target/'complete.json',result)
        return result
    finally:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for handle in handles:handle.close()
        resources.restore(gpus)
