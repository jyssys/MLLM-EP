"""Durable commands and scoped GPU ownership, including external worker cleanup."""
import os
import signal
import subprocess
import tarfile
import time
from .contract import *
from .lease import stop,restore,active


def GPU_telemetry(gpus):
    return dict(physical_GPUs=list(gpus),host_epoch_s=time.time(),
        csv=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,gpus)),
            '--query-gpu=index,uuid,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,memory.used',
            '--format=csv'],text=True),scope='Run-boundary snapshot outside serving. Not a continuous clock or power trace.')

def execution_environment(argv,gpus):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    interpreter=next((word for word in argv if word in [PYTHON,VLLM_PYTHON]),argv[0])
    env['PATH']=str(Path(interpreter).parent)+os.pathsep+env['PATH']
    return env


def execute(label,argv,gpus=(4,5,6,7),timeout=14400,restore_after=True):
    assert not (ROOT/'logs'/f'{label}.log').exists(),label
    stop(gpus)
    write(ROOT/'commands'/f'{label}_GPU_before.json',GPU_telemetry(gpus))
    env=execution_environment(argv,gpus)
    write(ROOT/'commands'/f'{label}.json',dict(argv=argv,GPUs=gpus,cwd=PROJECT,
        env={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','HF_HUB_OFFLINE','PATH']},
        source_hashes={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'poc_flashvep/ouro_8arm').glob('*.py')}))
    with tarfile.open(ROOT/'commands'/f'{label}_source.tar.gz','w:gz') as archive:
        for file in sorted((PROJECT/'poc_flashvep/ouro_8arm').glob('*.py')):
            archive.add(file,arcname=str(file.relative_to(PROJECT)))
    start=time.time();code=-1
    try:
        with (ROOT/'logs'/f'{label}.log').open('w') as log:
            proc=subprocess.Popen(argv,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(ROOT/'commands'/f'{label}_process.json',dict(pid=proc.pid,PGID=os.getpgid(proc.pid),UID=os.getuid(),log=str(ROOT/'logs'/f'{label}.log')))
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);code=124
            finally:
                remaining=[]
                for p in Path('/proc').iterdir():
                    try:
                        if p.name.isdigit() and p.stat().st_uid==os.getuid() and os.getpgid(int(p.name))==proc.pid:remaining.append(int(p.name))
                    except (ProcessLookupError,FileNotFoundError,PermissionError):pass
                if remaining:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    time.sleep(.5)
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                proc.wait()
    finally:
        end=time.monotonic()+15
        while active(gpus) and time.monotonic()<end:time.sleep(.2)
        write(ROOT/'commands'/f'{label}_completion.json',dict(code=code,wall_s=time.time()-start,GPUs=gpus))
        write(ROOT/'commands'/f'{label}_GPU_after.json',GPU_telemetry(gpus))
        if restore_after:restore(gpus)
    print('DONE',label,code,round(time.time()-start,1),flush=True)
    assert code==0,f'Inspect logs/{label}.log'
