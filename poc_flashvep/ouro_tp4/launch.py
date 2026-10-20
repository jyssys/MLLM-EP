"""Persist exact commands and failures. Signal only this launch's process group."""
import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from .contract import ROOT,PROJECT,PYTHON,write,sha


def execute(label,argv,timeout=7200):
    from poc_flashvep.ouro_request_pp.gpu_lease import active
    occupied=active()
    assert not occupied,f'Allowed GPUs are occupied. No job was started: {occupied}'
    log=ROOT/'logs'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
    assert not log.exists(),str(log)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN')
    if '/bin/python' in argv[0]:env['PATH']=str(Path(argv[0]).parent)+os.pathsep+env['PATH']
    write(ROOT/'commands'/f'{label}.json',dict(argv=argv,cwd=str(PROJECT),
        env={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','HF_HUB_OFFLINE','NCCL_DEBUG','PATH']},
        source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'poc_flashvep/ouro_tp4').glob('*.py')}))
    start=time.time()
    with log.open('w') as out:
        process=subprocess.Popen(argv,cwd=PROJECT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    # External engines can leave worker children after their driver returns.
    # This process created this new session. Never signal a different session.
    remaining=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            pid=int(entry.name)
            if entry.stat().st_uid==os.getuid() and os.getpgid(pid)==process.pid:remaining.append(pid)
        except (ProcessLookupError,FileNotFoundError,PermissionError):pass
    if remaining:
        os.killpg(process.pid,signal.SIGTERM)
        time.sleep(.5)
        still_alive=[]
        for pid in remaining:
            try:
                if os.getpgid(pid)==process.pid and (Path('/proc')/str(pid)).stat().st_uid==os.getuid():still_alive.append(pid)
            except (ProcessLookupError,FileNotFoundError):pass
        if still_alive:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    write(ROOT/'commands'/f'{label}_completion.json',dict(returncode=code,wall_s=time.time()-start,pid=process.pid))
    if remaining:write(ROOT/'commands'/f'{label}_child_cleanup.json',dict(PGID=process.pid,pids=remaining,signal='SIGTERM'))
    print('DONE',label,code,round(time.time()-start,1),flush=True)
    assert code==0,f'{label} failed. Inspect {log}'


def tp(label,args):
    return execute(label,[PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',
                         '-m','poc_flashvep.ouro_tp4.tp_driver',*args,'--label',label])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['native','operators','correctness','graph_correctness','pilot'])
    a=p.parse_args()
    if a.phase=='native':execute('native_reference_v1',[PYTHON,'-u','-m','poc_flashvep.ouro_tp4.native_reference'])
    elif a.phase=='operators':tp('tp_operators_v1',['--mode','operators'])
    elif a.phase=='correctness':tp('tp_correctness_eager_v1',['--mode','correctness'])
    elif a.phase=='graph_correctness':tp('tp_correctness_graph_v1',['--mode','correctness','--graphs','--smoke'])
    else:tp('tp_pilot_v1',['--mode','benchmark','--graphs','--indices','0,5'])
