"""Finite owned GPU jobs with approved burner restoration."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import ROOT,REPO,PYTHON,resources,write,digest,experiment_deadline


def run(name,gpu,module,args,seconds):
    assert module.startswith('poc_flashvep.mllm_epd_prefill_heavy_token_split.')
    owner=resources();owner.stop([gpu]);child=None;data={};stop=False
    def terminate(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    command=[str(PYTHON),'-m',module,*args]
    env=dict(os.environ,PATH=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH',''),
        CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_BATCH_INVARIANT='0')
    record=ROOT/'state'/('gpu_job_'+name+'.json')
    assert not record.exists(),'Do not overwrite a GPU job'
    started=time.time()
    try:
        with (ROOT/'logs'/(name+'.log')).open('ab') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        data=dict(gpu=gpu,pid=child.pid,sid=os.getsid(child.pid),guardian=os.getpid(),
            command=command,started=started,finished=False,source_SHA256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
        write(record,data)
        while child.poll() is None and time.time()<min(started+seconds,experiment_deadline()) and not stop:time.sleep(.5)
    finally:
        if child is not None and child.poll() is None:
            p=Path('/proc')/str(child.pid)
            assert p.stat().st_uid==os.getuid() and os.getsid(child.pid)==child.pid
            assert module.encode() in (p/'cmdline').read_bytes()
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
        data.update(finished=True,finish_time=time.time(),returncode=child.poll() if child else None)
        write(record,data)
        until=time.time()+45
        while any(r['gpu']==gpu for r in owner.processes()) and time.time()<until:time.sleep(.5)
        if not any(r['gpu']==gpu for r in owner.processes()):owner.restore([gpu])


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--module',required=True);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('arguments',nargs=argparse.REMAINDER);args=p.parse_args()
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    run(args.name,args.gpu,args.module,arguments,args.seconds)


if __name__=='__main__':main()
