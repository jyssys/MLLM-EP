"""Stop only assigned own burner workers. Preserve idle-device burners."""
import argparse
import datetime
import os
import signal
import subprocess
import time
from poc_looped_multigpu_kv_placement.contract import *
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor


def contexts(gpus):
    selected={UUIDS[GPUS.index(g)] for g in gpus}
    data=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return [int(line.split(',')[1]) for line in data.splitlines() if line.split(',')[0].strip().removeprefix('GPU-') in selected]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpus',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--timeout',type=int,default=1800);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();gpus=list(map(int,args.gpus.split(',')));assert gpus and set(gpus)<=set(GPUS)
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    workers={};initial=contexts(gpus)
    for pid in initial:
        worker,argv=burner_worker_ancestor(pid)
        proc=Path('/proc')/str(worker)
        words=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        assert int(words[words.index('--gpu')+1]) in gpus
        workers[worker]=dict(argv=words,env=dict(s.decode().split('=',1) for s in (proc/'environ').read_bytes().split(b'\0') if b'=' in s))
    # A restored worker can still load on CPU and have no GPU context yet.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            words=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            if (proc.stat().st_uid==os.getuid() and '/home/esjung/vllm-ep/utilize.py' in words
                    and '--worker' in words and '--gpu' in words
                    and int(words[words.index('--gpu')+1]) in gpus):
                workers[int(proc.name)]=dict(argv=words,env=dict(s.decode().split('=',1)
                    for s in (proc/'environ').read_bytes().split(b'\0') if b'=' in s))
        except (FileNotFoundError,PermissionError):continue
    write(output/'gpu_policy.json',dict(gpus=gpus,original_contexts=initial,original_burner_workers={p:v['argv'] for p,v in workers.items()},
        unused_device_burners_untouched=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    child=None;status=dict(success=False);restored=[]
    def own_job(pid):
        try:
            proc=Path('/proc')/str(pid)
            return proc.stat().st_uid==os.getuid() and ('LOOPED_KV_OUTPUT='+str(output)).encode() in (proc/'environ').read_bytes()
        except (FileNotFoundError,PermissionError):return False
    try:
        for worker in workers:
            try:
                assert os.getpgid(worker)==worker,'Refuse a broad process-group target'
                os.killpg(worker,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.monotonic()+45
        while contexts(gpus) and time.monotonic()<deadline:time.sleep(.5)
        assert not contexts(gpus),'Competing process remains. Do not replace or kill a foreign job.'
        write(output/'clean_start.json',dict(mask=args.gpus,
            GPUs=subprocess.check_output(['nvidia-smi','--id='+args.gpus,
                '--query-gpu=index,uuid,name,memory.total,memory.free,clocks.current.graphics,clocks.current.memory,power.limit,utilization.gpu',
                '--format=csv,noheader'],text=True),
            clock_policy='No clock or power setting changed.',
            all_GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                '--format=csv,noheader'],text=True)))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,LOOPED_KV_OUTPUT=str(output),
            OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',PYTHONPATH=str(PROJECT))
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        env['PATH']=str(Path(command[0]).parent)+':'+env.get('PATH','')
        write(output/'run_contract.json',dict(command=command,mask=args.gpus,
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')}))
        with (output/'console.log').open('w') as log:
            child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait(timeout=args.timeout)
        status=dict(success=code==0,exit_code=code)
    except (Exception,KeyboardInterrupt) as error:status=dict(success=False,error=repr(error))
    finally:
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            for pid in contexts(gpus):
                if own_job(pid):
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            deadline=time.monotonic()+15
            while any(own_job(pid) for pid in contexts(gpus)) and time.monotonic()<deadline:time.sleep(.5)
            for pid in contexts(gpus):
                if own_job(pid):
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            # Context release is asynchronous even after SIGKILL.
            deadline=time.monotonic()+25
            while any(own_job(pid) for pid in contexts(gpus)) and time.monotonic()<deadline:time.sleep(.5)
        deadline=time.monotonic()+15
        while contexts(gpus) and time.monotonic()<deadline:
            # Do not signal an unidentified process merely to restore a burner.
            time.sleep(.5)
        # Live burner output stays outside Git. The exact prior worker environment is private.
        for worker,record in workers.items():
            gpu=int(record['argv'][record['argv'].index('--gpu')+1])
            if contexts([gpu]):continue
            live=ASSETS/'burner_logs';live.mkdir(parents=True,exist_ok=True)
            with (live/f'{output.name}.gpu{gpu}.log').open('a') as log:
                proc=subprocess.Popen(record['argv'],env=record['env'],cwd='/home/esjung/vllm-ep',
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            restored.append(dict(physical_gpu=gpu,pid=proc.pid))
        # The user requests burning on every idle assigned device, including an initially idle device.
        restore_gpus={row['physical_gpu'] for row in restored}
        idle=[gpu for gpu in gpus if gpu not in restore_gpus and not contexts([gpu])]
        if idle:
            live=ASSETS/'burner_logs';live.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,VLLM_UTILIZE_GPUS=','.join(map(str,idle)),CUDA_VISIBLE_DEVICES=','.join(map(str,idle)))
            env.pop('PYTHONPATH',None)
            with (live/f'{output.name}.fallback.log').open('a') as log:
                proc=subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'],env=env,cwd='/home/esjung/vllm-ep',
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            restored.append(dict(physical_gpu=idle,pid=proc.pid,mode='SCOPED_LAUNCHER_FALLBACK'))
        write(output/'guard_status.json',dict(**status,restored_burner_workers=restored))
    print(status,flush=True);raise SystemExit(0 if status['success'] else 1)


if __name__=='__main__':main()
