"""Launch four independent numerical-emulation workers on permitted devices."""
import argparse
import os
import subprocess
import sys
import time
from poc_async_recurrent_pp.contract import *
from poc_looped_multigpu_kv_placement.scripts.gpu_session import contexts
from poc_async_recurrent_pp.burner_scope import pending_burner


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cap',type=int,default=2048)
    parser.add_argument('--per-dataset',type=int,default=16)
    parser.add_argument('--expand-source',type=Path)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    jobs=[]
    try:
        for gpu,arm in zip(GPUS,['exact','stale','delta','w0']):
            directory=output/arm
            directory.mkdir(exist_ok=False)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LOOPED_KV_OUTPUT=str(directory))
            command=[sys.executable,'-m','poc_async_recurrent_pp.scripts.eval_quality',
                '--arm',arm,'--cap',str(args.cap),'--per-dataset',str(args.per_dataset)]
            if args.expand_source:
                command+=['--expand-capped-only',str(args.expand_source/arm)]
            log=(directory/'console.log').open('w')
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((process,log,arm,gpu))
        statuses={}
        restored={}
        while len(statuses)<len(jobs):
            for process,log,arm,gpu in jobs:
                code=process.poll()
                if code is None:
                    continue
                statuses[arm]=code
                if gpu in restored or contexts([gpu]):
                    continue
                existing=pending_burner([gpu])
                if existing is None:
                    logs=ASSETS/'burner_logs'
                    logs.mkdir(exist_ok=True,parents=True)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),VLLM_UTILIZE_GPUS=str(gpu))
                    env.pop('PYTHONPATH',None)
                    env.pop('LOOPED_KV_OUTPUT',None)
                    with (logs/(output.name+'_'+arm+'_idle.log')).open('a') as burner_log:
                        burner=subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'],env=env,
                            cwd='/home/esjung/vllm-ep',stdin=subprocess.DEVNULL,
                            stdout=burner_log,stderr=subprocess.STDOUT,start_new_session=True)
                    existing=burner.pid
                restored[gpu]=existing
                write(output/'idle_burners.json',dict(physical_gpu_to_burner_PID=restored,
                    note='Restore each completed arm while other independent accuracy arms continue.'))
            if len(statuses)<len(jobs):
                time.sleep(2)
        write(output/'status.json',dict(exit_codes=statuses,independent_replicas=True,
            not_PP_or_TP=True,physical_gpu_by_arm=dict(zip(['exact','stale','delta','w0'],GPUS))))
        assert all(code==0 for code in statuses.values())
    finally:
        for process,log,arm,gpu in jobs:
            if process.poll() is None:
                process.terminate()
            log.close()


if __name__=='__main__':
    main()
