"""Launch four independent numerical-emulation workers on permitted devices."""
import argparse
import os
import subprocess
import sys
from poc_async_recurrent_pp.contract import *


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
            jobs.append((process,log,arm))
        statuses={arm:process.wait() for process,log,arm in jobs}
        write(output/'status.json',dict(exit_codes=statuses,independent_replicas=True,
            not_PP_or_TP=True,physical_gpu_by_arm=dict(zip(['exact','stale','delta','w0'],GPUS))))
        assert all(code==0 for code in statuses.values())
    finally:
        for process,log,arm in jobs:
            if process.poll() is None:
                process.terminate()
            log.close()


if __name__=='__main__':
    main()
