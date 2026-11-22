"""Detach a checkpointed campaign process from one interactive tool session."""
import argparse
import os
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');parser.add_argument('module');parser.add_argument('arguments',nargs='*')
    args=parser.parse_args();assert args.module.startswith('poc_flashvep.dense_ar_longbench400_pd.')
    receipt=ROOT/'state'/('background_'+args.label+'.json');assert not receipt.exists()
    path=ROOT/'logs'/('background_'+args.label+'.log')
    command=[str(PYTHON),'-u','-m',args.module]+args.arguments
    with path.open('w') as log:
        environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=environment,
            preexec_fn=lambda:os.sched_setaffinity(0,set(range(160,192))))
    value=dict(pid=child.pid,command=command,start_epoch=time.time(),log=str(path),
        controller_CPU_affinity=list(range(160,192)),
        scope='Owned campaign controller on reserved CPUs with an independent OS session. '
            'GPU jobs still receive their own pair-specific CPUs0–159 in run.py. Checkpoint files and live PID checks remain required.')
    write(receipt,value);print(value,flush=True)


if __name__=='__main__':main()
