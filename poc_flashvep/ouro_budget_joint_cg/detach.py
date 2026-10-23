"""Persist the next coordinator across tool-session interruptions."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,REPO,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('module',choices=['pilot_controller']);args=parser.parse_args()
    target=ROOT/'state'/f'{args.module}_process.json'
    if target.exists():
        previous=__import__('json').loads(target.read_text());proc=Path('/proc')/str(previous['pid'])
        if proc.exists() and f'poc_flashvep.ouro_budget_joint_cg.{args.module}' in (proc/'cmdline').read_bytes().decode().split('\0'):
            print('ALREADY_RUNNING',previous['pid']);return
    log_path=ROOT/'logs'/f'{args.module}_{int(time.time())}.log'
    with log_path.open('a') as log:
        child=subprocess.Popen(['/home/esjung/anaconda3/bin/python','-u','-m',f'poc_flashvep.ouro_budget_joint_cg.{args.module}'],
            cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(target,dict(pid=child.pid,module=args.module,wall_s=time.time(),log=str(log_path),uid=os.getuid()))
    print('DETACHED_COORDINATOR',child.pid,log_path)

if __name__=='__main__':
    main()
