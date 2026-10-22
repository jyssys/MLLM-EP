"""Retain software-test commands, output, and exit status."""
import os
import subprocess
import time
from .common import ROOT,REPO,FORK,PYTHON,write

def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(FORK)+':'+str(REPO))
    commands=[
        [PYTHON,'-m','unittest','poc_flashvep.ouro_adaptive_chunk_loop.test_contract','poc_flashvep.ouro_adaptive_chunk_loop.test_frontier','poc_flashvep.ouro_adaptive_chunk_loop.test_no_future_leakage'],
        [PYTHON,'-m','pytest','-q',str(FORK/'tests/test_split_prefill.py')]]
    result=[]
    for cmd in commands:
        started=time.time()
        p=subprocess.run(cmd,cwd=REPO,env=env,text=True,capture_output=True)
        result.append(dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_start=started,wall_end=time.time(),
            scope='CPU_SOFTWARE_CONTRACT_TESTS. Tiny random fixtures are not pretrained quality evidence.'))
    write(ROOT/'cpu_tests.json',result)
    assert all(r['exit_code']==0 for r in result),result
    print('CPU_SUITE_PASS',len(result),'commands')

if __name__=='__main__':
    main()
