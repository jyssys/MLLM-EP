"""Record software validation and environment commands without GPU allocation."""
import argparse
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--prose-only',action='store_true');a=p.parse_args()
    commands={
        'packages':[PYTHON,'-m','pip','freeze'],
        'nsys':['nsys','--version'],
        'source_diff_check':['git','-C','/home/esjung/external/ouro-huginn-g-mixed-rlt','diff','--check'],
        'cpu_tests':['env','CUDA_VISIBLE_DEVICES=',
            'PYTHONPATH=/home/esjung/external/ouro-huginn-g-mixed-rlt:/home/esjung/MLLM-EP-github',
            'OMP_NUM_THREADS=1',PYTHON,'-m','pytest','-q',
            'poc_flashvep/ouro_huginn_g_mixed/test_contract.py',
            '/home/esjung/external/ouro-huginn-g-mixed-rlt/tests/test_huginn.py',
            '-k','not gpu and not unsupported_execution'],
    }
    if a.prose_only:
        commands={'report_lint':['/home/esjung/anaconda3/bin/python',
            '/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py',
            *map(str,sorted(ROOT.glob('*.md'))),'--disable','synonym-rotation','--json']}
    for name,command in commands.items():
        started=time.time();result=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
        write(ROOT/'validation'/a.label/f'{name}.json',dict(command=command,returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr,start_wall=started,end_wall=time.time()))

if __name__=='__main__':main()
