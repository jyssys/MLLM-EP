"""Run registered stages serially, with parallel workers inside each stage."""
import argparse
import subprocess
import sys
import time
from .common import ROOT,REPO,write

def main():
    p=argparse.ArgumentParser();p.add_argument('stages',nargs='+');a=p.parse_args()
    for stage in a.stages:
        path=ROOT/'plans'/f'{stage}.json'
        assert path.exists(),path
        write(ROOT/'research_state.json',dict(stage=stage,status='RUNNING',updated_wall=time.time(),
            resume=f'python -m poc_flashvep.ouro_huginn_g_mixed.campaign {" ".join(a.stages)}'))
        result=subprocess.run([sys.executable,'-m','poc_flashvep.ouro_huginn_g_mixed.dispatch',
            '--manifest',str(path),'--manage-burners'],cwd=REPO)
        write(ROOT/'research_state.json',dict(stage=stage,status='DONE' if result.returncode==0 else 'FAILED',
            returncode=result.returncode,updated_wall=time.time()))
        assert result.returncode==0,(stage,result.returncode)

if __name__=='__main__':main()
