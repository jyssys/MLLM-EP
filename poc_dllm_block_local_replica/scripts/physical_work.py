import argparse
import subprocess
from pathlib import Path
from poc_dllm_block_local_replica.contract import PYTHON, new_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    for module,args in [('poc_dllm_ep_comm.scripts.run_dllm_ep4',['--limit','2']),
                        ('poc_dllm_block_local_replica.scripts.p2p',[])]:
        sub=out/('ep4' if 'run_dllm' in module else 'p2p');sub.mkdir()
        subprocess.run([PYTHON,'-m',module,'--output',str(sub),*args],check=True,timeout=780)
    new_json(out/'status.json',dict(ok=True,status='COMPLETE'))


if __name__=='__main__':main()
