"""Bounded fixed/refill/no-refill diagnostics. These are not new quality claims."""
import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    status=ROOT/'outputs/cost_surface_v1/guard_status.json'
    while not status.exists():time.sleep(2)
    assert read(status)['success']
    manifest=ROOT/'artifacts/observer_frozen.json'
    for mode,extra in [('fixed',['--fixed']),('refill',[]),('no_refill',['--no-refill'])]:
        subprocess.run([REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.gpu_session',
            '--gpus','7','--output',f'outputs/ablation_{mode}_v1','--timeout','1200','--',PYTHON,
            '-m','poc_decode_first_depth_aware_dp_v3.scripts.probe','--graphs','--batch','4',
            '--manifest',str(manifest),'--cap','64',*extra],cwd=PROJECT,check=True)

if __name__=='__main__':main()
