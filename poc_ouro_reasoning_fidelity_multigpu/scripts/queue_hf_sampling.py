"""Use GPU5 only after its preceding reference guard finishes."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    marker=ROOT/'results/hf457_additional_v1/guard_status.json'
    deadline=time.monotonic()+2700
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Additional HF states did not finish')
        time.sleep(5)
    assert read(marker)['success']
    subprocess.run([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
        '--gpus','5','--output','results/hf457_sampling_v1','--timeout','2400','--',
        CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_hf_sampling'],cwd=PROJECT,check=True)

if __name__=='__main__':
    main()
