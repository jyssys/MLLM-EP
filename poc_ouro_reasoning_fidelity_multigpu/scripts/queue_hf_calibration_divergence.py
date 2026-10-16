"""Run the calibration-only reference after the current TP4 job releases all ranks."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    marker=ROOT/'results/native_fidelity_TP4_v1/guard_status.json'
    deadline=time.monotonic()+3600
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('TP4 fidelity did not finish')
        time.sleep(5)
    assert read(marker)['success']
    subprocess.run([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
        '--gpus','6','--output','results/hf_calibration_divergence_v2','--timeout','1200','--',
        CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_calibration_divergence','--backend','hf'],cwd=PROJECT,check=True)

if __name__=='__main__':
    main()
