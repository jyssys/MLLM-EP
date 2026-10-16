"""Run the small GPU4 diagnostic after the broad fidelity capture releases GPU4."""
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
        '--gpus','4','--output','results/native_calibration_divergence_v1','--timeout','1200','--',
        PYTHON,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_calibration_divergence','--backend','native'],cwd=PROJECT,check=True)

if __name__=='__main__':
    main()
