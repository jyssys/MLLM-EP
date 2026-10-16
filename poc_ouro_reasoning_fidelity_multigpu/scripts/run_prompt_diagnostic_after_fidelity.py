"""Advance only to a bounded prompt diagnostic after numerical confirmation."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    deadline=time.monotonic()+14400
    marker=ROOT/'results/fidelity_campaign_complete.json'
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Fidelity gate did not finish within four hours')
        time.sleep(10)
    subprocess.run([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_fidelity','--calibrate-only'],cwd=PROJECT,check=True)
    subprocess.run([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_fidelity'],cwd=PROJECT,check=True)
    summary=read(ROOT/'results/fidelity_summary.json')
    if not (summary['calibration_pass'] and summary['confirmation_pass'] and summary['same_config_short_repeats_pass']):
        write(ROOT/'results/prompt_diagnostic_gate.json',dict(run=False,reason='Numerical or repeat gate needs diagnosis.'))
        return
    code=subprocess.call([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
        '--gpus','4,5,6,7','--output','results/prompt_diagnostic_v1','--timeout','3600','--',
        PYTHON,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_quality','--diagnostic'],cwd=PROJECT)
    write(ROOT/'results/prompt_diagnostic_gate.json',dict(run=True,exit_code=code,
        scope='Eight overlapping diagnostic AIME questions. P1 remains the preregistered primary regardless of score.'))

if __name__=='__main__':
    main()
