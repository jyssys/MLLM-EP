"""Bounded sequential GPU jobs. Each guard restores idle project burners."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    deadline=time.monotonic()+3000
    marker=ROOT/'results/native_diagnostics_v2/guard_status.json'
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Initial diagnostic did not finish')
        time.sleep(3)
    assert read(marker)['success'], 'Preserve the failed run before changing the campaign'
    jobs=[('hf454_fidelity_v1','4',[CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_hf_fidelity'])]
    jobs.extend((f'native_fidelity_TP{tp}_v1',','.join(map(str,GPUS[:tp])),
        [PYTHON,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_native_fidelity','--tp',str(tp)]) for tp in [1,2,4])
    progress=[]
    for label,mask,command in jobs:
        started=time.time()
        code=subprocess.call([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
            '--gpus',mask,'--output','results/'+label,'--timeout','7200','--']+command,cwd=PROJECT)
        progress.append(dict(job=label,exit_code=code,start_epoch=started,end_epoch=time.time()))
        write(ROOT/'results/fidelity_campaign_progress.json',progress)
        if code:
            raise RuntimeError('GPU job failed: '+label)
    write(ROOT/'results/fidelity_campaign_complete.json',dict(success=True))

if __name__=='__main__':
    main()
