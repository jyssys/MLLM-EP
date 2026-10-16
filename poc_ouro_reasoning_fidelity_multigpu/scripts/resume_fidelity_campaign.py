"""Use the spec-permitted functioning reference after the archived 4.54 failure."""
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    marker=ROOT/'results/hf454_fidelity_v1/guard_status.json'
    deadline=time.monotonic()+120
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Wait for the preceding guard to restore its burner')
        time.sleep(2)
    assert not read(marker)['success']
    jobs=[('hf457_fidelity_v1','4',[CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_hf_fidelity','--reference-version','4.57.1'])]
    jobs.extend((f'native_fidelity_TP{tp}_v1',','.join(map(str,GPUS[:tp])),
        [PYTHON,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_native_fidelity','--tp',str(tp)]) for tp in [1,2,4])
    progress=[]
    for label,mask,command in jobs:
        started=time.time()
        code=subprocess.call([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
            '--gpus',mask,'--output','results/'+label,'--timeout','7200','--']+command,cwd=PROJECT)
        progress.append(dict(job=label,exit_code=code,start_epoch=started,end_epoch=time.time()))
        write(ROOT/'results/resumed_fidelity_progress.json',progress)
        if code:
            raise RuntimeError('GPU job failed: '+label)
    write(ROOT/'results/fidelity_campaign_complete.json',dict(success=True,
        reference_version='4.57.1',author_recommended_version='4.54.1',version_drift=True))

if __name__=='__main__':
    main()
