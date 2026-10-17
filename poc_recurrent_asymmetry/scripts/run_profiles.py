"""Independent exact-profile workers with per-device burner restoration."""
import subprocess
from poc_recurrent_asymmetry.contract import *


def main():
    assert read(ROOT/'outputs/baseline_v1/status.json')['exact_reference']
    jobs=[]
    for gpu,context in zip(GPUS,[1024,4096,8192,16384]):
        command=[PYTHON,'-m','poc_recurrent_asymmetry.scripts.gpu_session',
            '--gpus',str(gpu),'--output',f'outputs/profile_P{context}_v1','--timeout','14400','--',
            PYTHON,'-m','poc_recurrent_asymmetry.scripts.run_sweep','--context',str(context)]
        jobs.append(subprocess.Popen(command,cwd=PROJECT))
    codes=[job.wait() for job in jobs]
    write(ROOT/'metrics/profile_controller.json',dict(exit_codes=codes,
        note='Four independent exact-reference replicas. Not TP4, PP4, CP4, or a measured serving DP4 layout.'))
    assert all(code==0 for code in codes)


if __name__=='__main__':
    main()
