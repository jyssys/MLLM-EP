"""Sequential guarded physics. Quality must release all four GPUs first."""
import subprocess
import sys
import time
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    deadline=time.monotonic()+14400
    while True:
        marker=ROOT/'results/quality_native_v3/status.json'
        failure=ROOT/'results/quality_native_v3/guard_status.json'
        if marker.exists() and read(marker).get('success') and failure.exists() and read(failure).get('success'):break
        if failure.exists() and not read(failure).get('success'):raise RuntimeError('Quality run failed. Preserve logs; do not overlap another job.')
        if time.monotonic()>deadline:raise TimeoutError('Quality wait exceeded the bounded campaign limit')
        time.sleep(20)
    # Every job owns the guard until cleanup restores burners. Never overlap jobs.
    jobs=[(4,4,0)]+[(4,4,B) for B in [512,2048,4096,256,1024]]
    jobs += [(TP,1,B) for TP in [1,2] for B in [512,2048,4096,256,1024]]
    subprocess.run([CPU,'-m','poc_looped_exact_depth_chunk_pipeline.scripts.score_public_reasoning'],cwd=PROJECT,check=True)
    quality=read(ROOT/'results/native_quality_r4.json')['quality_gate']
    if quality!='QUALITY_BASELINE_PASS':jobs=jobs[:1]
    write(ROOT/'results/physics_plan.json',dict(quality_gate=quality,expected_configurations=len(jobs),jobs=jobs,
        scope='Weak native quality stops the broad pipeline economics sweep. One fresh native hardware baseline and tiny reference/transfer checks remain diagnostic only.'))
    ledger=[]
    for TP,R,B in jobs:
        name=f'TP{TP}_R{R}_chunk{B}_v1';directory=ROOT/'results/physics'/name
        assert not directory.exists(),'Preserve previous attempts'
        cmd=[sys.executable,'-m','poc_looped_exact_depth_chunk_pipeline.scripts.gpu_session',
            '--gpus',','.join(map(str,GPUS[:TP])),'--timeout','2400','--output','results/physics/'+name,
            '--',PYTHON,'-m','poc_looped_exact_depth_chunk_pipeline.scripts.run_native_baselines',
            '--tp',str(TP),'--r',str(R),'--chunk',str(B)]
        result=subprocess.run(cmd,cwd=PROJECT,check=False)
        ledger.append(dict(config=name,exit_code=result.returncode));write(ROOT/'results/physics_progress.json',ledger)
        if result.returncode:raise RuntimeError('First physical failure stops this queue. No unbounded retry.')
    write(ROOT/'results/physics_completion.json',dict(success=True,completed=len(ledger),quality_gate=quality))


if __name__=='__main__':main()
