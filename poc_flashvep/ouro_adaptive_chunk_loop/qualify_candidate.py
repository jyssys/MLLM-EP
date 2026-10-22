"""Preserve the red/green G3 scheduler regression before GPU expansion."""
import hashlib
import os
from pathlib import Path
import subprocess
from .common import ROOT,REPO,FORK,PYTHON,write

CANDIDATE=Path('/home/esjung/external/ouro-deadline-g3-candidate-rlt')

def main():
    results=[]
    for name,runtime in [('RED_UNFILTERED_G3',FORK),('GREEN_HOMOGENEOUS_BOUNDARY',CANDIDATE)]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(runtime)+':'+str(REPO))
        commands=[[PYTHON,'-m','unittest','poc_flashvep.ouro_adaptive_chunk_loop.test_frontier'],
            [PYTHON,'-m','pytest','-q',str(runtime/'tests/test_split_prefill.py')]]
        for command in commands:
            p=subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True)
            results.append(dict(lane=name,runtime=str(runtime),command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,
                scope='CPU scheduler/executor regression. Not pretrained accuracy or GPU timing.'))
    assert results[0]['exit_code']!=0,'The regression must reproduce the G3 bug'
    assert all(r['exit_code']==0 for r in results if r['lane']=='GREEN_HOMOGENEOUS_BOUNDARY'),results
    write(ROOT/'g3_boundary_red_green_tests.json',results)
    patch=subprocess.check_output(['git','diff','--binary'],cwd=CANDIDATE,text=True)
    write(ROOT/'qualified_runtime_candidate.json',dict(runtime=str(CANDIDATE),patch=patch,
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),status='CPU_QUALIFIED_GPU_NOT_RUN',
        apply_only_after='All Phase0 jobs and the calibration dispatcher exit. No source mutation during calibration.'))
    print('G3_CPU_RED_GREEN_VERIFIED')

if __name__=='__main__':
    main()
