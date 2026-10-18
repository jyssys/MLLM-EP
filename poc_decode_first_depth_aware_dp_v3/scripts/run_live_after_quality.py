"""Keep the GPU phases sequential and preserve every failed run."""
import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    required=[ROOT/f'outputs/quality8192_part{i}_v1/guard_status.json' for i in range(4)]
    required.append(ROOT/'outputs/observer_eager_clean_v1/guard_status.json')
    while not all(p.exists() for p in required):time.sleep(2)
    assert all(read(p)['success'] for p in required)
    assert read(ROOT/'outputs/admission_smoke_v1/status.json')['success']
    for mode in ['trace','clean']:
        command=[REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.gpu_session',
            '--gpus','4,5,6,7','--output',f'outputs/live_{mode}_v1','--timeout','10800','--',
            REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.live_dp',
            '--plan',str(ROOT/f'artifacts/live_{mode}_plan.json')]
        if mode=='trace':command.append('--trace')
        subprocess.run(command,cwd=PROJECT,check=True)
    command=[REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.analyze_live',
        '--trace-run',str(ROOT/'outputs/live_trace_v1'),'--clean-run',str(ROOT/'outputs/live_clean_v1')]
    subprocess.run(command,cwd=PROJECT,check=True)

if __name__=='__main__':main()
