"""Run the broader context diagnostic after the four-GPU clean lane releases its GPUs."""
import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    status=ROOT/'outputs/live_clean_v1/guard_status.json'
    while not status.exists():time.sleep(2)
    assert read(status)['success']
    subprocess.run([REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.gpu_session',
        '--gpus','7','--output','outputs/cost_surface_v1','--timeout','3600','--',PYTHON,
        '-m','poc_decode_first_depth_aware_dp_v3.scripts.cost_surface'],cwd=PROJECT,check=True)

if __name__=='__main__':main()
