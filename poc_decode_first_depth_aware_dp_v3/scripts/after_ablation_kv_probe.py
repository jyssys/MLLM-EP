import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    status=ROOT/'outputs/ablation_no_refill_v1/guard_status.json'
    while not status.exists():time.sleep(2)
    assert read(status)['success']
    subprocess.run([REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.gpu_session',
        '--gpus','7','--output','outputs/batched_kv_probe_v1','--timeout','1200','--',PYTHON,
        '-m','poc_decode_first_depth_aware_dp_v3.scripts.batched_kv_probe'],cwd=PROJECT,check=True)

if __name__=='__main__':main()
