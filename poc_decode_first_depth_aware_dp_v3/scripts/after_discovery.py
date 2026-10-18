"""Start the missing quality partition after the scoped discovery job exits."""
import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    status=ROOT/'outputs/discovery128_graph_b1_v1/guard_status.json'
    while not status.exists():time.sleep(2)
    assert read(status)['success']
    command=[REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.gpu_session',
        '--gpus','4','--output','outputs/quality8192_part0_v1','--timeout','10800','--',
        PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.probe',
        '--quality','--cap','8192','--part','0','--parts','4']
    write(ROOT/'artifacts/quality_part0_command.json',dict(command=command))
    subprocess.run(command,cwd=PROJECT,check=True)

if __name__=='__main__':main()
