import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    status=ROOT/'outputs/hindsight_confirmation_v1/guard_status.json'
    while not status.exists():time.sleep(2)
    assert read(status)['success']
    subprocess.run([REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.confirm_hindsight',
        '--concurrency','64','--tag','c64'],cwd=PROJECT,check=True)

if __name__=='__main__':main()
