"""Bounded sequential continuation; never runs two GPU jobs concurrently."""
import json
import os
import subprocess
import time
from poc_multibd_moe_ep.contract import *


def main():
    deadline=time.monotonic()+5400
    status=ROOT/'artifacts/calibration_session/guard_status.json'
    while not status.exists():
        if time.monotonic()>deadline:raise TimeoutError('Calibration watchdog completion not observed')
        time.sleep(5)
    assert json.loads(status.read_text())['success'],'Calibration failure preserved; do not retry blindly'
    time.sleep(5)  # Let the restored supervisor reach its stable argv/state.
    for backend in ['ep4','tp4']:
        command=[PYTHON,'-m','poc_multibd_moe_ep.scripts.run_guarded',
            '--output',f'artifacts/confirmation_{backend}_session','--timeout','10800','--',
            PYTHON,'-m','poc_multibd_moe_ep.scripts.run_confirmation','--backend',backend]
        subprocess.run(command,check=True,cwd=PROJECT,env=os.environ)
        time.sleep(5)
    subprocess.run([PYTHON,'-m','poc_multibd_moe_ep.scripts.analyze_conversion'],check=True,cwd=PROJECT)
    subprocess.run([PYTHON,'-m','poc_multibd_moe_ep.scripts.evaluate_quality'],check=True,cwd=PROJECT)


if __name__=='__main__':main()
