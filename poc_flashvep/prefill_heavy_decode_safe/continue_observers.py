"""Serialize observer tests after long-context capacity measurements."""
import json,time
from .common import *


def main():
    initialize();limit=time.monotonic()+1800;progress=ROOT/'state/sequence_progress.json'
    while time.monotonic()<limit:
        if progress.exists():
            value=json.loads(progress.read_text());rows=value.get('results',[])
            if value.get('closed') and rows and rows[0]['name']=='capacity_ouro_L8192_O16_B124_start1':break
        time.sleep(5)
    else:raise TimeoutError('Long-context sequence did not close')
    plan=json.loads((ROOT/'state/observer_qualification_plan.json').read_text())
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.prefill_heavy_decode_safe.sequence',*plan['jobs']],cwd=REPO).returncode
    write(ROOT/'state/observer_controller_complete.json',dict(time=time.time(),returncode=code))


if __name__=='__main__':main()
