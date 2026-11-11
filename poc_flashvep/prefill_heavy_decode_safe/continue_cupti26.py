"""Wait for the fresh Native gate, then run the pinned-library controls."""
import json,time
from .common import *


def main():
    initialize();folder=ROOT/'raw/native_huginn_clean_recovery_state32';until=time.monotonic()+1200
    while time.monotonic()<until:
        q=folder/'qualification_with_frozen_reference.json';r=folder/'dispatcher_result.json'
        if q.exists():assert json.loads(q.read_text())['campaign_gate_passed'];break
        if r.exists() and not json.loads(r.read_text()).get('complete'):raise RuntimeError('Native state gate failed')
        time.sleep(10)
    else:raise TimeoutError('Fresh Native gate did not finish')
    plan=json.loads((ROOT/'state/cupti26_controls.json').read_text())
    jobs=plan['jobs'][:3]+[plan['Huginn_state8']]+plan['jobs'][3:]
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.prefill_heavy_decode_safe.sequence',*jobs],cwd=REPO).returncode
    write(ROOT/'state/cupti26_controller_complete.json',dict(time=time.time(),returncode=code))
    assert code==0,'Inspect the specific failed control before further expansion'


if __name__=='__main__':main()
