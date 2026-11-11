"""Run marker-free workload discovery after separate observer qualification."""
import json,time
from .common import *


def main():
    initialize();until=time.monotonic()+2400;receipt=ROOT/'state/observer_controller_complete.json'
    while not receipt.exists():
        assert time.monotonic()<until,'Observer tests did not close'
        time.sleep(5)
    plan=json.loads((ROOT/'state/coarse_discovery_plan.json').read_text())
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.prefill_heavy_decode_safe.sequence',*plan['jobs']],cwd=REPO).returncode
    write(ROOT/'state/coarse_controller_complete.json',dict(time=time.time(),returncode=code))


if __name__=='__main__':main()
