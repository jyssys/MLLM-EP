"""Keep GPU timing serial after the profiler-control sequence."""
import json,time
from .common import *


def main():
    initialize();receipt=ROOT/'state/cupti26_controller_complete.json';limit=time.monotonic()+2400
    while not receipt.exists():
        assert time.monotonic()<limit,'Profiler controller did not close'
        time.sleep(10)
    # A completed failed profiler control does not invalidate marker-free capacity work.
    plan=json.loads((ROOT/'state/capacity_discovery_plan.json').read_text())
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.prefill_heavy_decode_safe.sequence',*plan['jobs']],cwd=REPO).returncode
    write(ROOT/'state/capacity_controller_complete.json',dict(time=time.time(),returncode=code))


if __name__=='__main__':main()
