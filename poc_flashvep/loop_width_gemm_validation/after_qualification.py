"""Start serial controls only after every original runtime gate passes."""
import os,subprocess
from .common import *


def main():
    initialize();previous=json.loads((ROOT/'state/runtime_qualification_controller.json').read_text())
    while True:
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:break
        assert time.time()<LONG_START_CUTOFF,'The qualification controller exceeded the campaign start bound'
        time.sleep(5)
    progress=json.loads((ROOT/'state/runtime_progress.json').read_text())
    assert progress['closed'] and [r['name'] for r in progress['results']]==previous['jobs']
    assert all(r['returncode']==0 and r['qualification']['campaign_gate_passed'] for r in progress['results'])
    write(ROOT/'state/primary_runtime_qualification_completed.json',progress)
    plan=json.loads((ROOT/'state/serial_cost_plan.json').read_text())
    jobs=plan['qualification']+plan['jobs']
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.loop_width_gemm_validation.sequence',*jobs],cwd=REPO).returncode
    progress=json.loads((ROOT/'state/runtime_progress.json').read_text())
    complete=progress['closed'] and [r['name'] for r in progress['results']]==jobs and all(r['returncode']==0 for r in progress['results'])
    write(ROOT/'state/serial_chain_closed.json',dict(returncode=code,complete=complete,time=time.time()))
    return 0 if complete and code==0 else 2


if __name__=='__main__':raise SystemExit(main())
