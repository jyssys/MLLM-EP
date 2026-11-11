"""Continue only after the declared Native state qualification closes."""
import json,time
from .common import *


def main():
    initialize();plan=json.loads((ROOT/'state/initial_qualification_plan.json').read_text())
    until=time.monotonic()+1200
    while time.monotonic()<until:
        path=ROOT/'state/sequence_progress.json'
        progress=json.loads(path.read_text()) if path.exists() else {}
        if progress.get('closed'):
            assert [r['name'] for r in progress['results']]==plan['jobs'] and all(r['passed'] for r in progress['results']),'Initial qualification failed. Inspect it before any serving expansion.'
            write(ROOT/'state/initial_qualification_completed.json',progress);break
        time.sleep(10)
    else:raise TimeoutError('Initial qualification did not close within its bounded wait')
    jobs=json.loads((ROOT/'state/initial_anchor_controls.json').read_text())['jobs']
    command=['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m','poc_flashvep.prefill_heavy_decode_safe.sequence',*jobs]
    code=subprocess.run(command,cwd=REPO).returncode
    write(ROOT/'state/initial_anchor_controller_complete.json',dict(time=time.time(),returncode=code))
    assert code==0,'A pilot failed. Preserve it and inspect the specific failure.'


if __name__=='__main__':main()
