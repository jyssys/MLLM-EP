"""Wait for a prerequisite without counting wait time as research or measurement."""
import json
import subprocess
import time
from .common import ROOT, REPO, record

def main():
    plan=json.loads((ROOT/'plan_heldout.json').read_text())
    while not all((ROOT/'raw'/j['id']/'complete.json').exists() for j in plan['jobs']):
        failed=[j['id'] for j in plan['jobs'] if (ROOT/'raw'/j['id']/'failure.json').exists()]
        assert not failed,failed
        time.sleep(5)
    for phase,gpus in [('timing','0'),('generation_fidelity','0,1,2,3,4,5,6,7'),('generation_quality','0,1,2,3,4,5,6,7')]:
        record('FOLLOWUP_PHASE_START',phase=phase,gpus=gpus)
        with (ROOT/'logs'/f'dispatch_{phase}.log').open('a') as log:
            subprocess.run(['/home/esjung/anaconda3/bin/python','-m','poc_flashvep.looped_moe_headroom.dispatch',
                str(ROOT/f'plan_{phase}.json'),'--gpus',gpus],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        record('FOLLOWUP_PHASE_END',phase=phase)

if __name__=='__main__':
    main()
