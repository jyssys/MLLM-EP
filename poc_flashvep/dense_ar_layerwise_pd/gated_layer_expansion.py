"""Stop each expansion when exact Output or valid-state qualification fails."""
import fcntl,json
from .common import *
from .run import execute
from .state_qualification import state_compare
from .analyze import compare


def main():
    plan=json.loads((ROOT/'state/layer_expansion_plan.json').read_text());reference=plan['jobs'][0];results=[]
    try:
        for i,name in enumerate(plan['jobs']):
            if not execute(name,restore=False):
                results.append(dict(name=name,passed=False,gate='GPU_EXECUTION'));break
            if i==0:
                prior=json.loads((ROOT/'state/layer_numeric_plan.json').read_text())['jobs'][0]
                checks=[compare(reference,prior,allow_reference_prefix=True),
                    state_compare(reference,prior,'prefill',allow_reference_subset=True)]
            else:
                c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                checks=[compare(reference,name,allow_reference_prefix=True),
                    state_compare(reference,name,'prefill',c['layer_cut'],True),
                    state_compare(reference,name,'decode',None,True)]
            passed=all(x['passed'] for x in checks)
            row=dict(name=name,passed=passed,checks=checks)
            write(ROOT/'analysis'/(name+'_state_gate.json'),row);results.append(row)
            write(ROOT/'state/layer_expansion_progress.json',dict(results=results))
            print('LAYER_GATE',name,passed,flush=True)
            if not passed:break
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
