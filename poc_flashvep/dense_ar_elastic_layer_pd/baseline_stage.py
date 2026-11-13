"""Stop each baseline stage if its exact output or state contract fails."""
import fcntl
from .common import *
from .run import execute
from .analyze import compare,records,metrics


def previous_output(reference,candidate):
    old={tuple(r['input_ids']):r for r in records(PREVIOUS/'raw'/reference/'requests.jsonl')}
    new=records(ROOT/'raw'/candidate/'requests.jsonl');failures=[]
    for row in new:
        expected=old.get(tuple(row['input_ids']))
        if expected is None or expected['seed']!=row['seed'] or expected['output_ids'][:len(row['output_ids'])]!=row['output_ids']:
            failures.append(row['sample_id'])
    return dict(compared=len(new),failed_ids=failures,passed=bool(new) and not failures,
        scope='Unchanged predecessor Native input, seed, and greedy token prefix.')


def main():
    initialize();plan=json.loads((ROOT/'state/initial_baseline_plan.json').read_text());results=[]
    try:
        for name in plan['jobs']:
            if not execute(name,restore=False):
                write(ROOT/'state/baseline_stage_progress.json',dict(results=results,failed=name,reason='RUNTIME_FAILURE'));return
            if name.startswith('exact_'):
                count=json.loads((ROOT/'configs'/(name+'.json')).read_text())['count']
                ref='qual_native_O200_N8_A_v1' if count==8 else 'coarse_O16_N64_q4.777_native16_s1_v1'
                output=previous_output(ref,name)
                from ..dense_ar_dynamic_layer_pd import qualify
                qualify.ROOT=ROOT
                state=qualify.state_gate(name,count,mixed=True)
                result=dict(name=name,output=output,state=state,passed=output['passed'] and state)
            elif 'Continuous' in name:
                output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name)
                result=dict(name=name,output=output,metrics=metrics(name),passed=output['passed'])
            else:result=dict(name=name,metrics=metrics(name),passed=True)
            results.append(result);write(ROOT/'state/baseline_stage_progress.json',dict(results=results))
            print('BASELINE_GATE',name,result['passed'],flush=True)
            if not result['passed']:return
        write(ROOT/'state/initial_baselines_qualified.json',dict(time=time.time(),results=results))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
