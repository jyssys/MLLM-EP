"""Keep failed full-prompt cuts out of serving evaluation."""
import fcntl,json
from .common import *
from .run import execute
from .analyze import compare
from .state_qualification import state_compare,events


def main():
    plan=json.loads((ROOT/'state/full_prompt_correctness_plan.json').read_text());results=[];references={};failed=set()
    old='native_qwen8_L8000_O32_reference32_s0'
    try:
        for name in plan['jobs']:
            c=json.loads((ROOT/'configs'/(name+'.json')).read_text());count=c['count'];cut=c['layer_cut']
            if c['layer_mode']=='MIGRATE' and count==32 and cut in failed:continue
            if not execute(name,restore=False):
                results.append(dict(name=name,passed=False,gate='GPU_EXECUTION'));break
            if c['layer_mode']=='STATE_REFERENCE':
                checks=[compare(old,name,True)];references[count]=name
                if count==32:checks.append(state_compare(name,references[8],'prefill',None,True))
            else:
                reference=references[count]
                checks=[compare(reference,name),state_compare(reference,name,'prefill',cut),state_compare(reference,name,'decode')]
            closes={r:[x for x in events(name,r) if x['event']=='HIDDEN_GPU_POOL_CLOSE'] for r in ('prefill','decode')}
            lifecycle=all(len(x)==1 and x[0]['all_ACKed'] for x in closes.values())
            passed=all(x['passed'] for x in checks) and lifecycle
            row=dict(name=name,passed=passed,checks=checks,hidden_lifecycle=closes,
                layer_work='STRUCTURAL_ACCOUNTING_FROM_EXECUTED_STATIC_SPAN_AND_COMPLETE_VALID_KV')
            write(ROOT/'analysis'/(name+'_state_gate.json'),row);results.append(row)
            write(ROOT/'state/full_prompt_correctness_progress.json',dict(results=results))
            print('FULL_PROMPT_GATE',name,passed,flush=True)
            if not passed:
                if c['layer_mode']=='STATE_REFERENCE':break
                failed.add(cut)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
