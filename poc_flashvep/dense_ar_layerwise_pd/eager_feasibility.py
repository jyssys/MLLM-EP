"""Use explicit BF16 operator boundaries without changing model weights."""
import fcntl,json
from .common import *
from .prepare import base,save
from .run import execute
from .analyze import compare
from .state_qualification import state_compare,events


def qualify(n,reference):
    cfg=json.loads((ROOT/'configs'/(n+'.json')).read_text());mode=cfg.get('layer_mode')
    checks=[compare(reference,n)]
    if cfg['arm']!='DP2':
        checks.append(state_compare(reference,n,'prefill',cfg['layer_cut'] if mode=='MIGRATE' else None))
        if mode=='MIGRATE':checks.append(state_compare(reference,n,'decode'))
    audits={}
    if mode=='MIGRATE':
        for role in ('prefill','decode'):
            rows=[x for x in events(n,role) if x['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and x.get('measured')]
            wanted=list(range(cfg['layer_cut'])) if role=='prefill' else list(range(cfg['layer_cut'],36))
            seen={f'q{i}':0 for i in range(cfg['count'])};ok=bool(rows)
            for row in rows:
                ok=ok and row['actual']==(wanted if row['prefilling'] else list(range(36)))
                if row['prefilling']:
                    for rid in row['requests']:seen[rid]+=1
            audits[role]=dict(passed=ok and all(v==1 for v in seen.values()),prefill_counts=seen)
    lifetimes={}
    for role in ('prefill','decode') if cfg['arm']!='DP2' else ():
        path=ROOT/'raw'/n/f'KV_lifetime_{role}.json'
        kv=json.loads(path.read_text());closes=[x for x in events(n,role) if x['event']=='HIDDEN_GPU_POOL_CLOSE']
        lifetimes[role]=dict(passed=kv['passed'] and len(closes)==1 and closes[0]['all_ACKed'],KV=kv)
    passed=all(x['passed'] for x in checks) and all(x['passed'] for x in audits.values()) and all(x['passed'] for x in lifetimes.values())
    result=dict(name=n,reference=reference,passed=passed,checks=checks,actual_execution=audits,lifetimes=lifetimes,
        evidence='CORRECTNESS_ONLY',family=cfg['ar_execution_family'],scope='All weights and Attention remain original BF16. Exact tokens and bitwise queried valid KV/Hidden are required within this family. Compiled-family token equality is reported separately, not assumed.')
    write(ROOT/'analysis'/(n+'_eager_gate.json'),result);print('EAGER_GATE',n,passed,flush=True)
    return result


def main():
    jobs=[]
    for count in (8,32):
        modes=[('reference','STATE_REFERENCE',32,'PD_G1')]
        if count==8:modes.append(('local_full',None,32,'DP2'))
        modes +=[(f'c{k}','MIGRATE',k,'PD_G1') for k in (35,34,32,28)]
        for label,mode,cut,arm in modes:
            c=json.loads((ROOT/'configs'/f'layer_qwen8_L8000_O16_state_reference_c36_N{count}_gpu_v6.json').read_text())
            c.update(name=f'eager_qwen8_L8000_O16_N{count}_{label}_qual_v16',
                ar_enforce_eager=True,ar_execution_family='BI_EAGER_NO_GRAPH',layer_mode=mode,
                layer_cut=cut,arm=arm,partial_KV_wire=mode=='MIGRATE',separate_suffix_runner=mode=='MIGRATE',
                verify_layer_execution=mode=='MIGRATE',execution_events=True,batch_commands=True,
                state_capture=arm!='DP2',retain_fp32_boundary_residual=False,
                phase='EXPLICIT_BF16_BOUNDARY_NATIVE_AND_LAYER_QUALIFICATION')
            if arm=='DP2':c.update(hidden_transport=None,cohort_barrier=False,serialize_D_imports=False)
            jobs.append(save(c))
    write(ROOT/'state/eager_correctness_plan.json',dict(jobs=jobs,scope='Separate explicit-BF16 execution family. No precision or SLO relaxation. Optimized compiled Native remains an external strong runtime control.'))
    refs={};results=[];failed=set()
    try:
        for n in jobs:
            cfg=json.loads((ROOT/'configs'/(n+'.json')).read_text());count=cfg['count'];cut=cfg['layer_cut']
            if cfg['layer_mode']=='MIGRATE' and count==32 and cut in failed:continue
            if not execute(n,restore=False):break
            if cfg['layer_mode']=='STATE_REFERENCE':
                refs[count]=n
                result=dict(name=n,passed=True,kind='REFERENCE_PENDING_INDEPENDENT_CHECK')
                if count==32:result['cross_N8_output']=compare(n,refs[8],True)
                result['compiled_family_output_comparison']=compare('native_qwen8_L8000_O32_reference32_s0',n,True)
            else:
                result=qualify(n,refs[count])
                if not result['passed']:
                    if cfg['arm']=='DP2':break
                    failed.add(cut)
            results.append(result);write(ROOT/'state/eager_correctness_progress.json',dict(results=results))
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
