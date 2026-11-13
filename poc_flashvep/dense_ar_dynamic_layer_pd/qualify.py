"""Qualify every suffix and the request-specific cut protocol before serving."""
import fcntl,re
from .common import *
from .prepare import base
from .run import execute
from .analyze import compare,records,percentile


def events(name,role):
    folder=ROOT/'raw'/name
    return [row for path in folder.glob(f'extension_{role}_*.jsonl') for row in records(path)]


def state_gate(name,count,mixed=False):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    checks=[];dest=events(name,'decode');source=events(name,'prefill')
    numerical=[r for r in dest if r['event']=='NUMERIC_SUFFIX_REFERENCE' and r.get('measured')]
    if cfg.get('numerical_native_reference'):
        checks.append(dict(check='INDEPENDENT_FULL_REFERENCE_STATE',passed=bool(numerical) and all(r['passed'] for r in numerical),batches=len(numerical)))
    captures=[r for r in dest if r['event']=='CAPTURED_FULL_GRAPH_LAYERS']
    checks.append(dict(check='FULL36_GRAPH_DEFINITION',passed=bool(captures) and all(r['passed'] for r in captures)))
    commits=records(ROOT/'raw'/name/'actual_cut_commits.jsonl')
    cuts={rid:r['cut'] for r in commits for rid in r['requests']} if mixed else {f'q{i}':cfg['layer_cut'] for i in range(count)}
    actual_P={rid:[] for rid in cuts};actual_suffix={rid:[] for rid in cuts};actual_D={rid:0 for rid in cuts}
    for role,rows in (('prefill',source),('decode',dest)):
        for row in rows:
            if row['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and row.get('measured'):
                for rid in row['requests']:
                    if rid not in cuts:continue
                    cut=cuts[rid]
                    if role=='prefill':actual_P[rid].append(row['actual']);assert row['actual']==list(range(cut))
                    elif row['prefilling'] and cut<36:
                        actual_suffix[rid].append(row['actual']);assert row['actual']==list(range(cut,36))
                    else:actual_D[rid]+=1;assert row['actual']==list(range(36))
            elif row['event']=='FULL_GRAPH_REPLAY' and role=='decode':
                for rid in row['requests']:
                    if rid in cuts:actual_D[rid]+=1
    work=bool(cuts) and len(cuts)==count and all(len(actual_P[rid])==1 and
        len(actual_suffix[rid])==(cuts[rid]<36) and actual_D[rid]==cfg['cap']-1 for rid in cuts)
    checks.append(dict(check='ACTUAL_LAYER_WORK_ONCE',passed=work,cuts=cuts,P=actual_P,D_suffix=actual_suffix,D_full_Decode_steps=actual_D))
    for role in ('prefill','decode'):
        path=ROOT/'raw'/name/f'KV_lifetime_{role}.json';checks.append(dict(check='NATIVE_KV_LIFETIME_'+role,**json.loads(path.read_text())))
        path=ROOT/'raw'/name/f'reservation_lifetime_{role}.json'
        if mixed:checks.append(dict(check='CREDIT_LIFETIME_'+role,**json.loads(path.read_text())))
    if mixed:
        checks.append(dict(check='MULTIPLE_ACTUAL_CUTS',passed=len(set(cuts.values()))>1,distinct_cuts=sorted(set(cuts.values())),
            scope='Forced qualification is not evidence of an online state-aware decision.'))
    result=dict(name=name,evidence='CORRECTNESS_ONLY',checks=checks,passed=all(r['passed'] for r in checks))
    write(ROOT/'analysis'/(name+'_gate.json'),result);return result['passed']


def static_config(layers,count=8,diagnostic=True,serial=False):
    c=base(f'qual_static_Last{layers}_N{count}_v1',count=count,cap=16,dataset='qualification.json')
    c.update(arrival_offsets=[0.]*count,arrival_rate=None,layer_mode='MIGRATE',layer_cut=36-layers,
        hidden_transport='NIXL_GPU',hidden_source_backpressure=True,hidden_verify_hash=diagnostic,
        reuse_prefill_token=False,partial_KV_wire=True,separate_suffix_runner=True,
        verify_layer_execution=diagnostic,numerical_native_reference=diagnostic,state_capture=False,
        execution_events=not diagnostic,calibration_serial=serial,
        phase='EXACT_SUFFIX_STATE_QUALIFICATION' if diagnostic else 'ISOLATED_EVENT_COST_CALIBRATION',timeout=600)
    return c


def dynamic_config(name,count=8):
    c=base(name,count=count,cap=16,dataset='qualification.json')
    c.update(module='dynamic_native',dynamic_layer_extension=True,worker_state_feedback=True,
        layer_mode='MIGRATE',layer_cut=28,partial_KV_wire=True,separate_suffix_runner=True,
        hidden_transport='NIXL_GPU',hidden_source_backpressure=True,reuse_prefill_token=True,
        warmup_count=5,warmup_layer_counts=[0,2,4,8,12],state_capture=False,verify_layer_execution=True,
        numerical_native_reference=True,hidden_verify_hash=True,correctness_only=True,
        verify_native_prefix_KV=True,
        arrival_offsets=[0.]*count,arrival_rate=None,P_group_size=1 if count==8 else 4,
        policy='FORCED_QUALIFICATION',forced_layers=[0,2,4,8,12,0,4,8],
        D_admission_batch=4,max_reserved_requests=16,negotiation_timeout_s=2,timeout=600,
        phase='FORCED_MIXED_CUT_PROTOCOL_CORRECTNESS_NOT_DYNAMIC_PERFORMANCE')
    return c


def main():
    initialize();assert (ROOT/'configs/frozen_SLO_regimes.json').exists(),'Freeze SLOs before candidate GPU evaluation'
    results=[]
    try:
        for layers in (2,4,8,12):
            c=static_config(layers);name=save(c)
            if not execute(name,restore=False):return
            output=compare('qual_native_O200_N8_A_v1',name,True);state=state_gate(name,8)
            row=dict(name=name,output=output,state_pass=state,passed=output['passed'] and state)
            results.append(row);write(ROOT/'state/suffix_qualification_progress.json',dict(results=results))
            print('STATIC_SUFFIX_GATE',layers,row['passed'],flush=True)
            if not row['passed']:return
        for count in (8,32):
            c=dynamic_config(f'qual_forced_mixed_Last024812_N{count}_v1',count)
            if count==32:
                c.update(dataset='discovery.json',numerical_native_reference=True)
                # Use the matching first32 discovery prompts and unchanged seed.
                ref=next(p.stem for p in (ROOT/'configs').glob('coarse_O16_N64_*native16_s1_v1.json'))
            else:ref='qual_native_O200_N8_A_v1'
            name=save(c)
            if not execute(name,restore=False):return
            output=compare(ref,name,True);state=state_gate(name,count,True)
            row=dict(name=name,output=output,state_pass=state,passed=output['passed'] and state)
            results.append(row);write(ROOT/'state/suffix_qualification_progress.json',dict(results=results))
            print('MIXED_PROTOCOL_GATE',count,row['passed'],flush=True)
            if not row['passed']:return
        write(ROOT/'state/mixed_protocol_qualified.json',dict(time=time.time(),results=results,passed=True,
            actual_online_dynamic_decisions='NOT_RUN',scope='Forced mixed-cut correctness only. No performance, Predictor accuracy, or state-aware selection claim.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
