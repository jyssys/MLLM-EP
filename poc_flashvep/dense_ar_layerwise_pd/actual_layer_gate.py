"""Reject output-equivalent recomputation using backend invocation receipts."""
import json
from .common import *
from .analyze import compare,records
from .state_qualification import state_compare,events,states


def gate(name,reference=None,output_reference=None,receipt_suffix='actual_layer_gate'):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());count=cfg['count'];cut=cfg['layer_cut']
    if reference is None:reference=f'layer_qwen8_L8000_O16_state_reference_c36_N{count}_gpu_v6'
    checks=[compare(output_reference or reference,name),state_compare(reference,name,'prefill',cut),state_compare(reference,name,'decode')]
    numerical=None
    if not checks[-1]['passed'] and cfg.get('numerical_native_reference'):
        a=states(reference,'prefill');b=states(name,'decode');rows=events(name,'decode')
        refs={x['external_request_id']:x for x in rows if x['event']=='FULL_NATIVE_REFERENCE_STATE' and x['external_request_id'] in a}
        nums=[x for x in rows if x['event']=='NUMERIC_SUFFIX_REFERENCE' and x.get('measured')]
        coverage=[rid for x in nums for rid in x['requests']]
        prefix_exact=len(b)==count and all(b[rid]['KV'][:cut]==a[rid]['KV'][:cut] for rid in b)
        reference_exact=len(refs)==count and all(refs[rid]['KV']==a[rid]['KV'] and refs[rid]['final_hidden_SHA256']==a[rid]['final_hidden_SHA256'] for rid in refs)
        numeric_pass=len(coverage)==count and len(set(coverage))==count and all(x['passed'] and x['restoration_pass'] and x['actual_reference_layers']==list(range(36)) for x in nums)
        numerical=dict(passed=prefix_exact and reference_exact and numeric_pass,
            prefix_bitwise_exact=prefix_exact,independent_full_reference_bitwise_exact=reference_exact,
            suffix_KV_and_hidden_tolerance_pass=numeric_pass,atol=.02,rtol=.01,
            extra_reference_layers=count*36,evidence='CORRECTNESS_ONLY_NUMERIC_NOT_BITWISE')
    work={};lifetimes={}
    for role in ('prefill','decode'):
        rows=events(name,role);audits=[x for x in rows if x['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and x.get('measured')]
        expected_prefill=list(range(cut)) if role=='prefill' else list(range(cut,36))
        prefills={f'q{i}':0 for i in range(count)};decodes=dict(prefills)
        ok=bool(audits)
        for row in audits:
            wanted=expected_prefill if row['prefilling'] else list(range(36))
            ok=ok and row['actual']==wanted and row['passed']
            for rid in row['requests']:
                assert rid in prefills
                (prefills if row['prefilling'] else decodes)[rid]+=1
        ok=ok and all(v==1 for v in prefills.values())
        ok=ok and (all(v==0 for v in decodes.values()) if role=='prefill' else all(v==cfg['cap']-1 for v in decodes.values()))
        work[role]=dict(passed=ok,measured_backend_invocations=len(audits),prefill_counts=prefills,Decode_counts=decodes)
        p=ROOT/'raw'/name/f'KV_lifetime_{role}.json';kv=json.loads(p.read_text()) if p.exists() else dict(passed=False)
        closes=[x for x in rows if x['event']=='HIDDEN_GPU_POOL_CLOSE']
        lifetimes[role]=dict(KV=kv,Hidden_ACK_pass=len(closes)==1 and closes[0]['all_ACKed'])
    state_pass=all(x['passed'] for x in checks) or (all(x['passed'] for x in checks[:2]) and numerical and numerical['passed'])
    passed=bool(state_pass) and all(x['passed'] for x in work.values()) and all(x['KV']['passed'] and x['Hidden_ACK_pass'] for x in lifetimes.values())
    result=dict(name=name,passed=passed,state_reference=reference,output_reference=output_reference or reference,
        checks=checks,numerical_qualification=numerical,actual_layer_execution=work,lifetimes=lifetimes,
        evidence='CORRECTNESS_ONLY',scope='Actual Native Attention backend invocations, exact outputs and queried valid state, normal ACK/KV cleanup. No claim of cancellation or GPU failure recovery.')
    write(ROOT/'analysis'/(name+'_'+receipt_suffix+'.json'),result)
    print('ACTUAL_LAYER_GATE',name,passed,flush=True)
    return result


if __name__=='__main__':
    import sys
    gate(sys.argv[1])
