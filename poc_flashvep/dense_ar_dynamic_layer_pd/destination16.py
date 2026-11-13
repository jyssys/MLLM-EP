"""Test whether the inherited token budget unnecessarily limits D batching."""
import fcntl
from .common import *
from .prepare import base
from .qualify import static_config,dynamic_config,state_gate
from .online_pilots import serving_config
from .run import execute
from .analyze import compare,metrics,records


def finish(stage,passed,rows,reason=None):
    write(ROOT/'analysis/destination16_results.json',dict(time=time.time(),stage=stage,passed=passed,
        reason=reason,results=rows,scope='A128000-token D budget is legal only after exact state qualification. P remains Batch4. No diagnostic collection delay enters Serving.'))


def state_hashes(name,role):
    values={}
    for p in (ROOT/'raw'/name).glob(f'extension_{role}_*.jsonl'):
        for r in records(p):
            rid=r.get('external_request_id','')
            if r['event']=='VALID_QUERY_STATE' and rid.startswith('q') and rid[1:].isdigit() and r['query_tokens']==8000:
                assert r['query_start']==0
                assert [v['layer'] for v in r['KV']]==list(range(36))
                assert all(v['bytes']==32768000 for v in r['KV'])
                assert rid not in values,'A qualified prompt frontier must occur once'
                values[rid]=dict(KV=r['KV'],final_hidden_SHA256=r['final_hidden_SHA256'])
    return values


def reference_gate(name,reference,targets):
    expected=state_hashes(reference,'prefill');actual=state_hashes(name,'decode')
    failures=[rid for rid in targets if rid not in expected or rid not in actual or expected[rid]!=actual[rid]]
    value=dict(name=name,reference=reference,compared=len(targets),failures=failures,passed=bool(targets) and not failures,
        evidence='CORRECTNESS_ONLY',
        scope='Independent full Native prompt state on P versus completed suffix state on D. Compare all36 valid KV planes and final Hidden by SHA256. This is stricter than the unchanged numerical tolerance. No in-process second full model runs beside the candidate.')
    write(ROOT/'analysis'/(name+'_independent_state_gate.json'),value);return value


def main():
    initialize();results=[]
    old=ROOT/'analysis/destination16_results.json'
    if old.exists() and not (ROOT/'analysis/destination16_results_v1.json').exists():
        write(ROOT/'analysis/destination16_results_v1.json',json.loads(old.read_text()))
    try:
        ref=base('qual_D16_independent_full_state_N32_v2',count=32,cap=16,dataset='discovery.json')
        ref.update(arrival_offsets=[0.]*32,arrival_rate=None,layer_mode='STATE_REFERENCE',state_capture=True,
            state_first_query_only=True,save_state_tensors=False,hidden_transport=None,verify_layer_execution=True,
            D_admission_batch=16,native_ACK_hold=True,phase='INDEPENDENT_NATIVE_FULL_KV_AND_HIDDEN_HASH_REFERENCE')
        if not execute(save(ref),restore=False):finish('INDEPENDENT_REFERENCE',False,results,'RUNTIME_FAILURE');return
        ref_output=compare('coarse_O16_N64_q4.777_native16_s1_v1',ref['name'],True)
        assert ref_output['passed'] and len(state_hashes(ref['name'],'prefill'))==32
        for layers in (2,4,8,12):
            c=static_config(layers,count=16,diagnostic=True)
            c.update(name=f'qual_D16_Last{layers}_Source4_N16_v2',dataset='discovery.json',
                D_token_budget=128000,D_admission_batch=16,D_cohort_collect_ms=30000,
                hidden_destination_slots=16,native_ACK_hold=True,
                numerical_native_reference=False,state_capture=True,state_first_query_only=True,save_state_tensors=False,
                phase='LEGAL_DESTINATION_BATCH16_SUFFIX_STATE_QUALIFICATION_NOT_SERVING')
            if not execute(save(c),restore=False):finish('QUALIFICATION',False,results,'RUNTIME_FAILURE');return
            output=compare('coarse_O16_N64_q4.777_native16_s1_v1',c['name'],True)
            state=state_gate(c['name'],16)
            independent=reference_gate(c['name'],ref['name'],[f'q{i}' for i in range(16)])
            rows=[r for p in (ROOT/'raw'/c['name']).glob('extension_decode_*.jsonl') for r in records(p)]
            widths={len(r['requests']) for r in rows if r['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and r.get('measured') and r['prefilling']}
            passed=output['passed'] and state and independent['passed'] and 16 in widths
            result=dict(name=c['name'],output_gate=output,state_pass=state,independent_state_gate=independent,
                actual_suffix_batches=sorted(widths),passed=passed)
            results.append(result);write(ROOT/'state/destination16_progress.json',dict(results=results))
            print('D16_QUALIFICATION',layers,passed,flush=True)
            if not passed:finish('QUALIFICATION',False,results,'STATE_OR_BATCH_COVERAGE_FAILURE');return
        c=dynamic_config('qual_D16_mixed_Last0_12_N32_v2',32)
        c.update(dataset='discovery.json',D_token_budget=128000,D_admission_batch=16,
            hidden_destination_slots=16,D_cohort_collect_ms=30000,negotiation_timeout_s=180,
            max_reserved_requests=32,forced_layers=[0,12,12,12,12,0,12,12],native_ACK_hold=True,
            numerical_native_reference=False,state_capture=True,state_first_query_only=True,save_state_tensors=False,
            phase='DESTINATION_BATCH16_NATIVE_AND_SUFFIX_OWNERSHIP_NOT_SERVING')
        if not execute(save(c),restore=False):finish('MIXED_QUALIFICATION',False,results,'RUNTIME_FAILURE');return
        output=compare('coarse_O16_N64_q4.777_native16_s1_v1',c['name'],True);state=state_gate(c['name'],32,True)
        targets=[rid for r in records(ROOT/'raw'/c['name']/'actual_cut_commits.jsonl') if r['offloaded_layers'] for rid in r['requests']]
        independent=reference_gate(c['name'],ref['name'],targets)
        rows=[r for p in (ROOT/'raw'/c['name']).glob('extension_decode_*.jsonl') for r in records(p)]
        widths={len(r['requests']) for r in rows if r['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and r.get('measured') and r['prefilling']}
        passed=output['passed'] and state and independent['passed'] and 16 in widths
        results.append(dict(name=c['name'],output_gate=output,state_pass=state,independent_state_gate=independent,
            actual_suffix_batches=sorted(widths),passed=passed))
        if not passed:finish('MIXED_QUALIFICATION',False,results,'STATE_OR_BATCH_COVERAGE_FAILURE');return
        write(ROOT/'state/D16_protocol_qualified.json',dict(passed=True,qualification_names=[r['name'] for r in results],
            scope='Each Last2/4/8/12 suffix executed at actual Batch16. Mixed Native/Last12 ownership also passed. Forced qualification is not Online adaptation.'))
        jobs=[]
        for cap in (16,32):
            for layers in (2,4,8,12):
                c=static_config(layers,count=64,diagnostic=False);b=base('unused',cap=cap,rate=4.777100014491616)
                c.update(name=f'validation_D16_fast_Static{layers}_O{cap}_N64_v1',dataset='discovery.json',cap=cap,
                    request_caps=[cap]*64,arrival_offsets=b['arrival_offsets'],arrival_rate=b['arrival_rate'],
                    arrival_seed=b['arrival_seed'],kv_layout='HND',execution_events=False,D_admission_batch=16,
                    D_token_budget=128000,hidden_destination_slots=16,phase='STRONG_LEGAL_BATCH16_STATIC_VALIDATION')
                jobs.append(c)
            for fraction in (.25,.5):
                jobs.append(serving_config(f'validation_D16_fraction_Static12_f{int(100*fraction)}_O{cap}_N64_v1',
                    cap=cap,policy='STATIC',static_layers=12,migration_fraction=fraction,D_token_budget=128000,
                    phase='BATCH16_STATIC_FRACTION_VALIDATION'))
        for cap in (16,32,64):
            jobs.append(serving_config(f'predictor_D16_COST_BATCH_O{cap}_N64_v1',cap=cap,
                D_token_budget=128000,decode_prediction='BATCH_LOOKUP',decode_prediction_margin_ms=2.,
                phase='LEGAL_BATCH16_PAST_ONLY_COST_POLICY_DISCOVERY'))
        for c in jobs:
            if not execute(save(c),restore=False):finish('SERVING',False,results,'RUNTIME_FAILURE');return
            output=compare(f'coarse_O{c["cap"]}_N64_q4.777_native16_s1_v1',c['name'])
            result=dict(name=c['name'],output_gate=output,metrics=metrics(c['name']))
            results.append(result);write(ROOT/'state/destination16_progress.json',dict(results=results))
            print('D16_SERVING',c['name'],output['passed'],result['metrics']['TTFT_p95_ms'],result['metrics']['ITL_p99_ms'],flush=True)
            if not output['passed']:finish('SERVING',False,results,'OUTPUT_FAILURE');return
        finish('COMPLETE',True,results)
    except BaseException as error:
        finish('FAILED',False,results,repr(error));raise
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
