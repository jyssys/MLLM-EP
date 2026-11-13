"""Check larger receiver shapes against a preserved independent full model."""
import fcntl
from .common import *
from .run import execute
from .analyze import records,compare


def main():
    reference=PREVIOUS/'raw/qual_D16_independent_full_state_N32_v2'
    expected={r['external_request_id']:r for p in reference.glob('extension_prefill_*.jsonl')
        for r in records(p) if r['event']=='VALID_QUERY_STATE' and r.get('query_tokens')==8000
        and r['external_request_id'].startswith('q')}
    assert len(expected)==32,len(expected)
    results=[]
    try:
        for batch,layers in [(16,k) for k in (2,4,8,12)]+[(8,12)]:
            cfg=json.loads((ROOT/'configs'/f'cost_Source4_Destination{batch}_Last{layers}_N{2*batch}_v1.json').read_text())
            count=32;cfg.update(name=f'exact_receiver_D{batch}_Last{layers}_N32_full_state_v3',
                count=count,dataset='discovery.json',request_caps=[2]*count,arrival_offsets=[0.]*count,
                execution_events=False,verify_layer_execution=True,hidden_verify_hash=True,
                numerical_native_reference=False,full_prompt_state_witness=True,
                full_prompt_state_witness_roles=['decode'],correctness_only=True,
                defer_completion_log=True,negotiation_timeout_s=120,
                phase='LARGER_DESTINATION_FULL_STATE_CORRECTNESS_ONLY',
                diagnostic_timeout_scope='Allow checksum copies before reservation replies. This120-second limit is diagnostic only. Serving keeps its original2-second negotiation limit.')
            name=save(cfg)
            if not execute(name,restore=False):return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            states=json.loads((ROOT/'raw'/name/'full_prompt_state_decode.json').read_text())['requests']
            checks=[]
            for rid in sorted(expected):
                value=states.get(rid)
                checks.append(dict(request_id=rid,present=value is not None,
                    all36_KV_bitwise_equal=value is not None and value['KV']==expected[rid]['KV'],
                    all8000_hidden_bitwise_equal=value is not None and value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256']))
            from ..dense_ar_dynamic_layer_pd import qualify
            qualify.ROOT=ROOT;qualify.state_gate(name,count,mixed=True)
            gate=json.loads((ROOT/'analysis'/(name+'_gate.json')).read_text())
            work_checks=[r for r in gate['checks'] if r['check']!='MULTIPLE_ACTUAL_CUTS']
            passed=output['passed'] and all(all(r[k] for k in ('present','all36_KV_bitwise_equal','all8000_hidden_bitwise_equal')) for r in checks) and all(r['passed'] for r in work_checks)
            row=dict(name=name,output=output,state_checks=checks,work_checks=work_checks,passed=passed,
                receiver_batch=batch,layers=layers,evidence='CORRECTNESS_ONLY',
                scope='Forced homogeneous shape. Full-state hashes use the preserved independent Full36 reference. This is not an online dynamic cut or a performance run.')
            results.append(row);write(ROOT/'state/receiver_state_qualification_v3_progress.json',dict(results=results))
            print('RECEIVER_STATE',name,passed,flush=True)
            if not passed:return
        write(ROOT/'analysis/receiver_full_state_qualification_v3.json',dict(results=results,passed=all(r['passed'] for r in results)))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
