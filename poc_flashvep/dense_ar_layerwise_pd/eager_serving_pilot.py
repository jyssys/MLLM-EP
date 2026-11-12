"""Keep baseline and migrated work in one exact-BF16 execution family."""
import fcntl,json,time
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare
from .state_qualification import state_compare
from .actual_layer_gate import gate


def prepare():
    native_gate=json.loads((ROOT/'analysis/native_token_reuse_prefix_N32_independent_gate.json').read_text())
    assert native_gate['passed']
    refs={n:f'eager_qwen8_L8000_O16_N{n}_reference_qual_v16' for n in (8,32)}
    output_refs={8:'eager_qwen8_L8000_O16_N8_local_full_qual_v16',
        32:'eager_qwen8_L8000_O16_N32_independent_full_cohort_reference_v21'}
    checks=[native_gate,state_compare(refs[32],refs[8],'prefill',None,True)]
    for cut in (35,34,32,28):
        for count in (8,32):checks.append(gate(f'eager_qwen8_L8000_O16_N{count}_c{cut}_qual_v16',
            refs[count],output_refs[count],'independent_reference_gate_v22'))
    assert all(x['passed'] for x in checks)
    write(ROOT/'analysis/eager_aggregate_correctness_gate_v22.json',dict(checks=checks,passed=True,
        scope='Independent full-model outputs corroborate all four actual suffix paths. The earlier Native last-query replay changed valid state. Corrected Native preserves the real P first token. Earlier failed receipts remain unchanged.'))
    template=json.loads((ROOT/'configs/serving_qwen8_L8000_O16_N64_s3_native_pilot_v1.json').read_text());jobs=[]
    arms=[('native',None,32,'PD_G1',False),('guard','STATE_REFERENCE',32,'PD_G1',True)]
    arms +=[(f'c{k}','MIGRATE',k,'PD_G1',True) for k in (35,34,32,28)]
    arms +=[('colocated_cohort',None,32,'DP2',False)]
    for label,mode,cut,arm,guard in arms:
        c=dict(template,name=f'eager_serving_qwen8_L8000_O16_N64_s3_{label}_v22',
            ar_enforce_eager=True,ar_execution_family='BI_EAGER_NO_GRAPH',layer_mode=mode,layer_cut=cut,
            arm=arm,serialize_D_imports=guard,cohort_barrier=guard,separate_suffix_runner=mode=='MIGRATE',
            partial_KV_wire=mode=='MIGRATE',normal_lifetime_audit=True,
            hidden_transport=None if arm=='DP2' else 'NIXL_GPU',colocated_cohorts=arm=='DP2',
            verify_layer_execution=False,execution_events=False,state_capture=False,
            retain_fp32_boundary_residual=False,residual_dtype_bridge=False,
            reuse_prefill_token=arm=='PD_G1' and mode!='MIGRATE',
            phase='EXACT_BOUNDARY_MARKER_FREE_SERVING_SELECTION',
            scope='Exact-BF16 Native and Layer methods share Eager CUDA operator execution. Original aggressive compiled Native remains a separate runtime control, not an accuracy-matched denominator.')
        jobs.append(save(c))
    write(ROOT/'state/eager_serving_pilot_plan.json',dict(jobs=jobs))
    return jobs


def main():
    while True:
        p=ROOT/'state/eager_correctness_progress.json'
        rows=json.loads(p.read_text())['results'] if p.exists() else []
        if len(rows)==11:break
        assert time.time()<LONG_START_CUTOFF
        time.sleep(5)
    results=[];names=prepare()
    try:
        for n in names:
            if not execute(n,restore=False):break
            row=dict(name=n,metrics=metrics(n),output_gate=compare(names[0],n))
            results.append(row);write(ROOT/'state/eager_serving_pilot_progress.json',dict(results=results))
            print('EAGER_SERVING',n,{k:row['metrics'][k] for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms')},row['output_gate']['passed'],flush=True)
            if not row['output_gate']['passed']:break
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
