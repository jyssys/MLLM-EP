"""Qualify the no-migration span, then measure a bounded static pilot."""
import fcntl,json,random,sys
from .common import *
from .prepare import base,save
from .run import execute
from .analyze import compare,metrics
from .state_qualification import state_compare


def prepare():
    gates=json.loads((ROOT/'state/full_prompt_correctness_progress.json').read_text())['results']
    assert len(gates)==10 and all(x['passed'] for x in gates)
    capacity=3.6999870452996393
    jobs=[]
    for label,mode,cut,guard in [('native',None,36,False),('guard','STATE_REFERENCE',36,True),
                               ('span','SPAN_ZERO',36,True),('c35','MIGRATE',35,True),
                               ('c34','MIGRATE',34,True),('c32','MIGRATE',32,True),('c28','MIGRATE',28,True)]:
        name=f'serving_qwen8_L8000_O16_N64_s3_{label}_pilot_v1'
        c=base(name,count=64,cap=16);rng=random.Random(2026111203);offsets=[0.]
        for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(capacity*1.3))
        c.update(dataset='discovery_qwen8_L8000.json',arrival_offsets=offsets,arrival_rate=capacity*1.3,
            arrival_seed=2026111203,layer_mode=mode,layer_cut=cut if cut<36 else 32,
            state_capture=False,save_state_tensors=False,hidden_verify_hash=False,
            hidden_transport='NIXL_GPU',hidden_source_slots=16,hidden_destination_slots=4,
            partial_KV_wire=mode=='MIGRATE',serialize_D_imports=guard,D_admission_batch=4,
            D_cohort_collect_ms=2,cohort_barrier=guard,batch_commands=True,
            warmup_batch_sizes=[1,2,3,4],warmup_output_cap=16,fixed_KV_bytes=42*1024**3,
            isolate_layer_compile_cache=True,minimal_serving_trace=True,timeout=300,
            clean_timing=True,phase='BOUNDED_STATIC_LAYER_SERVING_SELECTION',
            scope='Static cut with bounded destination cohorts. No online future-slack predictor. Marker-free model execution. This selection trace is not held-out confirmation.')
        jobs.append(save(c))
    write(ROOT/'state/serving_pilot_plan.json',dict(jobs=jobs,capacity_requests_s=capacity,selection_seed=2026111203))
    return jobs


def run():
    out=[]
    try:
        for count in (8,32):
            n=f'layer_qwen8_L8000_O16_span_zero_N{count}_qual_v7'
            if not execute(n,restore=False):return
            ref=f'layer_qwen8_L8000_O16_state_reference_c36_N{count}_gpu_v6'
            checks=[compare(ref,n),state_compare(ref,n,'prefill')]
            gate=dict(name=n,passed=all(x['passed'] for x in checks),checks=checks)
            write(ROOT/'analysis'/(n+'_state_gate.json'),gate)
            print('SPAN_GATE',n,gate['passed'],flush=True)
            if not gate['passed']:return
        for n in prepare():
            ok=execute(n,restore=False)
            row=dict(name=n,complete=ok)
            if ok:
                row.update(metrics=metrics(n),output_gate=compare('native_qwen8_L8000_O16_N128_s4_cohort_cleanA_v4',n,True))
            out.append(row);write(ROOT/'state/serving_pilot_progress.json',dict(results=out))
            print('SERVING',n,ok,{k:v for k,v in row.get('metrics',{}).items() if k in ('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms')},flush=True)
            if not ok:break
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':run()
