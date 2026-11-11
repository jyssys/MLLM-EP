"""Test Native pipelining before interpreting a slot-limited queue as compute."""
import json
from .common import *


def main():
    initialize();jobs=[];matched=[]
    for model,length in (('ouro',16384),('huginn',3900),('huginn',3500),('huginn',3072)):
        original=f'coarse_{model}_L{length}_O16_P1_mu115_N64_a2026111230'
        cfg=json.loads((ROOT/'configs'/(original+'.json')).read_text());name=original.replace('_P1_','_P2_')
        cfg.update(name=name,P_slots=2,token_budget=2*length,P_prefill_graph_rows=2*length,
            phase='NATIVE_SLOT_PIPELINE_RECOVERY_SAME_ARRIVALS',comparison_reference=original,
            selection_reason='P1 pilots have a real queue but weak service proximity, especially on Huginn. Test whether P2 overlaps Native transfer retention without changing weights, math, QPS, or requests.')
        save(cfg);jobs.append(name)
        if model=='huginn':
            source=f'capacity_huginn_L{length}_{"O16_" if length==3900 else ""}B124_start1'
            cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text());name=f'matched_P_pool_huginn_L{length}_B12_start1'
            cfg.update(name=name,kv_blocks=16384,capacity=2,slots=2,P_slots=2,graph_rows=2,token_budget=2*length,
                P_prefill_graph_rows=2*length,jobs=[dict(batch=b,cap=16,repeats=3) for b in (1,2)],count=2,
                request_caps=[16]*2,arrival_offsets=[0.]*2,prompt_ids=cfg['prompt_ids'][:2],
                phase='MATCHED_NATIVE_P_KV_POOL_FINITE_CAPACITY',
                data_scope=dict(distinct_prompt_ids=2,calibration_pool_records=24,prefix_caching=False,
                    scope='Matched 16384-block P pool. B4 is infeasible at these lengths. Prepared Decode remains a separate local barrier reference.'))
            save(cfg);matched.append(name)
    write(ROOT/'state/slot_recovery_plan.json',dict(jobs=jobs,matched_capacity=matched,
        scope='Native baseline discovery only. Select the stronger legal Native configuration before any Loop benefit comparison. Raw D-refusal prevalence does not by itself identify a causal P throughput bottleneck.'))
    print(' '.join(jobs[:2]+matched[:1]+jobs[2:]+matched[1:]))


if __name__=='__main__':main()
