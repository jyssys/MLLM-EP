"""Distinguish compute concurrency from the earlier transfer-slot-only probe."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .capacity_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/event_bound_confirmation_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    reference='reserve_huginn_L3072_native_reference_state8'
    cfg=json.loads((ROOT/'configs'/(reference+'.json')).read_text());state='native_slots3_huginn_L3072_pool23296_state8'
    cfg.update(name=state,P_slots=3,kv_blocks=23296,token_budget=9216,P_prefill_graph_rows=9216,
        required_qualification=reference,campaign_state_reference=reference,qualification_fresh_native=False,
        state_hash_defer_until_close=False,state_pinned_slots=6,
        phase='NATIVE_THREE_COMPUTE_SLOTS_FOUR_D_TARGETS_EXACT_STATE')
    for k in ('qualification_output_reference','qualification_state_subset_reference'):cfg.pop(k,None)
    save(cfg)
    cfg=json.loads((ROOT/'configs/matched_graph_reserve_capacity_huginn_L3072_B12_start1.json').read_text())
    cap='native_slots3_capacity_huginn_L3072_pool23296_B123_start1'
    cfg.update(name=cap,capacity=3,slots=3,P_slots=3,kv_blocks=23296,count=3,request_caps=[16]*3,
        arrival_offsets=[0.]*3,token_budget=9216,P_prefill_graph_rows=9216,
        jobs=[dict(batch=b,cap=16,repeats=3) for b in (1,2,3)],required_qualification=state,
        phase='NATIVE_THREE_COMPUTE_SLOT_MATCHED_GRAPH_REFERENCE')
    save(cfg)
    names=[]
    for slots in (2,3):
        cfg=json.loads((ROOT/'configs/reserve_huginn_L3072_O16_P2_mu115_N64_a2026111230.json').read_text())
        name=f'native_slots{slots}_huginn_L3072_pool23296_O16_pilot64_a2026111230'
        cfg.update(name=name,P_slots=slots,kv_blocks=23296,token_budget=slots*3072,P_prefill_graph_rows=slots*3072,
            required_qualification=state,capacity_reference=cap,comparison_reference=None,
            phase='PAIRED_NATIVE_COMPUTE_SLOT_DISCOVERY_WITH_FOUR_D_TARGETS',
            selection_reason='The 3500-token source-three pilot had only three complete D KV targets and regressed. At 3072, four D targets fit. Test three real compute slots, not the previously failed three-transfer/two-compute configuration. Both controls use the same legal 23296-block P pool.')
        save(cfg);names.append(name)
    jobs=[state,cap,*names];write(ROOT/'state/native_three_slots_3072_plan.json',dict(jobs=jobs,
        scope='Native optimization probe only. Reject a slower baseline. No Loop offloading or SLO change. The earlier transfer-slot probe and longer-context failure remain separate results.'))
    if sequence(jobs,'native_three_slots_3072_controller'):
        capacity=analyze(cap);values=[screen(n,reference=names[0]) for n in names]
        a,b=[r['metrics'] for r in values]
        comparison={k:b[k]/a[k] for k in ('TTFT_p95_ms','E2E_p95_ms','TPOT_p95_ms','ITL_p99_ms','completed_QPS')}
        write(ROOT/'analysis/native_three_slots_3072_results.json',dict(results=values,comparison=comparison,capacity=capacity,
            scope='One paired Native pilot. No Loop benefit or physical-idle claim.'))
    write(ROOT/'state/native_three_slots_3072_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
