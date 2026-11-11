"""Keep the startup reserve and reduce only the source KV pool."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for length in (3900,3072):
        parent=f'memory_huginn_L{length}_O16_P2_mu115_N64_a2026111230'
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
        name=parent.replace('memory_','reserve_');cfg.update(name=name,kv_blocks=24064,
            phase='NATIVE_KV_POOL_WITH_UNCHANGED_STARTUP_HBM_GUARD',comparison_reference=parent,
            selection_reason='The 24832-block P pool left 2219311104 bytes during L3900 P2 warm-up. The required reserve is 3221225472 bytes. Reduce P KV by 768 blocks. Preserve the guard, D pool, graphs, math, and arrivals.')
        state=json.loads((ROOT/'configs'/f'memory_huginn_L{length}_native_reference_state8.json').read_text())
        state_name=f'reserve_huginn_L{length}_native_reference_state8'
        state.update(name=state_name,kv_blocks=24064,
            phase='FRESH_NATIVE_STATE_REFERENCE_WITH_SAFE_HBM_RESERVE')
        save(state);cfg['required_qualification']=state_name
        cap_parent=f'capacity_huginn_L{length}_{"O16_" if length==3900 else ""}B124_start1'
        cap=json.loads((ROOT/'configs'/(cap_parent+'.json')).read_text())
        cap_name=f'reserve_capacity_huginn_L{length}_B12_start1'
        cap.update(name=cap_name,kv_blocks=24064,capacity=2,slots=2,P_slots=2,
            graph_rows=2,token_budget=2*length,P_prefill_graph_rows=2*length,count=2,
            prompt_ids=cap['prompt_ids'][:2],arrival_offsets=[0.,0.],request_caps=[16,16],
            jobs=[dict(batch=b,cap=16,repeats=3) for b in (1,2)],
            phase='MATCHED_24064_BLOCK_NATIVE_FINITE_CAPACITY')
        save(cap);cfg['capacity_reference']=cap_name;save(cfg)
        jobs.extend((state_name,cap_name,name))
    write(ROOT/'state/hbm_guard_recovery_plan.json',dict(jobs=jobs,
        scope='The failed warm-up remains preserved. Do not relax the HBM safety guard. Qualification precedes Serving.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
