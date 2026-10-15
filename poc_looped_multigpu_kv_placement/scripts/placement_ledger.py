"""Legal ownership and logical transfer accounting. No invented candidate latency."""
import math
from poc_looped_multigpu_kv_placement.contract import *


def main():
    own=json.loads((ROOT/'results/raw/ouro_tp1_native_v1/loaded_ownership.json').read_text())[0]
    weight=own['unique_parameter_storage_bytes'];embedding=own['parameters']['model.embed_tokens.weight']['storage_bytes']
    tp2=json.loads((ROOT/'results/raw/ouro_tp2_native_v1/loaded_ownership.json').read_text())
    tp4=json.loads((ROOT/'results/raw/ouro_tp4_native_v1/loaded_ownership.json').read_text())
    head=own['parameters']['lm_head.weight']['storage_bytes'];core=weight-embedding-head
    slots=192;layers=48;passes=4;heads=16;dim=128;hidden=2048;dtype=2;positions=32704+63
    total_KV=2*slots*heads*dim*dtype*positions
    # Transfer counts are logical useful payloads. Protocol traffic is not measured here.
    cp_query_one_edge=8*dim*dtype;cp_return_one_edge=8*dim*dtype+8*4
    cp_fresh_KV_one_edge=2*8*dim*dtype
    layouts=[dict(name='NATIVE_TP4',status='MEASURED_NATIVE',
        weights_per_rank_bytes=[rank['unique_parameter_storage_bytes'] for rank in tp4],logical_KV_per_rank_bytes=[total_KV/4]*4,
        native_layer_reductions_per_forward=384,embedding_reduction_per_forward=1,
        extra_CP_phases_per_forward=0),
        dict(name='TP2_COMPUTE_CP2_KV',status='CONCEPTUAL_NOT_IMPLEMENTED',
        weights_per_rank_bytes=[rank['unique_parameter_storage_bytes'] for rank in tp2]+[0,0],logical_KV_per_rank_bytes=[total_KV/4]*4,
        owners={'physical4':'TP0 compute and half of its KV sequence','physical5':'TP1 compute and half of its KV sequence',
            'physical6':'Remaining TP0 KV sequence','physical7':'Remaining TP1 KV sequence'},
        native_layer_reductions_per_forward=384,embedding_reduction_per_forward=1,
        extra_CP_phases_per_forward=192,
        extra_query_and_partial_output_LSE_bytes_per_cached_forward=2*(cp_query_one_edge+cp_return_one_edge)*slots,
        fresh_remote_KV_write_bytes_per_cached_forward=2*cp_fresh_KV_one_edge*slots,
        initial_prefill_remote_KV_bytes_reference=total_KV/2,
        constraints=['KV-head ownership stays exact.', 'Current queries need prefix partial outputs and global softmax normalization.',
            'Fresh K/V append reaches each owning shard before later decode.', 'TP2 weights and compute use only two GPUs.'],
        latency_ms=None,unresolved=['Distributed attention launch/sync/merge cost','KV conversion and write overlap',
            'TP2 compute contention with remote CP attention','Graph/workspace and real pool allocation']),
        dict(name='RECURRENT_DEPTH_OWNERS',status='CONCEPTUAL_NOT_IMPLEMENTED',
        weights_per_rank_bytes=[core+embedding,core,core,core+head],logical_KV_per_rank_bytes=[total_KV/4]*4,
        owners={str(g):f'Loop {i}, all heads and all 48 physical core layers' for i,g in enumerate(GPUS)},
        inter_loop_hidden_bytes_per_cached_forward=3*hidden*dtype,
        inter_loop_hidden_bytes_per_prefill=3*32704*hidden*dtype,
        correction_or_next_token_feedback_bytes=8,
        constraints=['B1 loop dependencies remain sequential.', 'Each loop owner stores the complete shared core weights.',
            'The final output token returns to the first owner.', 'Independent-request pipeline claims require a DP baseline.'],
        latency_ms=None,unresolved=['Physical stage timing and pipeline utilization','Additional graph/workspace allocations'])]
    # A 25% increase from five resident long-context requests requires seven requests.
    # Their KV alone exceeds aggregate HBM before weights and workspace.
    hbm=subprocess_hbm()
    required=math.ceil(5*1.25)
    write(ROOT/'results/oracle_costs.json',dict(
        account='CONCEPTUAL_OWNERSHIP_AND_LOGICAL_TRANSFER_LEDGER. Not a candidate runtime or speedup.',
        positions_filled=positions,model_weight_storage_bytes=weight,core_weight_storage_bytes=core,
        KV_total_bytes=total_KV,layouts=layouts,
        largest_context_capacity_bound=dict(measured_native_TP4_resident=5,required_for_25_percent=required,
            KV_only_bytes_required=required*total_KV,aggregate_physical_HBM_bytes=hbm,
            exceeds_HBM_even_with_zero_weights=required*total_KV>hbm,
            limitation='Bound applies to this exact full-KV workload. It is not a global latency optimality result.'),
        calibrated_components='Native TP1/2/4 request latencies and diagnostic kernel services are measured.',
        candidate_cost_model='PARTIAL. Extra communication and different compute ownership are UNKNOWN. No zero-cost oracle speedup.'))
    print('Wrote exact ownership and logical transfer ledgers. Candidate latency remains UNKNOWN.')


def subprocess_hbm():
    import subprocess
    values=subprocess.check_output(['nvidia-smi','--id=4,5,6,7','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
    return sum(int(x) for x in values.split())*1024**2


if __name__=='__main__':main()
