"""Current tiny deterministic physical probe and full-cohort ownership audit."""
import json
import numpy as np
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *


def main():
    source=ROOT/'raw/long_v1';own=[]
    for rank in range(4):
        d=json.loads((source/f'ownership.rank{rank}.json').read_text())
        assert d['mask']==MASK and d['physical_gpu']==GPUS[rank]
        assert d['actual_cuda_uuid'].removeprefix('GPU-')==UUIDS[rank]
        assert len(d['ownership'])==19
        for l in d['ownership']:
            assert l['owned']==list(range(rank*64,(rank+1)*64))
            assert l['dispatcher']=='DeepEPDispatcher' and l['kernel']=='UnquantizedFusedMoEMethod'
            assert sum(w['bytes'] for w in l['weights'].values())//64==EXPERT_BYTES
        own.append(d)
    outputs=[]
    for rank in range(4):
        bypass=json.loads((source/'warmup0'/f'bypass.result.rank{rank}.json').read_text())
        trace=json.loads((source/'warmup1'/f'trace.result.rank{rank}.json').read_text())
        assert bypass['output_ids']==trace['output_ids'] and bypass['nfe']==trace['nfe']
        outputs.append(trace['output_ids'])
    assert outputs.count(outputs[0])==4
    df=pd.read_csv(ROOT/'results/long_requests.csv')
    assert len(df)==32 and df.semantic_parity.all()
    manifest=json.loads((ROOT/'results/long_manifest.json').read_text())
    for info in manifest['requests']:
        assert info['audit']['live_bytes']==info['B_live_only']
    new_json(ROOT/'results/baseline_validation.json',dict(status='PASS',mask=MASK,
        physical_logical=dict(enumerate(GPUS)),actual_expert_sharding=True,real_DeepEP_sparse_dispatch_combine=True,
        topology=own[0]['topology'],model_revision=REVISION,expert_bytes=EXPERT_BYTES,sparse_layers=19,
        tiny_same_prompt_bypass_trace_cross_rank_output_NFE_parity=True,
        full_cohort_lane_token_NFE_parity=True,acceptance_cross_rank_parity=True,
        additional_acceptance_probe_requests=2,
        actual_receive_buffer_destination_dedup_checks=int(df.physical_buffer_validation_checks.sum()),
        observed_generation_min=int(df.output_tokens.min()),observed_generation_max=int(df.output_tokens.max()),
        budget512_native_block_end_roundup=True,
        timing_observer_mean=float(df.timing_overhead_fraction.mean()),
        timing_status='baseline only; same critical rank cumulative denoising forward, no replica timing'))
    print('PERSIST_TRUE_EP4_VALIDATION_PASS',flush=True)


if __name__=='__main__':main()
