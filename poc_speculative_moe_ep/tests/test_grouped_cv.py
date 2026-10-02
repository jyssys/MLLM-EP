import numpy as np
import pandas as pd
from poc_speculative_moe_ep.scripts.analyze import cost_models


def test_request_folds_do_not_leak():
    rng=np.random.default_rng(42)
    rows=[]
    features=['unique_experts','expert_union','max_active_experts_rank','expert_skew','expert_histogram_skew',
              'remote_routed_paths','remote_destination_dedup_rows','rank_fanout','max_rank_bytes',
              'rank_byte_skew','source_destination_edges','max_bytes_per_layer','actual_max_rank_bytes',
              'actual_remote_dispatch_bytes','actual_unique_experts','executed_paths','dummy_padding_paths']
    for request in range(10):
        for step in range(4):
            x=dict(request_id=f'req{request}',gamma=2,verified_rows=3,committed_progress=2,accepted_draft_tokens=1,acceptance=.5)
            x.update({f:float(rng.random()) for f in features})
            x.update(ep_comm_ms=float(rng.random()+1),moe_ms=float(rng.random()+2),target_forward_ms=float(rng.random()+3))
            rows.append(x)
    result=cost_models(pd.DataFrame(rows))
    for fold in result['folds']:
        assert not set(fold['train_requests'])&set(fold['test_requests'])
    assert result['paired_request_mae_improvement_ms']['n']==10
