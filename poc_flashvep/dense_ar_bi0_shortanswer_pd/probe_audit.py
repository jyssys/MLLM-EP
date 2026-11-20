"""Separate imported KV integrity from the final-query recomputation."""
import torch
from .common import ROOT,write,digest


def main():
    torch.set_num_threads(1);rows=[]
    teacher=ROOT/'raw/NumericalSync_Standard_ShortAnswer_N8_BI0_v2'
    actual=ROOT/'raw/NumericalSync_Native_ShortAnswer_N8_BI0_v2'
    for index in range(8):
        rid=f'q{index}';a_path=teacher/f'numerical_single_{rid}.pt'
        a=torch.load(a_path,weights_only=True)
        for role in ('prefill','decode'):
            b_path=actual/f'numerical_{role}_{rid}.pt';b=torch.load(b_path,weights_only=True)
            error=(a['KV_samples']-b['KV_samples']).abs()
            # Saved shape: layer, sampled token, KV heads, combined K/V channels.
            assert error.ndim==4 and error.shape[1]==len(a['positions'])
            per_position=error.permute(1,0,2,3).flatten(1).amax(1).tolist()
            rows.append(dict(request_id=rid,role=role,positions=a['positions'].tolist(),
                maximum_absolute_error_by_position=per_position,
                all_positions_before_last_exact=all(v==0 for v in per_position[:-1]),
                last_position_exact=per_position[-1]==0,
                hidden_exact=torch.equal(a['hidden_last'],b['hidden_last']),
                reference_query_start=a['query_start'],reference_query_count=a['query_count'],
                actual_query_start=b['query_start'],actual_query_count=b['query_count'],
                reference_SHA256=digest(a_path),actual_SHA256=digest(b_path)))
    write(ROOT/'analysis/BI0_synchronized_probe_position_audit_v1.json',dict(rows=rows,
        evidence='CORRECTNESS_ONLY',
        scope='Device-synchronized sparse tensors. Import is checked separately with full valid KV hashes. '
        'Native D recomputes one final prompt query with different BI0 geometry. '
        'The synchronization intervention removes the earlier P mismatch. It does not identify the exact stream-ordering cause of the unsynchronized probe.'))


if __name__=='__main__':main()
