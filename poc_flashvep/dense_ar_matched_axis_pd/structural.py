"""Exact axis work accounting. These values are not GPU durations."""
from .common import ROOT,write,initialize
from ..dense_ar_layer_token_pd.token_axis import matched_token_plan
from ..dense_ar_layer_token_pd.structural import counts


def main():
    initialize()
    rows=[]
    for layers in (4,8,12):
        plan=matched_token_plan(layers,144,group_size=1)
        per=[counts(8000,layers,s) for s in plan]
        total={key:sum(r[key] for r in per) for key in per[0]}
        all_pairs=36*8000*8001//2*144
        total.update(layer_P_attention_pairs=all_pairs-total['layer_D_attention_pairs'],
            token_P_attention_pairs=all_pairs-total['token_D_attention_pairs'],
            total_attention_pairs=all_pairs,
            layer_P_GEMM_rows=144*8000*(36-layers),
            token_P_GEMM_rows=36*sum(8000-s for s in plan))
        assert total['layer_D_GEMM_rows']==total['token_D_GEMM_rows']
        assert total['layer_P_GEMM_rows']==total['token_P_GEMM_rows']
        assert total['layer_valid_KV_bytes']==total['token_valid_KV_bytes']
        rows.append(dict(layers=layers,count=144,token_suffix_histogram={str(s):plan.count(s) for s in sorted(set(plan))},
            D_token_to_layer_attention_pair_ratio=total['token_D_attention_pairs']/total['layer_D_attention_pairs'],
            P_token_to_layer_attention_pair_ratio=total['token_P_attention_pairs']/total['layer_P_attention_pairs'],
            totals=total))
    write(ROOT/'analysis/matched_axis_structural_work.json',dict(rows=rows,evidence='STRUCTURAL_ACCOUNTING',
        scope='Exact causal QK/PV pairs. Multiply-add counts as two FLOPs. Excludes softmax, padding, metadata, physical traffic and non-GEMM operators. Full-system attention work is equal. Its P/D distribution is not equal. Equal logical KV bytes do not imply equal source KV-credit occupancy. No GPU time or serving speedup is inferred.'))
    print('EXACT_AXIS_WORK',[(r['layers'],r['D_token_to_layer_attention_pair_ratio']) for r in rows])


if __name__=='__main__':
    main()
