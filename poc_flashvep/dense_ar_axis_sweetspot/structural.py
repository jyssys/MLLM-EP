"""Count causal attention pairs and persistent-GEMM padding separately."""
import json
import math
from .common import ROOT,MODEL,write,initialize
from ..dense_ar_layer_token_pd.token_axis import matched_token_plan


def pair_count(q,k):
    assert 0<q<=k
    return q*(k-q)+q*(q+1)//2


def main():
    initialize()
    n=8000;L=36;N=72
    config=json.loads((MODEL/'config.json').read_text())
    h=config['hidden_size'];intermediate=config['intermediate_size']
    heads=config['num_attention_heads'];KVheads=config['num_key_value_heads']
    dimension=config.get('head_dim',h//heads)
    KVbytes=2*KVheads*dimension*2
    hidden_residual_bytes=2*n*h*2
    projections=dict(qkv=(heads*dimension+2*KVheads*dimension,h),o=(h,heads*dimension),
        gate_up=(2*intermediate,h),down=(h,intermediate))
    projection_weight_bytes_per_layer=2*sum(NN*K for NN,K in projections.values())
    rows=[]
    for m in (2,4,8,12):
        suffixes=matched_token_plan(m,N,group_size=1)
        assert sum(suffixes)*L==n*m*N
        layer_D=N*m*pair_count(n,n)
        token_D=sum(L*pair_count(s,n) for s in suffixes)
        layer_P=N*(L-m)*pair_count(n,n)
        token_P=sum(L*pair_count(n-s,n-s) for s in suffixes)
        assert layer_D+layer_P==token_D+token_P==N*L*pair_count(n,n)
        row=dict(layers=m,requests=N,suffix_histogram={str(s):suffixes.count(s) for s in sorted(set(suffixes))},
            P_valid_GEMM_rows=N*n*(L-m),D_valid_GEMM_rows=N*n*m,
            Layer_P_attention_pairs=layer_P,Layer_D_attention_pairs=layer_D,
            Token_P_attention_pairs=token_P,Token_D_attention_pairs=token_D,
            D_attention_pair_ratio_Token_over_Layer=token_D/layer_D,
            total_attention_pairs_both=layer_P+layer_D,
            Layer_KV_payload_bytes=N*n*(L-m)*KVbytes,
            Token_KV_payload_bytes=sum((n-s)*L*KVbytes for s in suffixes),
            Layer_extra_hidden_residual_bytes=N*hidden_residual_bytes,
            Layer_D_persistent_tile_rows=N*m*math.ceil(n/128)*128,
            Token_D_persistent_tile_rows=sum(L*math.ceil(s/128)*128 for s in suffixes),
            Layer_P_persistent_tile_rows=N*(L-m)*math.ceil(n/128)*128,
            Token_P_persistent_tile_rows=sum(L*math.ceil((n-s)/128)*128 for s in suffixes),
            major_projection_calls_per_request=dict(Layer_P=4*(L-m),Layer_D=4*m,Token_P=4*L,Token_D=4*L),
            unique_major_projection_weight_footprint_bytes=dict(Layer_P=(L-m)*projection_weight_bytes_per_layer,
                Layer_D=m*projection_weight_bytes_per_layer,Token_P=L*projection_weight_bytes_per_layer,
                Token_D=L*projection_weight_bytes_per_layer),
            D_GEMM_FLOPs_both=N*n*m*sum(2*K*NN for NN,K in projections.values()),
            D_attention_FLOPs=dict(Layer=4*layer_D*heads*dimension,Token=4*token_D*heads*dimension),
            scope='Valid GEMM rows match exactly. Causal attention work shifts from P to D in Token. Total attention work stays equal. Tile rows model the installed persistent kernel128-row tiling. These counts are not GPU time or measured HBM traffic.')
        assert row['Layer_KV_payload_bytes']==row['Token_KV_payload_bytes']
        rows.append(row)
    write(ROOT/'analysis/matched_geometry_structural_accounting.json',dict(
        model_revision='b968826d9c46dd6066d109eabc6255188de91218',
        model_shapes=dict(hidden=h,intermediate=intermediate,Q_heads=heads,KV_heads=KVheads,head_dim=dimension),
        projection_weight_shapes_N_K=projections,KV_bytes_per_token_layer=KVbytes,
        projection_weight_bytes_per_layer=projection_weight_bytes_per_layer,rows=rows,
        evidence='STRUCTURAL_ACCOUNTING',
        scope='Original checkpoint geometry. No measured speedup follows. Unique weight footprint is not HBM traffic or a claim that each byte is read once. Cache layout, sampling, RMSNorm, RoPE, transport and host costs remain outside the FLOP count.'))
    print('STRUCTURAL_ACCOUNTING',[(r['layers'],round(r['D_attention_pair_ratio_Token_over_Layer'],4)) for r in rows])


if __name__=='__main__':
    main()
