"""Account for rows, causal attention pairs and valid transfer payloads."""
import json
import math
from fractions import Fraction
from .common import ROOT,write,digest
from .partial_axis import token_prefix


def main():
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    total=sum(r['input_tokens'] for r in data);rows=[];totals=[]
    for cut in (1,2,4,6,8,12):
        plans=[]
        for row in data:
            length=row['input_tokens'];prefix=token_prefix(length,cut);suffix=length-prefix
            triangular=lambda x:x*(x+1)//2
            value=dict(prompt_id=row['prompt_id'],task=row['task'],input_tokens=length,cut=cut,
                Token_prefix=prefix,Token_suffix=suffix,
                Layer_P_rows=(36-cut)*length,Layer_D_rows=cut*length,
                Token_P_rows=36*prefix,Token_D_rows=36*suffix,
                Layer_P_attention_pairs=(36-cut)*triangular(length),Layer_D_attention_pairs=cut*triangular(length),
                Token_P_attention_pairs=36*triangular(prefix),Token_D_attention_pairs=36*(triangular(length)-triangular(prefix)),
                Layer_valid_KV_payload_bytes=(36-cut)*length*4096,Token_valid_KV_payload_bytes=36*prefix*4096,
                Layer_extra_hidden_residual_bytes=length*16384,
                Layer_full_prompt_chunks=math.ceil(length/8192),Token_P_chunks=math.ceil(prefix/8192),Token_D_chunks=math.ceil(suffix/8192),
                Source_Layer_reserved_full36_KV_bytes=36*length*4096,
                Source_Token_reserved_full36_KV_bytes=36*prefix*4096,
                exact_valid_rows=cut*length==36*suffix)
            assert value['Layer_P_attention_pairs']+value['Layer_D_attention_pairs']==36*triangular(length)
            assert value['Token_P_attention_pairs']+value['Token_D_attention_pairs']==36*triangular(length)
            plans.append(value);rows.append(value)
        target=cut*total;actual=sum(p['Token_D_rows'] for p in plans)
        totals.append(dict(cut=cut,N=200,Layer_D_rows=target,Token_D_rows=actual,
            relative_row_difference=actual/target-1,exact_aggregate_rows=actual==target,
            minimal_whole_dataset_repeats_for_exact_16aligned_prefix=Fraction((36-cut)*total,576).denominator,
            Layer_D_attention_pairs=sum(p['Layer_D_attention_pairs'] for p in plans),
            Token_D_attention_pairs=sum(p['Token_D_attention_pairs'] for p in plans)))
    write(ROOT/'analysis/BI0_axis_structural_accounting.json',dict(dataset_SHA256=digest(ROOT/'data/longbench_shortanswer200_v2.json'),
        total_input_tokens=total,rows=rows,totals=totals,evidence='STRUCTURAL_ACCOUNTING',
        formulae=dict(equal_rows='L*m = 36*s',exact_D_attention_ratio='(2*L-s+1)/(L+1)',
            matched_valid_KV='(36-m)*L*4096 == 36*(L-s)*4096 when rows match',
            Layer_extra_hidden_residual='2 * L * 4096 * 2 bytes'),
        scope='Valid rows and causal pairs, not FLOPs measured on GPU. KV bytes exclude block padding, metadata, retries and ACK. '
        'Both methods have the same total logical attention pairs. Their P/D placement differs. Full200 nearest alignment is not exactly matched work.'))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
