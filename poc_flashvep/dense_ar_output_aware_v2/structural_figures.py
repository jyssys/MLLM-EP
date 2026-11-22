"""Plot conserved work separately from measured kernel costs."""
import json
import math
from pathlib import Path
from .common import ROOT,write,MODEL,digest
from .c0_analysis import csv_save,INPUTS
from ..dense_ar_longbench400_pd.geometry import Partition,matched_suffixes

def main():
    model_path=Path(MODEL)/'config.json';model=json.loads(model_path.read_text())
    hidden=model['hidden_size'];intermediate=model['intermediate_size']
    kv=model['num_key_value_heads']*model['head_dim']
    write(ROOT/'analysis/model_gemm_dimension_contract_v2.json',dict(
        model_config_SHA256=digest(model_path),model_layers=model['num_hidden_layers'],
        hidden_size=hidden,intermediate_size=intermediate,
        expected_weight_N_K=dict(QKV=[hidden+2*kv,hidden],O=[hidden,hidden],
            GateUp=[2*intermediate,hidden],Down=[hidden,intermediate]),
        M_scope='Valid query rows depend on actual chunk,batch and partition. These dimensions do not measure kernel padding or dispatch.',
        evidence='STRUCTURAL_ACCOUNTING',actual_operator_kernel_cost='NOT_MEASURED'))
    rows=[]
    for bucket in range(4):
        data=json.loads((ROOT/'data'/f'input{bucket}_heldout_exact16.json').read_text());lengths=[r['input_tokens'] for r in data]
        for cut in (2,4,8,12):
            suffix,work=matched_suffixes(lengths,cut);assert work['exact_match']
            parts={'Layer':[Partition(length,cut,0).record() for length in lengths],
                'Token':[Partition(length,0,s).record() for length,s in zip(lengths,suffix)]}
            totals={axis:{field:sum(p[field] for p in values) for field in
                ('P_rows','D_rows','P_attention_pairs','D_attention_pairs','valid_KV_bytes','hidden_residual_bytes')} for axis,values in parts.items()}
            assert totals['Layer']['D_rows']==totals['Token']['D_rows']
            assert totals['Layer']['valid_KV_bytes']==totals['Token']['valid_KV_bytes']
            rows.append(dict(input_bucket=bucket,input_label=INPUTS[bucket],cut=cut,requests=16,totals=totals,
                D_attention_Token_over_Layer=totals['Token']['D_attention_pairs']/totals['Layer']['D_attention_pairs'],
                extra_P_attention_pairs_Layer=totals['Layer']['P_attention_pairs']-totals['Token']['P_attention_pairs'],
                Layer_P_minimum_chunks_at_B8192=sum(math.ceil(L/8192) for L in lengths),
                Token_P_minimum_chunks_at_B8192=sum(math.ceil((L-s)/8192) for L,s in zip(lengths,suffix)),
                evidence='STRUCTURAL_ACCOUNTING',GPU_operator_time='NOT_MEASURED_BY_THIS_FIGURE'))
    csv_save(ROOT/'figures/pd_attention_and_payload_structural_v2.csv',rows)
    write(ROOT/'analysis/pd_attention_and_payload_structural_v2.json',dict(rows=rows,
        scope='Exact aggregate transformer token-layer rows on frozen original16-request subsets. '
            'Total causal Attention pairs remain constant. Token places fewer pairs on P and more on D. '
            'Valid KV bytes exclude block padding,descriptors and metadata. Hidden/residual bytes are separate. '
            'Rows do not count embeddings or LM-head GEMMs. Native proxy last-query recomputation remains a separate actual operation. '
            'These counts do not establish Attention FLOPs as GPU time,GEMM efficiency,DMA duration or serving speedup.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for bucket in range(4):
        part=[r for r in rows if r['input_bucket']==bucket]
        axes[0].plot([r['cut'] for r in part],[r['D_attention_Token_over_Layer'] for r in part],'o-',label=INPUTS[bucket])
        axes[1].plot([r['cut'] for r in part],[r['extra_P_attention_pairs_Layer']/1e9 for r in part],'o-',label=INPUTS[bucket])
    axes[0].set(xlabel='Layer cut m',ylabel='Token / Layer D Attention-pair ratio',title='Equal aggregate GEMM rows')
    axes[1].set(xlabel='Layer cut m',ylabel='Additional Layer P Attention pairs (billions)',title='Work placement,not extra total work')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.text(.02,.015,'STRUCTURAL_ACCOUNTING. Original heldout16 prompts/bin. Equal rows do not imply equal Attention work per GPU or equal kernel time.\n'
        'Valid KV payloads match at equal P rows. Layer also transfers hidden/residual state. Actual transport includes alignment and metadata.',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1))
    for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'pd_attention_and_payload_structural_v2.{ext}',dpi=170)
    plt.close(fig);print('STRUCTURAL_ATTENTION_AND_PAYLOAD_FIGURES',len(rows),flush=True)

if __name__=='__main__':main()
