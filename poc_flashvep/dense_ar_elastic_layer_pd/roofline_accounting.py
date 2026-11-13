"""Compute optimistic bounds, not measured utilization or online budgets."""
from .common import *


def main():
    config=json.loads((MODEL/'config.json').read_text())
    h=config['hidden_size'];d=config['head_dim'];q=config['num_attention_heads']*d
    kv=config['num_key_value_heads']*d;intermediate=config['intermediate_size'];length=8000
    parameters=h*(q+2*kv)+q*h+3*h*intermediate
    peak=989.5e12;bandwidth=3.35e12;rows=[]
    for batch in (1,4,8,16):
        m=batch*length
        gemm_flops=2*m*parameters
        causal_attention_flops=4*config['num_attention_heads']*d*batch*length*(length+1)/2
        # Four fused projection groups: QKV, O, gate/up, and down.
        gemm_bytes=2*parameters+2*m*(4*h+2*q+2*kv+3*intermediate)
        for layers in (2,4,8,12):
            rows.append(dict(batch=batch,tokens_per_request=length,layers=layers,
                included_GEMM_FLOPs=layers*gemm_flops,included_causal_attention_FLOPs=layers*causal_attention_flops,
                GEMM_input_output_weight_bytes=layers*gemm_bytes,
                GEMM_arithmetic_intensity_FLOP_per_byte=gemm_flops/gemm_bytes,
                optimistic_included_work_lower_bound_ms=1000*max(
                    layers*(gemm_flops+causal_attention_flops)/peak,layers*gemm_bytes/bandwidth)))
    write(ROOT/'analysis/roofline_structural_accounting.json',dict(rows=rows,
        nominal_dense_BF16_peak_FLOPs_s=peak,nominal_HBM_bytes_s=bandwidth,
        source='https://www.nvidia.com/en-us/data-center/h100/',
        peak_scope='Nominal dense BF16 peak. NVIDIA lists twice this value with sparsity. The model is dense and does not use sparsity.',
        evidence='STRUCTURAL_ACCOUNTING',actual_HBM_traffic='NOT_MEASURED',actual_tensor_core_utilization='NOT_MEASURED',
        scope='Optimistic bound for layer GEMMs and causal Attention arithmetic. Exclude norms, RoPE, activation kernels, embedding, final head, launch, synchronization, and additional Attention/KV traffic. Admission uses measured costs, not this ideal bound.'))
    print('ROOFLINE_ACCOUNTING',len(rows))


if __name__=='__main__':main()
