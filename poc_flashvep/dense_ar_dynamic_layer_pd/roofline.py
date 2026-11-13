"""Report operation accounting and vendor ceilings, not measured GPU counters."""
from .common import *


def main():
    length=8000;hidden=4096;intermediate=12288;qkv=6144;layers=36
    weights=hidden*qkv+hidden*hidden+2*hidden*intermediate+intermediate*hidden
    rows=[]
    for batch in (1,2,3,4,8,16):
        for suffix in (2,4,8,12,36):
            linear_FLOPs=2*weights*batch*length*suffix
            attention_FLOPs=4*hidden*batch*length*(length+1)/2*suffix
            weight_bytes=2*weights*suffix
            KV_bytes=2*2*8*128*batch*length*suffix
            # Lower-bound bytes exclude intermediate activation reads and writes.
            minimum_bytes=weight_bytes+KV_bytes
            peak_dense=1979e12/2;peak_bandwidth=3.35e12
            GEMMs=[dict(operation=label,M=batch*length,N=n,K=k,FLOPs=2*batch*length*n*k*suffix)
                for label,n,k in (('QKV',6144,4096),('Attention_output',4096,4096),
                    ('Gate_up',24576,4096),('Down',4096,12288))]
            rows.append(dict(batch=batch,token_length=length,suffix_layers=suffix,GEMM_shapes=GEMMs,
                linear_FLOPs=linear_FLOPs,causal_attention_FLOPs=attention_FLOPs,
                linear_weight_bytes=weight_bytes,KV_write_bytes=KV_bytes,
                minimum_bytes=minimum_bytes,optimistic_arithmetic_intensity=(linear_FLOPs+attention_FLOPs)/minimum_bytes,
                optimistic_roofline_floor_ms=1000*max((linear_FLOPs+attention_FLOPs)/peak_dense,minimum_bytes/peak_bandwidth),
                evidence='STRUCTURAL_ACCOUNTING',
                scope='Vendor peak ceiling and minimal byte accounting. No measured FLOPs, HBM traffic, Tensor Core utilization, Decode slack, or latency guarantee.'))
    write(ROOT/'analysis/roofline_operation_accounting.json',dict(rows=rows,
        dense_BF16_TFLOPs_s=989.5,HBM_TB_s=3.35,
        source='https://www.nvidia.com/en-us/data-center/h100/',
        peak_note='The vendor 1979 TFLOPs/s BF16 specification includes sparsity. Divide it by two for this dense theoretical ceiling.',
        limitations='Attention traffic, activation traffic, launch gaps, graph and Host costs require empirical calibration. A roofline floor is not an execution budget.'))


if __name__=='__main__':main()
