"""A conditional cold-matrix model. This is not a measured hardware bound."""
from .common import *
from .micro_analysis import csv_write


def main():
    initialize();results=[]
    # NVIDIA lists 1979 TFLOP/s with sparsity. Dense BF16 uses half that value.
    peak=989.5e12;bandwidth=3.35e12
    for model in ('huginn','ouro'):
        raw=json.loads((ROOT/'raw'/f'gemm_{model}_search_s1/projection_results.json').read_text())
        depth=32 if model=='huginn' else 4;length=2048 if depth==32 else 4096
        projections=[r for r in raw['results'] if r['rows']==length]
        for batch in (1,2,4,8):
            loop=token=0.;loop_flops=token_flops=0.;loop_bytes=token_bytes=0.
            for row in projections:
                n,k=row['weight_shape'];repeat=row['multiplicity_per_loop']
                def components(m):
                    f=2*m*n*k;b=2*(m*k+n*k+m*n)
                    return max(f/peak,b/bandwidth)*1000,f,b
                t,f,b=components(length);loop+=repeat*t;loop_flops+=repeat*f;loop_bytes+=repeat*b
                t,f,b=components(length//depth*batch)
                token+=repeat*t*depth/batch;token_flops+=repeat*f*depth/batch;token_bytes+=repeat*b*depth/batch
            results.append(dict(model=model,token_batch=batch,loop_ms=loop,token_ms=token,
                token_over_loop=token/loop,loop_dense_FLOPs=loop_flops,token_dense_FLOPs=token_flops,
                loop_cold_matrix_bytes=loop_bytes,token_cold_matrix_bytes=token_bytes,
                scope='MODELED cold-matrix roofline. Every GEMM reads each matrix once from HBM and writes its result once. Cache reuse, fusion, launch cost, Attention, packing, and ready-work availability are absent. This is not an optimal-kernel proof, a measured HBM counter, or serving speedup.'))
    write(ROOT/'analysis/conditional_cold_matrix_roofline.json',dict(
        hardware_assumption=dict(device='H100 SXM',dense_BF16_peak_FLOPs_s=peak,HBM_peak_bytes_s=bandwidth,
            source='https://www.nvidia.com/en-us/data-center/h100/',
            scope='Product specification assumptions, not sustained rates measured on this node.'),results=results))
    csv_write(ROOT/'analysis/conditional_cold_matrix_roofline.csv',results);print(json.dumps(results))


if __name__=='__main__':main()
