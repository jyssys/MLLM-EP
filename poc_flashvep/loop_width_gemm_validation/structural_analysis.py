"""Count logical work and an explicit GEMM traffic model, not hardware traffic."""
from .common import *
from .micro_analysis import csv_write


def pairs(start,count):
    return ((start+count)*(start+count+1)-start*(start+1))//2


def main():
    initialize();rows=[];families=[]
    for model in ('huginn','ouro'):
        value=json.loads((ROOT/'raw'/f'gemm_{model}_search_s1/projection_results.json').read_text())
        depth=32 if model=='huginn' else 4;length=2048 if depth==32 else 4096;suffix=length//depth
        weights=0
        for row in value['results']:
            if row['rows']!=length:continue
            n,k=row['weight_shape'];multiplicity=row['multiplicity_per_loop'];weights+=2*n*k*multiplicity
            for m in (suffix,length):
                flops=2*m*n*k
                traffic=2*(m*k+n*k+m*n)
                rows.append(dict(model=model,role=row['role'],M=m,N=n,K=k,multiplicity=multiplicity,
                    logical_dense_FLOPs=flops,one_read_matrix_bytes=traffic,
                    one_read_arithmetic_intensity=flops/traffic,weight_bytes=2*n*k,
                    scope='Structural BF16 matrix model. Each input and weight is read once and output is written once. This is not measured HBM traffic, a cache model, or an optimal-kernel lower bound.'))
        families.append(dict(model=model,length=length,depth=depth,token_suffix=suffix,
            loop_recurrent_rows=length,token_recurrent_rows=suffix*depth,
            loop_causal_pairs_per_core_layer=pairs(0,length),
            token_causal_pairs_per_core_layer=pairs(length-suffix,suffix)*depth,
            recurrent_projection_weight_bytes_per_pass=weights,
            serial_token_weight_passes=depth,serial_loop_weight_passes=1,
            token_prelude_tokens=suffix if model=='huginn' else 0,
            loop_prelude_tokens=0,
            token_prompt_coda_tokens=suffix-1 if model=='huginn' else 0,
            loop_prompt_coda_tokens=length-1 if model=='huginn' else 0,
            scope='Logical D-side work. Recurrent rows do not imply equal Attention, Prelude, Coda, FLOPs, or P saving. Repeated weight passes are runtime structure, not measured memory transfers.'))
    csv_write(ROOT/'analysis/gemm_geometry_accounting.csv',rows)
    csv_write(ROOT/'analysis/loop_token_structural_accounting.csv',families)
    write(ROOT/'analysis/structural_accounting.json',dict(projections=rows,families=families,
        interpretation='A faster GEMM can reduce padding and launch overhead. It does not remove recurrent Hidden/Attention dependencies. Batching, persistent kernels, cache reuse, or larger Token suffixes can change the performance frontier. No universal Loop dominance follows.'))
    print(json.dumps(families))


if __name__=='__main__':main()
