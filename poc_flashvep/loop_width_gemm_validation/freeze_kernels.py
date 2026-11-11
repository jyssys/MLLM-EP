"""Freeze candidates from the first start. Later starts are confirmation."""
from .common import *
from .micro_analysis import choices


def main():
    initialize();models={}
    for model in ('huginn','ouro'):
        path=ROOT/'raw'/f'gemm_{model}_search_s1'/'projection_results.json'
        value=json.loads(path.read_text());entries=[]
        for row in value['results']:
            best,_=choices(row)
            # Do not add a new dispatch for a negligible point improvement.
            if best['ms']>row['reference_graph']['p50_ms']*.98:
                best=dict(method='qualified_reference',ms=row['reference_graph']['p50_ms'])
            entries.append(dict(role=row['role'],M=row['rows'],N=row['weight_shape'][0],K=row['weight_shape'][1],**best))
        models[model]=dict(entries=entries,selection_source=str(path),selection_source_SHA256=digest(path),
            cublasLt_version=value['metadata']['cublasLt_version'])
    target=ROOT/'configs/operator_kernel_policy_s1.json'
    assert not target.exists(),'The runtime policy is immutable'
    write(target,dict(models=models,library=json.loads((ROOT/'tools/lt_build.json').read_text()),
        scope='Operator candidates selected only from independent start 1. The policy is not model-qualified until output/KV/Hidden tests pass. Native and all split methods use the same map.'))
    print(target,digest(target))


if __name__=='__main__':main()
