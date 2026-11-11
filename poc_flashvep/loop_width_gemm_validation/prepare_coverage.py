"""Cover actual batched P shapes before claiming a tuned serving comparison."""
from .common import *


def main():
    initialize();jobs=[]
    shapes={'huginn':[192,320,384,768,1280,1536,3584,3968,4032],
        'ouro':[6144,7680,8192]}
    for start in (1,2,3):
        for model in ('huginn','ouro'):
            cfg=json.loads((ROOT/'configs'/f'gemm_{model}_search_s{start}.json').read_text())
            cfg.update(name=f'gemm_{model}_batch_coverage_s{start}',rows=shapes[model],
                coverage_scope='Additional P batch and Token co-batch geometries. Existing start-1 choices remain frozen.')
            save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/batch_coverage_plan.json',dict(jobs=jobs,
        evidence='Observed N32 source batches include Huginn M3968 and Ouro M6144/M8192. These shapes must not receive asymmetric untuned fallback in the final comparison.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
