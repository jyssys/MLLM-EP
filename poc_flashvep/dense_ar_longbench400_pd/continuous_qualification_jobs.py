"""Exercise variable natural outputs and real mixed rows before performance claims."""
import json
import numpy as np
from .common import ROOT,write
from .experiment import create
from .next_jobs import plan


def main():
    rows=[]
    for bucket in range(4):
        rows.extend(json.loads((ROOT/f'data/input{bucket}_discovery_exact16.json').read_text())[:8])
    np.random.default_rng(2026112182).shuffle(rows)
    path=ROOT/'data/continuous_qualification32_natural.json'
    if path.exists():assert json.loads(path.read_text())==rows
    else:write(path,rows)
    jobs=[]
    for method,cut in [('Native',0),('Layer',4),('Token',4),('Hybrid',4)]:
        extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if method=='Hybrid' else {}
        jobs.append(create(f'C3Q5_{method}{cut}_Natural32_v1',method,path.name,cut=cut,
            pair=(2,3),qps=8,seed=2026112183,correctness=True,axis_state_capture=False,
            continuous_source_credit_limit=4,wall_timeout=900,serving_timeout=700,**extra))
    plan('continuous_natural32_qualification_v1',jobs)
    print('Frozen C3 Natural32 qualification jobs.',flush=True)


if __name__=='__main__':main()
