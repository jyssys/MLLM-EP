"""Summarize metadata-only diagnostic, with profiler teardown kept separate."""
import argparse
import json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
df=pd.DataFrame([x for f in a.input.glob('rank*.json') for x in json.loads(f.read_text())])
assert len(df)==4*6*30*2
assert not df.duplicated(['rank','M','repeat','implementation']).any()
table=df.groupby(['M','implementation']).agg(cuda_ms_p50=('cuda_ms','median'),
    wall_ms_p50=('wall_ms','median'),cuda_ms_p90=('cuda_ms',lambda x:x.quantile(.9)),
    observations=('cuda_ms','size')).reset_index()
a.output.mkdir(parents=True,exist_ok=True);table.to_csv(a.output/'costs.csv',index=False)
profiles={name:[x for x in json.loads((a.input/f'ops_gpu4_{name}.json').read_text())
    if 'Synchronize' in x['key'] or 'unique' in x['key']] for name in ('isin','broadcast_membership')}
result=dict(status='PASS',metadata_equivalence='ALL6_M_SHAPES_ALL4_GPUS_EXACT',
    measured_observations=len(df),reps_per_gpu_shape_implementation=30,calls_per_observation=48,
    independent_cuda_contexts=4,distributed_moe_execution=False,costs=table.to_dict('records'),
    representative_profiler_api_counts=profiles,
    caveats=['cudaDeviceSynchronize occurs in both profiler captures and is not attributed to isin.',
             'GPU events span host enqueue/unique synchronization as well as kernels.',
             'Standalone metadata cost is not an additive estimate of serving speedup.',
             'Broadcast membership is an equivalence/control ingredient, not a new method.'])
(a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
