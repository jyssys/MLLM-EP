"""CPU-only analysis of selected, explicitly instrumented real EP route samples."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[json.loads(s) for f in a.input.glob('stages.pid*.jsonl') for s in f.read_text().splitlines()]
stats=[];totals=defaultdict(lambda:dict(tokens=0,assignments=0,removed=0,rerouted=0,samples=0))
for record in rows:
    if not record.get('route_file'):continue
    route=torch.load(record['route_file'],map_location='cpu',weights_only=True)
    valid=route['token_ids']!=-1
    for modality,mask in [('vision',route['vision']&valid),('text',route['text']&valid),('all_valid',valid)]:
        original=route['original_ids'][mask];changed=route['ids'][mask]
        if not len(original):continue
        removed=int((changed<0).sum());rerouted=int(((changed>=0)&(changed!=original)).sum())
        row={k:record[k] for k in ('invocation','step','layer','ep_rank','dp_rank','policy','phase','cohort')}
        row.update(modality=modality,tokens=len(original),assignments=original.numel(),
            removed=removed,rerouted=rerouted,
            original_active_experts=int(original.unique().numel()),
            final_active_experts=int(changed[changed>=0].unique().numel()))
        stats.append(row)
        target=totals[(record['policy'],record['phase'],modality)]
        for key in ('tokens','assignments','removed','rerouted'):target[key]+=row[key]
        target['samples']+=1
summary=[]
for (policy,phase,modality),x in sorted(totals.items()):
    summary.append(dict(policy=policy,source_phase=phase,modality=modality,**x,
        removed_fraction=x['removed']/x['assignments'],rerouted_fraction=x['rerouted']/x['assignments'],
        scope='SELECTED_LAYERS_FIRST8_STEPS_PER_POLICY_INCLUDES_WARMUP_NOT_FULL_WORKLOAD_ESTIMATE'))
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'rank_route_samples.json').write_text(json.dumps(stats))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
