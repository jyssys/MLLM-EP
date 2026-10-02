"""Request/repeat-level critical-rank aggregation; never bootstrap layers."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from poc_dispatch_combine_asymmetry.contract import ROOT, write


def ci(values):
    arr=np.asarray(values,dtype=float)
    rng=np.random.default_rng(3407)
    samples=arr[rng.integers(len(arr),size=(2000,len(arr)))].mean(axis=1)
    return {'n':len(arr),'mean':float(arr.mean()),'median':float(np.median(arr)),
            'p90':float(np.percentile(arr,90)), 'ci95':np.percentile(samples,[2.5,97.5]).tolist(),
            'bootstrap_unit':'request_repeat_not_layer','bootstrap_replicates':2000}


def critical(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[row['ep_rank']].append(row)
    if set(groups)!=set(range(4)):
        raise ValueError(f'incomplete ranks: {list(groups)}')
    for rank, layer_rows in groups.items():
        if len(layer_rows)!=48 or sorted(r['layer'] for r in layer_rows)!=list(range(48)):
            raise ValueError(f'rank {rank}: expected exactly 48 layers, got {len(layer_rows)}')
    rank=max(groups,key=lambda r:sum(x['moe_ms'] for x in groups[r]))
    selected=groups[rank]
    result={key:sum(r[key] for r in selected)/48 for key in ('dispatch_ms','expert_ms','combine_ms','moe_ms')}
    result['ratio']=result['dispatch_ms']/result['combine_ms']
    result['critical_rank']=rank
    result['max_dispatch_layer_ms']=max(r['dispatch_ms'] for r in selected)
    result['subcomponents']={name:sum(r['subcomponents'][name]['ms'] for r in selected)/48
        for name in ('get_dispatch_layout','dispatch','combine') if all(name in r.get('subcomponents',{}) for r in selected)}
    return result


def native(run):
    waves=collections.defaultdict(list)
    for path in run.glob('stage.pid*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['repeat']>=0:waves[row['wave']].append(row)
    summaries=[]
    for wave, rows in sorted(waves.items()):
        data=critical(rows)
        record=rows[0]
        summaries.append({**{k:record[k] for k in ('model','arm','size','repeat','lane','prompt_tokens')},
                          'overhead_pair':record.get('overhead_pair',False),
                          'wave':wave,'run':str(run),**data})
    return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    summaries=[]
    for run in args.runs:summaries.extend(native(run))
    groups=collections.defaultdict(list)
    for row in summaries:
        if not row['overhead_pair']:
            groups[f"{row['model']}_{row['arm']}_{row['size']}_{row['lane']}"].append(row)
    metrics={key:{name:ci([r[name] for r in rows]) for name in ('dispatch_ms','expert_ms','combine_ms','moe_ms','ratio','max_dispatch_layer_ms')}
             for key,rows in groups.items()}
    for key,rows in groups.items():
        metrics[key]['prompt_tokens']=rows[0]['prompt_tokens']
        metrics[key]['subcomponents']={name:ci([r['subcomponents'][name] for r in rows])
              for name in rows[0]['subcomponents']}
    comparisons={}
    for size in ('high','medium','low'):
        a=groups.get(f'mllm_visual_{size}_clean',[]);b=groups.get(f'mllm_text_{size}_clean',[])
        if a and b:
            index={r['repeat']:r for r in b}
            paired=[(r,index[r['repeat']]) for r in a if r['repeat'] in index]
            comparisons[size]={name:ci([x[name]-y[name] for x,y in paired])
                               for name in ('dispatch_ms','expert_ms','combine_ms')}
            comparisons[size]['dispatch_percent']=ci([100*(x['dispatch_ms']/y['dispatch_ms']-1) for x,y in paired])
    write(args.output,{'critical_rule':'one rank maximizing cumulative 48-layer D+E+C; all components from that rank; local durations not a global E2E clock',
        'summaries':summaries,'metrics':metrics,'visual_minus_text':comparisons})
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
