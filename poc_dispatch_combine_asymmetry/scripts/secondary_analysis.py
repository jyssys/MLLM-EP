import collections
import json
from pathlib import Path

import numpy as np

from poc_dispatch_combine_asymmetry.contract import ROOT, write
from poc_dispatch_combine_asymmetry.scripts.analyze import ci


def main():
    groups=collections.defaultdict(list)
    for file in (ROOT/'artifacts/route_controlled_replay').glob('rank*.json'):
        for r in json.loads(file.read_text()):groups[r['arm'],r['size'],r['repeat']].append(r)
    summaries=[]
    for (arm,size,rep),rows in groups.items():
        ranks=collections.defaultdict(list)
        for r in rows:ranks[r['ep_rank']].append(r)
        assert set(ranks)==set(range(4)) and all(len(v)==3 for v in ranks.values())
        rank=max(ranks,key=lambda k:sum(r['dispatch_ms']+r['combine_ms'] for r in ranks[k]))
        selected=ranks[rank]
        summaries.append({'arm':arm,'size':size,'repeat':rep,'critical_rank':rank,
                          'logical_prompt_tokens':selected[0]['logical_prompt_tokens'],
                          **{k:float(np.mean([r[k] for r in selected])) for k in ('layout_ms','dispatch_ms','combine_ms')}})
    result={}
    for size in ('low','medium','high'):
        a=[r for r in summaries if r['size']==size and r['arm']=='visual']
        b={r['repeat']:r for r in summaries if r['size']==size and r['arm']=='text'}
        result[size]={'visual':{k:ci([r[k] for r in a]) for k in ('layout_ms','dispatch_ms','combine_ms')},
                      'text':{k:ci([r[k] for r in b.values()]) for k in ('layout_ms','dispatch_ms','combine_ms')},
                      'visual_minus_text_dispatch_ms':ci([r['dispatch_ms']-b[r['repeat']]['dispatch_ms'] for r in a]),
                      'visual_minus_text_dispatch_percent':ci([100*(r['dispatch_ms']/b[r['repeat']]['dispatch_ms']-1) for r in a])}
    write(ROOT/'artifacts/replay_summary.json',{'evidence':'ROUTE_CONTROLLED_TRUE_EP_REPLAY','metrics':result,'records':summaries,
        'scope':'only captured layers 0/24/47, aggregate within repeat, no 48-layer extrapolation',
        'source_layout':'TP0/TP1 split, SP padding plus one native dummy row on each idle-DP source retained',
        'ready_barrier':'explicit all-rank barrier outside CUDA timers, vision/attention/host producers removed',
        'experts':'no expert execution; received hidden tensor used as representative combine payload'})
    overhead={}
    for tag in ('llm_controls_v3','mllm_internal_v3'):
        records=json.loads((ROOT/f'artifacts/runs/{tag}/worker.dp0.json').read_text())['records']
        per=collections.defaultdict(dict)
        for r in records:
            if r.get('overhead_pair') and r['repeat']>=0:per[r['arm'],r['repeat']][r['lane']]=r['host_wave_ms']
        for arm in set(a for a,_ in per):
            pairs=[v for (a,_),v in per.items() if a==arm]
            overhead[tag+'_'+arm]=ci([100*(v['clean']/v['bypass']-1) for v in pairs])
    native=json.loads((ROOT/'artifacts/final_metrics.json').read_text())['metrics']
    perturbation={}
    for key,value in native.items():
        if not key.endswith('_internal'):continue
        clean=native[key.replace('_internal','_clean')]
        perturbation[key]={k:100*(value[k]['mean']/clean[k]['mean']-1) for k in ('dispatch_ms','combine_ms','moe_ms')}
        perturbation[key]['status']='UNVALIDATED_PERTURBED_INTERNAL_DURATION' if max(abs(perturbation[key][k]) for k in ('dispatch_ms','combine_ms','moe_ms'))>10 else 'DIAGNOSTIC_ONLY'
    write(ROOT/'artifacts/overhead.json',{'paired_hook_bypass_host_wave':overhead,'internal_vs_clean_unpaired_component_perturbation':perturbation,
        'limitation':'host wave includes DP idle/drain coordination; not an isolated MoE overhead guarantee; internal timing never headline'})


if __name__=='__main__':main()
