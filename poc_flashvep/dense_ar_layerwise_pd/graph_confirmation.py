"""Confirm graph-family results on new paired arrival seeds."""
import json,random
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare
from .resume_campaign import restore,record


def main():
    progress=json.loads((ROOT/'state/graph_feasibility_progress_v28.json').read_text())['results']
    quals=[r for r in progress if 'passed' in r]
    assert len(quals)==6 and all(r['passed'] for r in quals)
    pilots=[r for r in progress if 'metrics' in r]
    native=next(r for r in pilots if '_native_' in r['name'])
    base=native['metrics'];candidates=[r for r in pilots if '_c35_' in r['name'] or '_c28_' in r['name']]
    extra=ROOT/'state/graph_extended_cuts_progress_v27.json'
    if extra.exists():candidates.extend(r for r in json.loads(extra.read_text())['results'] if 'metrics' in r)
    candidates=[r for r in candidates if r['output_gate']['passed'] and r['metrics']['Decode_absolute_SLO_pass']]
    if not candidates:
        write(ROOT/'analysis/graph_confirmation_selection.json',dict(passed=False,reason='NO_EXACT_ABSOLUTE_DECODE_SAFE_PILOT'))
        return
    safe=[r for r in candidates if r['metrics']['ITL_p99_ms']<=1.05*base['ITL_p99_ms'] and r['metrics']['TPOT_p95_ms']<=1.05*base['TPOT_p95_ms']]
    best=min(safe or candidates,key=lambda r:r['metrics']['TTFT_p95_ms'])
    colocated=next(r for r in pilots if '_colocated_' in r['name'])
    write(ROOT/'analysis/graph_confirmation_selection.json',dict(passed=True,selected=best['name'],
        pilot_Decode_noninferiority_pass=bool(safe),candidate_names=[r['name'] for r in candidates],
        scope='Selection uses only N64 pilot results. If no point-estimate protection gate passes, the N128 comparison remains bounded diagnostic confirmation.'))
    templates={k:json.loads((ROOT/'configs'/(r['name']+'.json')).read_text()) for k,r in [('native',native),('layer',best),('colocated',colocated)]}
    groups=[]
    for index,seed in enumerate((2026111231,2026111232,2026111233,2026111234,2026111235)):
        rng=random.Random(seed);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(base['offered_QPS']))
        order=[('native','layer','colocated'),('layer','colocated','native'),('colocated','native','layer'),
            ('colocated','layer','native'),('layer','native','colocated')][index]
        group=dict(seed=seed,arms={},order=list(order))
        for label in order:
            c=dict(templates[label],name=f'graph_confirm_qwen8_L8000_O16_N128_s{seed%100}_{label}_v27',
                count=128,request_caps=[16]*128,arrival_offsets=offsets,arrival_seed=seed,timeout=360,
                hidden_source_backpressure=label=='layer',phase='GRAPH_FAMILY_HELDOUT_ARRIVAL_CONFIRMATION',
                scope='Balanced independent Start order. Same 128 distinct controlled prefixes. Sixty-four prompts appeared in selection. Marker-free latency uses the same low-cost geometry counters in all arms. No sustainable capacity claim.')
            group['arms'][label]=save(c)
        groups.append(group)
    write(ROOT/'state/graph_confirmation_plan_v27.json',dict(groups=groups,selection=best['name']))
    try:
        for group in groups:
            for label in group['order']:
                name=group['arms'][label];complete=execute(name,restore=False)
                row=dict(name=name,complete=complete)
                if complete:row['metrics']=metrics(name)
                record('graph_N128_confirmation',row)
                if not complete:return
            checks=[compare(group['arms']['native'],group['arms'][label]) for label in ('layer','colocated')]
            record('graph_N128_outputs',dict(seed=group['seed'],checks=checks,passed=all(r['passed'] for r in checks)))
            if not all(r['passed'] for r in checks):return
    finally:restore()


if __name__=='__main__':main()
