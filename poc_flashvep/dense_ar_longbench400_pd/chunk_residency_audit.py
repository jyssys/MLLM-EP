"""Measure continuation fragmentation and intervening host decode steps."""
import bisect
import collections
import json
import math
import time
import numpy as np
from .common import ROOT,write
from .window_analysis import duration,overlap


def analyze(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows']
    steps=json.loads((folder/'engine_steps_decode.json').read_text())['rows']
    starts=[r['start'] for r in steps]
    phase=collections.defaultdict(list)
    decode_steps=set()
    unmapped=0
    for q in queries:
        index=bisect.bisect_right(starts,q['time'])-1
        if index<0 or q['time']>steps[index]['end']:
            unmapped+=1
            continue
        if q['start']>=q['original_input_tokens']:
            decode_steps.add(index)
        else:
            phase[q['request_id']].append((q,index))
    parts={}
    path=folder/'axis_events_decode.json'
    if path.exists():
        parts={r['request_id']:r for r in json.loads(path.read_text())['rows'] if r['event']=='PARTITION'}
    rows=[]
    for rid,prefills in sorted(phase.items()):
        if not rid.startswith('q') or not rid[1:].isdigit():
            continue
        prefills=sorted(prefills,key=lambda x:x[0]['time'])
        part=parts.get(rid)
        total=sum(q['count'] for q,_ in prefills)
        if cfg.get('axis_kind')=='Hybrid' and cfg.get('partition_aligned_chunks') and not (
            cfg.get('fuse_short_hybrid') and part['length']<=8192):
            minimum=math.ceil(part['prefix']/8192)+math.ceil(part['suffix']/8192)
        else:
            minimum=math.ceil(total/8192)
        gaps=[]
        intervening=[]
        for (_,a),(_,b) in zip(prefills,prefills[1:]):
            if b<=a:
                continue
            left,right=steps[a]['end'],steps[b]['start']
            if right>left:
                gaps.append((left,right))
                intervening.extend((steps[j]['start'],steps[j]['end']) for j in range(a+1,b) if j in decode_steps)
        rows.append(dict(request_id=rid,D_prompt_query_tokens=total,prefill_model_calls=len(prefills),
            ideal_chunk_lower_bound=minimum,excess_calls_over_lower_bound=len(prefills)-minimum,
            prefill_query_sizes=[q['count'] for q,_ in prefills],
            inter_prefill_host_gap_ms=1000*duration(gaps),
            inter_prefill_gap_overlap_other_decode_host_ms=1000*duration(overlap(gaps,intervening))))
    assert rows and unmapped==0,(name,unmapped)
    fields=('D_prompt_query_tokens','prefill_model_calls','ideal_chunk_lower_bound',
        'excess_calls_over_lower_bound','inter_prefill_host_gap_ms','inter_prefill_gap_overlap_other_decode_host_ms')
    result=dict(name=name,rows=rows,means={f:float(np.mean([r[f] for r in rows])) for f in fields},
        requests=len(rows),requests_with_extra_chunks=sum(r['excess_calls_over_lower_bound']>0 for r in rows),
        requests_with_intervening_decode=sum(r['inter_prefill_gap_overlap_other_decode_host_ms']>0 for r in rows),
        evidence='TRACE_ANALYSIS',
        scope='The B8192 lower bound counts query tokens,not token-layer rows. '
            'The aligned Hybrid bound includes its mandatory prefix/suffix boundary. '
            'Extra calls can reflect legal multi-request packing,not an implementation error. '
            'Intervening decode time is an overlap of host engine-step enclosures,not exclusive GPU compute or causal slowdown. '
            'An earlier first prepare can move waiting inside this enclosure without changing completion time. '
            'This audit does not prove that disabling packing improves serving.')
    write(ROOT/'analysis'/(name+'_chunk_residency_audit.json'),result)
    return result


def main():
    cases=[]
    for qps in (1.8,2.8):
        for label in ('NativeSetup','LayerSetup','TokenSetup','HybridSetup'):
            name=f'RemoteSetupAB_Discovery200_q{qps}_{label}_v1'
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text())['complete']:
                continue
            r=analyze(name)
            cases.append(r)
            print(name,r['means'],flush=True)
    write(ROOT/'analysis/continuation_chunk_residency_summary_v1.json',dict(time=time.time(),cases=cases,
        evidence='TRACE_ANALYSIS',scope='Paired discovery traces. Cut4 pure axes and Hybrid4/f1/8 have different D rows. '
            'Use the iso8 and credit-controlled traces for matched-work causal comparisons. '
            'No numerical or timestamp changes. Keep the independent intervention requirement.'))


if __name__=='__main__':
    main()
