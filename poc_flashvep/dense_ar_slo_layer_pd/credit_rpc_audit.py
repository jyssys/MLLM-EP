"""Separate observed credit RPC waiting from unmeasured GPU idle."""
from .common import *
from .analyze import records,percentile


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if not result or start>result[-1][1]:result.append([start,end])
        else:result[-1][1]=max(result[-1][1],end)
    return result


def audit(name):
    folder=ROOT/'raw'/name;window=json.loads((ROOT/'analysis'/(name+'_sustainable_v2.json')).read_text())
    begin=window['window_begin'];end=window['window_end'];steps=[]
    for p in folder.glob('native_decode_*.json'):steps.extend((r['start'],r['end']) for r in json.loads(p.read_text())['host_step_spans'])
    steps=union(steps);observed=[];intervals=[]
    for row in records(folder/'reservation_observations.jsonl'):
        finish=row['source_receive'];start=finish-row['negotiation_ms']/1000
        if finish<=begin or start>=end:continue
        clipped=(max(start,begin),min(finish,end));intervals.append(clipped)
        overlap=sum(max(0.,min(clipped[1],b)-max(clipped[0],a)) for a,b in steps)
        observed.append(dict(generation=row['generation'],start=start,end=finish,
            negotiation_ms=row['negotiation_ms'],D_Host_step_overlap_ms=1000*overlap))
    duration=sum(b-a for a,b in union(intervals));fraction=duration/(end-begin)
    result=dict(name=name,samples=len(observed),measurement_seconds=end-begin,
        negotiation_ms=dict(p50=percentile([r['negotiation_ms'] for r in observed],.5),
            p95=percentile([r['negotiation_ms'] for r in observed],.95),maximum=max((r['negotiation_ms'] for r in observed),default=None)),
        negotiation_interval_union_seconds=duration,window_fraction=fraction,
        D_Host_step_overlap_seconds=sum(r['D_Host_step_overlap_ms'] for r in observed)/1000,
        modeled_instant_RPC_raw_tokens_s=window['raw_completed_output_tokens_s']/(1-fraction),
        observations=observed,evidence='TRACE_ANALYSIS',
        scope='The Source waits for prospective credit before submitting its next prefix. Intervals include sender, socket, destination dispatch and reply. Destination engine.step is a Host enclosure, not a kernel interval. Overlap does not isolate causal shares. The instant-RPC rate removes all observed wait while holding every other cost and action fixed. It is an optimistic component model, not measured speedup, physical idle, or a capacity guarantee.')
    write(ROOT/'analysis'/(name+'_credit_RPC_audit.json'),result)
    print('CREDIT_RPC',name,duration,fraction,result['modeled_instant_RPC_raw_tokens_s'],flush=True)
    return result


if __name__=='__main__':audit('canonical_capacity_Static12_O16_q5.0_s2026111462_W600_v1')
