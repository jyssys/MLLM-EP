"""Qualify model-stream enclosures without calling them kernel activity."""
from .common import *
from .analyze import pct,requests,table
from .profile_analysis import union


def analyze(model):
    name=f'local_{model}_native_event_trace128'
    folder=ROOT/'raw'/name;reference=ROOT/'raw'/f'local_{model}_native_n128_s2026110701_off'
    traces=list(folder.glob('dlocal_trace_*.json'))
    if not traces:return None
    trace=json.loads(traces[0].read_text())
    anchor=(trace['anchor_host_lower']+trace['anchor_host_upper'])/2
    ops=trace['operations']
    intervals=union([(anchor+v['begin_device_ms']/1000,anchor+v['end_device_ms']/1000) for v in ops])
    window=json.loads((folder/'window.json').read_text());base=json.loads((reference/'window.json').read_text())
    lower,upper=intervals[0][0]-window['start'],intervals[-1][1]-window['start']
    def metrics(path,origin):
        itl=[];tpot=[]
        for r in requests(path):
            if r['first_token']-origin<=upper+.5 and r['finish']-origin>=lower-.5:tpot.append(r['tpot_ms'])
            for e in r['token_events']:
                if not e['first'] and e['new_tokens']==1 and e['previous_time']-origin<=upper+.5 and e['time']-origin>=lower-.5:
                    itl.append(1000*(e['time']-e['previous_time']))
        return dict(TPOT_p95_ms=pct(tpot,95),ITL_p95_ms=pct(itl,95),ITL_p99_ms=pct(itl,99),max_ITL_ms=max(itl,default=None),samples=len(itl))
    measured=metrics(folder,window['start']);clean=metrics(reference,base['start'])
    ratios={k:measured[k]/clean[k] for k in ('TPOT_p95_ms','ITL_p95_ms','ITL_p99_ms','max_ITL_ms') if measured[k] is not None and clean[k]}
    duration_ratio=(window['end']-window['start'])/(base['end']-base['start'])
    qualified=len(ratios)==4 and max(ratios.values())<=1.05 and duration_ratio<=1.05
    gaps=[1000*(b[0]-a[1]) for a,b in zip(intervals,intervals[1:])]
    rows=[]
    for v in ops:
        rows.append(dict(operation=v['operation'],loop_count=v['loop_count'],enclosure_ms=v['enclosure_ms'],
            rows=sum(s['count'] for s in v['spans']),host_submit=v['host_submit'],
            device_start=anchor+v['begin_device_ms']/1000,device_end=anchor+v['end_device_ms']/1000))
    table(folder/'model_stream_operations.csv',rows)
    result=dict(model=model,name=name,qualified=qualified,local_marked=measured,local_clean=clean,local_ratios=ratios,
        cohort_duration_ratio=duration_ratio,operations=len(ops),host_anchor_uncertainty_ms=1000*(trace['anchor_host_upper']-trace['anchor_host_lower']),
        gap_p50_ms=pct(gaps,50),gap_p95_ms=pct(gaps,95),gap_p99_ms=pct(gaps,99),max_gap_ms=max(gaps,default=None),
        interval_union_fraction=sum(b-a for a,b in intervals)/(intervals[-1][1]-intervals[0][0]),
        evidence='MEASURED_CUDA_EVENT_MODEL_STREAM_ENCLOSURES',
        kernel_union='NOT_MEASURED',SM_active='NOT_MEASURED',memory_bandwidth='NOT_MEASURED',
        scope='Enclosures contain kernel work and launch gaps. Queue snapshots are host queues, not guaranteed device readiness. These intervals do not establish hard safe insertion deadlines.')
    write(folder/'event_trace_qualification.json',result)
    write(folder/'event_enclosure_intervals.json',dict(intervals=intervals,gaps_ms=gaps,qualified=qualified))
    print(json.dumps(result))
    return result


if __name__=='__main__':
    import sys
    for model in sys.argv[1:] or ('ouro','huginn'):analyze(model)
