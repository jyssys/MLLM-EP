"""Keep unqualified activity traces separate from clean decode-slack evidence."""
import csv
import json
from .common import ROOT, write
from .analyze import pct, requests


def union(intervals):
    merged=[]
    for left,right in sorted(intervals):
        if right<=left:
            continue
        if merged and left<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(right,merged[-1][1]))
        else:
            merged.append((left,right))
    return merged


def analyze(name,clean):
    folder,reference=ROOT/'raw'/name,ROOT/'raw'/clean
    capture=json.loads((folder/'activity_capture_decode.json').read_text())
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    start,stop=capture['start'],capture['stop']
    lower,upper=start['device_clock_ns'],stop['device_clock_ns']
    mapping=json.loads((folder/'worker_map.json').read_text())
    device=next(v['visible_device'] for v in mapping.values() if v['role']=='decode')
    values=list(csv.DictReader((folder/'kernel_activity_decode.csv').open()))
    intervals=[(max(lower,int(r['start_ns'])),min(upper,int(r['end_ns']))) for r in values
        if int(r['device'])==device and int(r['end_ns'])>=lower and int(r['start_ns'])<=upper]
    merged=union(intervals)
    gaps=[(right[0]-left[1])/1e6 for left,right in zip(merged,merged[1:])]
    window=json.loads((folder/'window.json').read_text());base=json.loads((reference/'window.json').read_text())
    offset=start['monotonic_s']-window['start']
    length=stop['monotonic_s']-start['monotonic_s']
    def summaries(path,origin):
        rows=requests(path);events=[];tpots=[]
        for r in rows:
            if r['first_token']-origin<=offset+length+.5 and r['finish']-origin>=offset-.5:
                tpots.append(r['tpot_ms'])
            for e in r['token_events']:
                if e['first'] or e['new_tokens']!=1:
                    continue
                # Include intervals that straddle a collection boundary.
                if e['previous_time']-origin<=offset+length+.5 and e['time']-origin>=offset-.5:
                    events.append(1000*(e['time']-e['previous_time']))
        return dict(request_TPOT_p95_ms=pct(tpots,95),ITL_p95_ms=pct(events,95),
            ITL_p99_ms=pct(events,99),max_ITL_ms=max(events,default=None),samples=len(events))
    measured=summaries(folder,window['start']);control=summaries(reference,base['start'])
    ratios={k:measured[k]/control[k] for k in ('request_TPOT_p95_ms','ITL_p95_ms','ITL_p99_ms','max_ITL_ms')
        if measured[k] is not None and control[k] is not None and control[k]>0}
    overall=(window['end']-window['start'])/(base['end']-base['start'])
    qualified=bool(capture['dropped']==capture['invalid']==capture['status']==0 and merged
        and len(ratios)==4 and max(ratios.values())<=1.05 and overall<=1.05)
    result=dict(name=name,clean=clean,model=cfg['model'],qualified=qualified,
        collection_seconds=length,kernels=len(intervals),
        kernel_union_fraction=sum(right-left for left,right in merged)/(upper-lower),
        kernel_free_gap_p50_ms=pct(gaps,50),kernel_free_gap_p95_ms=pct(gaps,95),
        kernel_free_gap_p99_ms=pct(gaps,99),max_kernel_free_gap_ms=max(gaps,default=None),
        local_profile=measured,local_clean=control,local_ratios=ratios,cohort_duration_ratio=overall,
        gate='All four local decode ratios and cohort duration ratio <=1.05; no lost/invalid kernels.',
        scope='Kernel-free intervals exclude observed compute kernels only. DMA and other device resources can be active. A gap is not a deadline-safe insertion budget. Unqualified traces are diagnostic only.')
    write(folder/'dlocal_profile_qualification.json',result)
    write(folder/'kernel_gap_samples.json',dict(gaps_ms=gaps,qualified=qualified))
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(analyze(sys.argv[1],sys.argv[2])))
