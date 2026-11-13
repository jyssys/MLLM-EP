"""Separate diagnostic kernel inactivity from qualified Clean Capacity."""
import argparse,csv,statistics
from .common import *
from .analyze import metrics,percentile


def union(intervals,left,right):
    merged=[]
    for a,b in sorted((max(a,left),min(b,right)) for a,b in intervals if b>left and a<right):
        if a>=b:continue
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return merged


def complement(merged,left,right):
    gaps=[];cursor=left
    for a,b in merged:
        if a>cursor:gaps.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<right:gaps.append([cursor,right])
    return gaps


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=1);a=p.parse_args();version=a.version
    name=f'sparse_Native_O32_N64_capture_v{version}';folder=ROOT/'raw'/name
    if not (folder/'dispatcher_result.json').exists():return
    state=json.loads((folder/'dispatcher_result.json').read_text())
    if not state['complete']:
        write(ROOT/f'analysis/sparse_physical_analysis_v{version}.json',dict(status='FAILED',name=name,receipt=state));return
    anchors={r:json.loads((folder/f'activity_{r}_sparse.json').read_text()) for r in ('prefill','decode')}
    left=max(v['start']['host_upper'] for v in anchors.values())
    right=min(v['end']['host_lower'] for v in anchors.values());assert left<right
    measured={}
    for role,value in anchors.items():
        anchor_ns=value['start']['cupti_ns'];anchor_host=value['start']['host_midpoint']
        data=list(csv.DictReader((folder/f'activity_{role}.csv').open()))
        spans=[(anchor_host+(int(r['start_ns'])-anchor_ns)/1e9,
            anchor_host+(int(r['end_ns'])-anchor_ns)/1e9) for r in data if r['kind']=='kernel']
        merged=union(spans,left,right);gaps=complement(merged,left,right)
        lengths=[1000*(b-a) for a,b in gaps];busy=sum(b-a for a,b in merged)
        interior=[1000*(b-a) for a,b in gaps if a>left and b<right]
        censored=sum(a==left or b==right for a,b in gaps)
        graph=sum(r['kind']=='kernel' and int(r['graph_id'])!=0 for r in data)
        measured[role]=dict(window_seconds=right-left,kernel_union_seconds=busy,kernel_busy_percent=100*busy/(right-left),
            kernel_inactive_seconds=(right-left)-busy,kernel_inactive_percent=100*(1-busy/(right-left)),
            idle_window_count=len(gaps),idle_mean_ms=statistics.mean(lengths) if lengths else None,
            idle_p50_ms=percentile(lengths,.5),idle_p95_ms=percentile(lengths,.95),
            idle_p99_ms=percentile(lengths,.99),idle_max_ms=max(lengths,default=None),actual_graph_kernel_records=graph,
            edge_censored_idle_windows=censored,
            interior_idle=dict(count=len(interior),mean_ms=statistics.mean(interior) if interior else None,
                p50_ms=percentile(interior,.5),p95_ms=percentile(interior,.95),
                p99_ms=percentile(interior,.99),max_ms=max(interior,default=None)),
            thresholds_ms={str(t):dict(count=sum(v>=t for v in lengths),total_seconds=sum(v/1000 for v in lengths if v>=t)) for t in (1,2,5,10,20,30,50,100)},
            counters=value['counters'],evidence='DIAGNOSTIC_MEASURED_KERNEL_UNION',
            scope='Union across streams. Copy records are separate. All-gap statistics include clipped edge windows. Interior statistics exclude them. Kernel inactivity is not a Decode-safe execution budget.')
        write(ROOT/'analysis'/f'diagnostic_{role}_intervals_v{version}.json',dict(left=left,right=right,union=merged,idle=gaps,
            evidence='DIAGNOSTIC_MEASURED_NOT_CLEAN_CAPACITY'))
    names=[f'sparse_Native_O32_N64_{x}_v{version}' for x in ('A','capture','C')]
    rows={n:metrics(n) for n in names};complete=all(r.get('complete') for r in rows.values())
    ratios={};aa={}
    keys=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','E2E_p95_ms','maximum_ITL_ms')
    if complete:
        a,b,c=(rows[n] for n in names)
        aa={k:c[k]/a[k] for k in keys}
        ratios={k:b[k]/((a[k]+c[k])/2) for k in keys}
    signatures=[]
    for n in names:
        source=ROOT/'raw'/n/'source_hashes.json'
        signatures.append({k.split('/worker_source/')[-1]:v for k,v in json.loads(source.read_text()).items()
            if k.split('/worker_source/')[-1] in {
                'poc_flashvep/dense_ar_dynamic_layer_pd/'+f for f in
                ('native.py','activity.py','boundaries.py','extension.py','watch.py')}})
    observer_matches=len(signatures[0])==5 and all(s==signatures[0] for s in signatures)
    passed=complete and observer_matches and all(.95<=v<=1.05 for v in ratios.values()) and all(.95<=v<=1.05 for v in aa.values())
    write(ROOT/f'analysis/sparse_physical_analysis_v{version}.json',dict(name=name,common_window=dict(start=left,end=right),
        diagnostic_measurements=measured,controls=rows,profile_to_mean_A_C=ratios,A_A_C_over_A=aa,
        complete=complete,observer_signatures=signatures,observer_revision_matched=observer_matches,interference_gate_passed=passed,
        Clean_physical_capacity='MEASURED_BOUNDED_WINDOW_ONLY' if passed else 'NOT_MEASURED',
        Decode_safe_capacity='NOT_MEASURED',
        scope='This conservative screen requires every primary tail and A/A ratio within5%. Failure does not prove zero physical headroom.'))
    print(json.dumps(dict(diagnostic_measurements=measured,interference_gate_passed=passed,ratios=ratios,aa=aa),indent=2))


if __name__=='__main__':main()
