"""Retain graph enclosure semantics when constructing an inactivity bound."""
import argparse,csv,json
from .common import *
from .analyze import gaps,summary,csv_write,native,local_itl
from .gates import interference


def physical_bound(name,padding_ms=.05):
    folder=ROOT/'raw'/name
    meta={r:json.loads((folder/f'physical_activity_{r}.json').read_text()) for r in ('prefill','decode')}
    assert not any(m.get('quiescent_only') for m in meta.values())
    assert meta['decode']['graph_enclosures'] and not meta['prefill']['graph_enclosures']
    assert all(m['status']==m['counters']['dropped']==0 and
        (m['counters']['invalid']==0 or m.get('window_record_scope',{}).get('passed')) for m in meta.values())
    left=max(m['start']['host_midpoint'] for m in meta.values())
    right=min(m['stop']['host_midpoint'] for m in meta.values());assert right>left
    devices={r:int(m['model_cuda_device_index']) for r,m in meta.items()}
    assert devices['prefill']!=devices['decode']
    P=[];D=[];counts={};foreign=[];records={};padding=padding_ms/1000
    for actor,m in meta.items():
        s,t=m['start'],m['stop'];scale=(t['host_midpoint']-s['host_midpoint'])/(t['cupti_ns']-s['cupti_ns'])
        def host(ns):return s['host_midpoint']+(int(ns)-s['cupti_ns'])*scale
        counts[actor]={};records[actor]=0
        with (folder/f'physical_activity_{actor}.csv').open() as f:
            for row in csv.DictReader(f):
                a,b=host(row['start_ns']),host(row['end_ns'])
                if a>=right or b<=left:continue
                k=row['kind'];device=int(row['device']);counts[actor][k]=counts[actor].get(k,0)+1
                records[actor]+=1
                if k=='kernel' and device==devices['prefill']:P.append((a,b))
                involved={device}
                if k not in ('kernel','graph'):
                    involved|={int(row[x]) for x in ('src_device','dst_device') if row.get(x) not in (None,'','-1','4294967295')}
                if devices['decode'] in involved:D.append((a-padding,b+padding))
                if k in ('kernel','graph') and device!=devices[actor]:foreign.append(dict(actor=actor,device=device,kind=k))
    assert counts['decode'].get('graph',0)>0,'No actual model graph record'
    assert not foreign,('Unexpected compute context',foreign[:10])
    result={}
    for role,intervals in (('prefill',P),('decode',D)):
        union,idle=gaps(intervals,left,right);busy=sum(b-a for a,b in union);ds=[1000*(b-a) for a,b in idle]
        value=dict(role=role,left=left,right=right,window_seconds=right-left,
            occupied_union_seconds=busy,occupied_union_percent=100*busy/(right-left),
            complement_seconds=right-left-busy,complement_percent=100*(1-busy/(right-left)),
            complement_windows=summary(ds),
            thresholds={str(t):dict(count=sum(d>=t for d in ds),seconds=sum(d for d in ds if d>=t)/1000) for t in (1,2,5,10,20,30,50,100)},
            exact_kernel_union=role=='prefill',interval_padding_ms=padding_ms if role=='decode' else 0,
            exact_D_kernel_busy='NOT_MEASURED',complete_D_kernel_idle_distribution='NOT_MEASURED',
            scope='Actual P kernel union.' if role=='prefill' else
                'Conservative D inactivity bound outside padded graph GPU enclosures, eager kernels, copies, and memsets from both actors. Graph enclosures include internal idle. This is not an exact D kernel union, a complete idle distribution, or a Decode-safe capacity claim. RDMA device activity can remain outside CUPTI coverage.')
        csv_write(folder/f'graph_bound_{role}_occupied.csv',(dict(start=a,end=b) for a,b in union),('start','end'))
        csv_write(folder/f'graph_bound_{role}_inactive.csv',(dict(start=a,end=b,duration_ms=1000*(b-a)) for a,b in idle),('start','end','duration_ms'))
        result[role]=value
    write(folder/'graph_envelope_bound_analysis.json',dict(roles=result,record_counts=counts,meta=meta,foreign_compute=foreign))
    return result


def analyze(plan):
    initialize();plan=Path(plan);cfg=json.loads(plan.read_text())
    clean=cfg['clean_reference'];repeat=cfg['jobs'][-1];name=cfg['jobs'][0]
    names=[clean,name,repeat];metrics={n:native(n,clean) for n in names}
    assert all(m['complete'] and m['correctness_passed'] for m in metrics.values())
    trace=physical_bound(name)
    fields=('response_wall_seconds','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
    aa=interference(metrics[repeat],metrics[clean],fields)
    screens={n:interference(metrics[name],metrics[n],fields) for n in (clean,repeat)}
    origin=json.loads((ROOT/'raw'/name/'measurement_begin.json').read_text())['time']
    l,r=trace['decode']['left']-origin,trace['decode']['right']-origin
    local={n:local_itl(n,l,r) for n in names}
    local_screens={n:interference(local[name],local[n],('p99','maximum')) if min(local[n]['count'],local[name]['count'])>=20 else dict(passed=False,reason='Fewer than 20 intervals') for n in (clean,repeat)}
    passed=aa['passed'] and all(s['passed'] for s in [*screens.values(),*local_screens.values()])
    value=dict(plan=str(plan),metrics=metrics,Native_AA=aa,whole_run_screens=screens,
        local_screens=local_screens,local_ITL=local,trace=trace,qualified=passed,
        status='QUALIFIED_CONSERVATIVE_INACTIVITY_BOUND_PILOT' if passed else 'DIAGNOSTIC_ONLY',
        Phase_1_qualified=False)
    write(ROOT/'analysis'/(plan.stem+'_results.json'),value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('metrics','trace')}))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('plan');analyze(p.parse_args().plan)
