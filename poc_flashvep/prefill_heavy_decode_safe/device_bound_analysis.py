"""Validate coverage before using an event-enclosure inactivity lower bound."""
import csv,json
from .common import *
from .analyze import gaps,merged,summary,csv_write,native,local_itl
from .gates import interference


def events(name,padding_ms=2.):
    folder=ROOT/'raw'/name;paths=list(folder.glob('D_device_bounds_*.json'));assert len(paths)==1
    data=json.loads(paths[0].read_text());pad=padding_ms/1000
    return data,merged((r['gpu_start']-pad,r['gpu_end']+pad) for r in data['rows'])


def activity(name,role):
    folder=ROOT/'raw'/name;meta=json.loads((folder/f'physical_activity_{role}.json').read_text())
    assert meta['status']==meta['counters']['invalid']==meta['counters']['dropped']==0
    s,t=meta['start'],meta['stop'];scale=(t['host_midpoint']-s['host_midpoint'])/(t['cupti_ns']-s['cupti_ns'])
    def host(ns):return s['host_midpoint']+(int(ns)-s['cupti_ns'])*scale
    with (folder/f'physical_activity_{role}.csv').open() as f:
        kernels=[(host(r['start_ns']),host(r['end_ns'])) for r in csv.DictReader(f) if r['kind']=='kernel']
    return meta,kernels


def coverage(name,padding_ms=2.):
    data,union=events(name,padding_ms);meta,kernels=activity(name,'decode')
    left,right=meta['start']['host_midpoint'],meta['stop']['host_midpoint']
    _,outside=gaps(union,left,right);misses=[]
    for a,b in outside:
        misses.extend((max(a,c),min(b,d)) for c,d in kernels if c<b and d>a)
    missing=sum(b-a for a,b in merged(misses))
    value=dict(name=name,padding_ms=padding_ms,window_seconds=right-left,
        actual_kernel_records=len(kernels),uncovered_kernel_seconds=missing,
        passed=bool(kernels) and missing<=1e-6,
        event_clock=data,
        scope='Coverage validation in a separate diagnostic trace. Passing coverage does not validate clean latency or exact physical-idle percentages. Padding remains explicit. All kernel activity must fit the GPU-event enclosure union.')
    write(ROOT/'analysis'/('device_bound_coverage_'+name+'.json'),value);return value


def bound(name,padding_ms=2.):
    data,D=events(name,padding_ms);meta,P=activity(name,'prefill')
    left,right=meta['start']['host_midpoint'],meta['stop']['host_midpoint'];result={}
    for role,intervals in (('prefill',P),('decode',D)):
        union,idle=gaps(intervals,left,right);occupied=sum(b-a for a,b in union);ds=[1000*(b-a) for a,b in idle]
        result[role]=dict(left=left,right=right,window_seconds=right-left,
            occupied_union_percent=100*occupied/(right-left),complement_seconds=right-left-occupied,
            complement_percent=100*(1-occupied/(right-left)),complement_windows=summary(ds),
            exact_kernel_union=role=='prefill',padding_ms=padding_ms if role=='decode' else 0,
            exact_D_kernel_busy='NOT_MEASURED',complete_D_idle_distribution='NOT_MEASURED',
            scope='Actual P kernel union' if role=='prefill' else 'Conservative inactivity bound outside validated multi-stream GPU-event enclosures. Not an exact D kernel union or Decode-safe budget.')
        csv_write(ROOT/'raw'/name/f'event_bound_{role}_inactive.csv',
            (dict(start=a,end=b,duration_ms=1000*(b-a)) for a,b in idle),('start','end','duration_ms'))
    write(ROOT/'raw'/name/'event_bound_analysis.json',dict(roles=result,clock_mapping=data))
    return result


def analyze(plan):
    cfg=json.loads(Path(plan).read_text());cov=coverage(cfg['coverage_run'])
    clean=cfg['clean_reference'];capture=cfg['capture'];repeat=cfg['repeat']
    names=[clean,capture,repeat];metrics={n:native(n,clean) for n in names}
    assert all(v['complete'] and v['correctness_passed'] for v in metrics.values())
    trace=bound(capture);origin=json.loads((ROOT/'raw'/capture/'measurement_begin.json').read_text())['time']
    t=trace['prefill'];local={n:local_itl(n,t['left']-origin,t['right']-origin) for n in names}
    fields=('response_wall_seconds','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
    aa=interference(metrics[repeat],metrics[clean],fields)
    screens={n:interference(metrics[capture],metrics[n],fields) for n in (clean,repeat)}
    local_screens={n:interference(local[capture],local[n],('p99','maximum')) if min(local[capture]['count'],local[n]['count'])>=20 else dict(passed=False) for n in (clean,repeat)}
    passed=cov['passed'] and aa['passed'] and all(v['passed'] for v in [*screens.values(),*local_screens.values()])
    value=dict(plan=str(plan),qualified=passed,status='QUALIFIED_EVENT_INACTIVITY_BOUND_PILOT' if passed else 'DIAGNOSTIC_ONLY',
        Native_AA=aa,whole_run_screens=screens,local_screens=local_screens,local_ITL=local,
        trace=trace,metrics=metrics,coverage_validation=cov,exact_D_kernel_idle='NOT_MEASURED',Phase_1_qualified=False)
    write(ROOT/'analysis'/(Path(plan).stem+'_results.json'),value);print(json.dumps({k:v for k,v in value.items() if k not in ('metrics','trace','coverage_validation')}))
    return value
