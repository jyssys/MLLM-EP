"""Do not convert instrumented kernel inactivity into verified safe slack."""
import csv,math,statistics
from collections import Counter
from .common import *
from .analyze import latency,summary,requests,csv_write


def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out


def idle_windows(intervals,left,right):
    union=merge((max(left,a),min(right,b)) for a,b in intervals if b>left and a<right)
    windows=[];cursor=left
    for a,b in union:
        if a>cursor:windows.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<right:windows.append((cursor,right))
    return union,windows


def captured(name,role,left=None,right=None):
    folder=ROOT/'raw'/name;meta=json.loads((folder/f'idle_kernels_{role}.json').read_text())
    a,b=meta['start'],meta['stop']
    def host(ns):return a['host_midpoint']+(int(ns)-a['cupti_ns'])/(b['cupti_ns']-a['cupti_ns'])*(b['host_midpoint']-a['host_midpoint'])
    left=max(left or a['host_midpoint'],a['host_midpoint']);right=min(right or b['host_midpoint'],b['host_midpoint'])
    intervals=[]
    with (folder/f'idle_kernels_{role}.csv').open() as f:
        for row in csv.DictReader(f):intervals.append((host(row['start_ns']),host(row['end_ns'])))
    union,windows=idle_windows(intervals,left,right)
    duration=[1000*(b-a) for a,b in windows];busy=sum(b-a for a,b in union);idle=sum(duration)/1000
    result=dict(role=role,window=[left,right],window_seconds=right-left,kernel_records=len(intervals),
        capture_status=meta['status'],dropped=meta['dropped'],invalid=meta['invalid'],
        kernel_union_seconds=busy,kernel_busy_percent=100*busy/(right-left),
        kernel_idle_seconds=idle,kernel_idle_percent=100*idle/(right-left),
        idle_windows=summary(duration),idle_window_count=len(windows),
        idle_p90_ms=sorted(duration)[min(len(duration)-1,int(.9*(len(duration)-1)))] if duration else None,
        thresholds={str(t):dict(count=sum(v>=t for v in duration),seconds=sum(v for v in duration if v>=t)/1000)
            for t in (1,2,5,10,20,30)},
        scope='Actual kernel interval union in an instrumented five-second window. Not copy-engine inactivity, Decode safety, or a measured whole-cohort idle total.')
    csv_write(ROOT/'analysis'/(name+'_'+role+'_idle_windows.csv'),[dict(start=a,end=b,duration_ms=1000*(b-a)) for a,b in windows])
    csv_write(ROOT/'analysis'/(name+'_'+role+'_kernel_union.csv'),[dict(start=a,end=b) for a,b in union])
    return result,windows


def host_samples(name,left,right):
    folder=ROOT/'raw'/name
    def rows(filename):
        p=folder/filename
        return [r for r in map(json.loads,p.read_text().splitlines()) if left<=r['time']<=right] if p.exists() else []
    admissions=rows('admission_reasons.jsonl');queues=rows('queue_observations.jsonl')
    waiting=[len(r['waiting']) for r in admissions]
    dcredits=[v for r in queues for v in r['worker_credits'].values() if v['role']=='decode']
    hist=Counter()
    for path in folder.glob('atomic_spans_*.json'):
        v=json.loads(path.read_text())
        if v['role']=='decode':
            for r in v['time_histogram']:
                if left<=r['time'] and r['time']+.5<=right and r['operation']=='recurrent' and r['logical_rows']==r['requests']:
                    hist[r['requests']]+=r['submissions']
    return dict(coordinator_waiting_request_samples=summary(waiting),
        D_reserved_blocks=summary(r['reserved_blocks'] for r in dcredits),D_reserved_slots=summary(r['slots'] for r in dcredits),
        actual_D_decode_batch_submissions=dict(hist),
        P_loop_ready_frontier_overlap='NOT_MEASURED',
        scope='Coordinator waiting samples and KV reservations are not GPU-ready queue or actual source-cut completion timestamps.')


def point_screen(value,reference):
    ratios={k+'/'+s:value[k][s]/reference[k][s] for k,s in
        (('TTFT_ms','p95'),('TPOT_ms','p95'),('ITL_ms','p99'),('ITL_ms','maximum'))}
    return dict(ratios=ratios,passed=all(.95<=v<=1.05 for v in ratios.values()))


def forecast_audit(name):
    folder=ROOT/'raw'/name;paths=list(folder.glob('drain_forecast_audit_*.json'))
    if not paths:return dict(name=name,status='NOT_MEASURED')
    marker=json.loads((folder/'measurement_begin.json').read_text())['time']
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    entries=json.loads(paths[0].read_text())['records']
    finishes={r['request_id']:r['time'] for r in entries if r['event']=='OUTPUT' and r['finished']}
    result=[];missing=0
    for row in entries:
        if row['event']!='ADVERTISED_FORECAST' or row['time']<marker or not row['cohort']:continue
        cohort=row['cohort']
        if not all(0<=r['seed']-cfg['seed']<cfg['count'] and r['request_id'] in finishes for r in cohort):missing+=1;continue
        actual=1000*(max(finishes[r['request_id']] for r in cohort)-row['observed'])
        error=actual-row['prediction_ms']
        result.append(dict(observed=row['observed'],sent=row['time'],cohort_size=len(cohort),
            prediction_ms=row['prediction_ms'],actual_known_cohort_drain_ms=actual,
            error_actual_minus_prediction_ms=error,absolute_error_ms=abs(error),underpredicted=error>0))
    csv_write(ROOT/'analysis'/(name+'_forecast_errors.csv'),result)
    return dict(name=name,status='OBSERVED_KNOWN_COHORT_FORECAST',advertisements=len(result),missing=missing,
        MAE_ms=statistics.mean(r['absolute_error_ms'] for r in result) if result else None,
        absolute_error_ms=summary(r['absolute_error_ms'] for r in result),
        signed_error_ms=summary(r['error_actual_minus_prediction_ms'] for r in result),
        underprediction_fraction=statistics.mean(r['underpredicted'] for r in result) if result else None,
        actual_drain_ms=summary(r['actual_known_cohort_drain_ms'] for r in result),
        prediction_ms=summary(r['prediction_ms'] for r in result),
        scope='All measured nonempty known-cohort advertisements, not independent requests or only P-selected advertisements. Actual drain is the final D output-send receipt among those exact known IDs. Future arrivals can interfere but do not enter the target cohort. This forecast prepares checkpoints. It is not a completion guarantee.')


def main():
    initialize();rows=[]
    for cap in (16,4):
        names={mode:f'idle_output{cap}_{mode}_n64' for mode in ('clean','repeat','capture')}
        deferred=f'idle_output{cap}_deferred_n64'
        receipt=ROOT/'raw'/deferred/'dispatcher_result.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('complete'):
            names['capture']=deferred
        receipt=ROOT/'raw'/names['capture']/'dispatcher_result.json'
        complete=receipt.exists() and json.loads(receipt.read_text()).get('complete')
        metric={mode:latency(ROOT/'raw'/name) for mode,name in names.items() if mode!='capture' or complete}
        reference=OLD_ROOT/'raw'/f'pipeline7_output{cap}_zero_seed2026110940'
        original={r['seed']:r for r in requests(reference)}
        exact={mode:all(r['output_ids']==original[r['seed']]['output_ids'] and r['input_ids']==original[r['seed']]['input_ids'] for r in requests(ROOT/'raw'/name)) for mode,name in names.items() if mode in metric}
        if not complete:
            window=json.loads((ROOT/'raw'/names['clean']/'window.json').read_text())
            span=window['last_arrival']-window['start']
            rows.append(dict(output_cap=cap,names=names,metrics=metric,exact_outputs=exact,
                strict_capture_qualified=False,capture_status='NOT_MEASURED',capture=None,
                Native_AA=point_screen(metric['repeat'],metric['clean']),
                clean_host=host_samples(names['clean'],window['start']+.2*span,window['start']+.9*span),
                whole_cohort_idle_seconds='NOT_MEASURED',Decode_safe_loops='NOT_MEASURED',
                reason='CUPTI-path CUDA launch failure. Incomplete requests do not enter performance or idle statistics.'))
            continue
        captures={role:json.loads((ROOT/'raw'/names['capture']/f'idle_kernels_{role}.json').read_text()) for role in ('prefill','decode')}
        left=max(v['start']['host_midpoint'] for v in captures.values());right=min(v['stop']['host_midpoint'] for v in captures.values())
        values={};idle=None
        for role in captures:
            values[role],windows=captured(names['capture'],role,left,right)
            if role=='decode':idle=windows
        screens={mode:point_screen(metric['capture'],metric[mode]) for mode in ('clean','repeat')}
        # An explicitly relaxed packing count. No source readiness, transfer,
        # Decode reserve, request cap, or concurrency is supplied for free to a
        # feasible oracle. This result is only a diagnostic cost sensitivity.
        costs=[21.808764518737792,28.894943237304688,34.14363386535645]
        packing=[dict(one_loop_cost_ms=cost,window_count=sum(1000*(b-a)>=cost for a,b in idle),
            unbounded_ready_loop_units=sum(int(1000*(b-a)//cost) for a,b in idle)) for cost in costs]
        tile_cost=8.206883544921874
        tile_slots=sum(int(1000*(b-a)//tile_cost) for a,b in idle)
        row=dict(output_cap=cap,names=names,metrics=metric,exact_outputs=exact,
            instrumentation=screens,strict_capture_qualified=all(s['passed'] for s in screens.values()),
            Native_AA=point_screen(metric['repeat'],metric['clean']),capture=values,
            host=host_samples(names['capture'],left,right),
            offline_compute_only_cost_sensitivity=packing,
            offline_tiled_compute_only_sensitivity=dict(tokens_per_tile=256,tile_cost_ms=tile_cost,
                tile_slots=tile_slots,complete_loop_units=tile_slots//8,uncompleted_tiles=tile_slots%8,
                scope='Optimistic packing sensitivity. Cost is an instrumented Loop28 tile256 p99 operation enclosure from the prior diagnostic, not a clean kernel or independent validated primitive. Eight sequential full-prompt tiles count as one Loop. No handoff, source readiness, Coda, or Decode reserve is supplied. Not a safe or request-feasible oracle.'),
            physical_request_feasible_loops='NOT_MEASURED',Decode_safe_loops='NOT_MEASURED',
            tiled_feasible_loops='NOT_MEASURED',whole_cohort_idle_seconds='NOT_MEASURED')
        window=json.loads((ROOT/'raw'/names['clean']/'window.json').read_text())
        span=window['last_arrival']-window['start']
        row['clean_host']=host_samples(names['clean'],window['start']+.2*span,window['start']+.9*span)
        rows.append(row)
    forecasts=[forecast_audit(f'idle_drain_output{cap}_audit_n64') for cap in (16,4)]
    for row in forecasts:
        if row['status']!='OBSERVED_KNOWN_COHORT_FORECAST':continue
        name=row['name'];folder=ROOT/'raw'/name
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        reference={r['seed']:r for r in requests(OLD_ROOT/'raw'/f'pipeline7_output{cfg["cap"]}_zero_seed2026110940')}
        values=requests(folder)
        row['exact_output_prefixes']=len(values)==cfg['count'] and all(
            r['input_ids']==reference[r['seed']]['input_ids'] and r['output_ids']==reference[r['seed']]['output_ids'] for r in values)
    write(ROOT/'analysis/output_idle_diagnosis.json',dict(rows=rows,forecasts=forecasts,
        scope='Kernel capture and forecast diagnosis only. No new scheduler. A failed sensitivity screen cannot support clean idle or safe-loop claims.'))
    print(json.dumps([dict(cap=r['output_cap'],qualification=r['strict_capture_qualified'],screens=r.get('instrumentation'),D=r['capture']['decode'] if r['capture'] else None,packing=r.get('offline_compute_only_cost_sensitivity')) for r in rows]))
    print(json.dumps(forecasts))


if __name__=='__main__':main()
