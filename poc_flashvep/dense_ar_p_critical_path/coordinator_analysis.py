"""Keep worker gaps, delivery gaps and overlapping host observations separate."""
import json
import time
import numpy as np
from .common import ROOT,bind,write,digest
from .critical_analysis import records,Timeline,csv_save


def gc_intervals(rows):
    active={};spans=[]
    for row in rows:
        key=row['generation']
        if row['phase']=='start':
            assert key not in active
            active[key]=row['time']
        elif row['phase']=='stop':
            assert key in active
            spans.append((active.pop(key),row['time']))
        else:raise AssertionError(row['phase'])
    assert not active,'A closed observer must contain both GC boundaries'
    return spans


def quantiles(values):
    return dict(count=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),
        p99=float(np.percentile(values,99)),maximum=float(max(values))) if values else dict(count=0)


def delivery_rows(folder):
    output=[]
    for request in records(folder/'requests.jsonl'):
        events=request['token_events']
        for before,after in zip(events,events[1:]):
            client=1000*(after['time']-before['time'])
            worker=1000*(after['worker_time']-before['worker_time'])
            output.append(dict(name=folder.name,request_id='q'+str(request['sample_id']),
                prompt_id=request['prompt_id'],token_count=after['count'],
                client_left=before['time'],client_right=after['time'],
                worker_left=before['worker_time'],worker_right=after['worker_time'],
                client_ITL_ms=client,worker_emission_gap_ms=worker,
                delivery_delay_change_ms=client-worker,
                delivery_after_ms=1000*(after['time']-after['worker_time'])))
    return output


def analyze_case(name):
    folder=ROOT/'raw'/name;probe=json.loads((folder/'coordinator_probe.json').read_text())
    window=json.loads((folder/'window.json').read_text());left,right=window['start'],window['end']
    cfg=json.loads((folder/'configuration.json').read_text())
    append=probe['append_rows']
    spans=dict(serialization=[(r['start'],r['serialization_end']) for r in append],
        evidence_write_or_buffer=[(r['serialization_end'],r['end']) for r in append],
        GC=gc_intervals(probe['gc_rows']),
        receive=[(r['start'],r['end']) for r in probe['receive_enclosures']],
        slow_poll=[(r['start'],r['end']) for r in probe['slow_poll_enclosures']],
        async_sender=[(r['start'],r['end']) for r in probe['existing_sender_enclosures']])
    timelines={key:Timeline(value) for key,value in spans.items()}
    # These columns overlap. They cannot be summed as exclusive causes.
    loops=[]
    for row in probe['loop_gaps']:
        a,b=max(left,row['left']),min(right,row['right'])
        if b<=a:continue
        loops.append(dict(name=name,**row,measured_overlap_ms=1000*(b-a),
            **{key+'_overlap_ms':1000*value.area(a,b) for key,value in timelines.items()}))
    tails=delivery_rows(folder)
    for row in tails:
        a,b=row['client_left'],row['client_right']
        for key,timeline in timelines.items():row[key+'_overlap_ms']=1000*timeline.area(a,b)
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    result=dict(name=name,evidence='DIAGNOSTIC',source_SHA256=cfg['source_capsule_SHA256'],
        dataset_SHA256=cfg['dataset_SHA256'],GPU_pair=cfg['gpus'],seed=cfg['seed'],
        deferred=probe['deferred'],traced=probe['traced'],buffered_bytes=probe['buffered_bytes'],
        cyclic_GC_deferred=probe.get('cyclic_GC_deferred',False),
        measurement_boundaries=probe.get('measurement_boundaries',[]),
        metrics=summarize(name),
        measured_host_union_ms={key:1000*value.area(left,right) for key,value in timelines.items()},
        append_serialization_ms=quantiles([1000*(r['serialization_end']-r['start']) for r in append
            if left<=r['start']<=right]),
        append_write_or_buffer_ms=quantiles([1000*(r['end']-r['serialization_end']) for r in append
            if left<=r['start']<=right]),
        GC_ms=quantiles([1000*(b-a) for a,b in spans['GC'] if left<=a<=right]),
        loop_gaps=sorted(loops,key=lambda r:r['gap_ms'],reverse=True)[:100],
        client_ITL_ms=quantiles([r['client_ITL_ms'] for r in tails]),
        worker_emission_gap_ms=quantiles([r['worker_emission_gap_ms'] for r in tails]),
        delivery_delay_change_ms=quantiles([r['delivery_delay_change_ms'] for r in tails]),
        worst_client_intervals=sorted(tails,key=lambda r:r['client_ITL_ms'],reverse=True)[:30],
        scope='All host observations are enclosures. GC can overlap serialization,receive or writes. '
            'Async sender time is not coordinator blocking. No column proves exclusive CPU use or GPU work. '
            'Only evidence-file writes differ in the diagnostic A/B. No serving win or observer qualification follows from one start. '
            'Client ITL equals worker-emission gap plus change in delivery delay for each token interval.')
    csv_save(ROOT/'figures'/(name+'_loop_gaps.csv'),loops)
    csv_save(ROOT/'figures'/(name+'_delivery_intervals.csv'),tails)
    return result


def analyze(plan_name='coordinator_delivery_io_diagnostic_v1.json'):
    bind();plan=ROOT/'jobs'/plan_name
    result=[];status=[]
    for case in json.loads(plan.read_text())['cases']:
        for name in case['jobs']:
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists():status.append(dict(name=name,status='NOT_RUN_OR_RUNNING'));continue
            if not json.loads(receipt.read_text())['complete']:
                status.append(dict(name=name,status='FAILED'));continue
            result.append(analyze_case(name));status.append(dict(name=name,status='DIAGNOSTIC_COMPLETE'))
    value=dict(time=time.time(),plan_SHA256=digest(plan),cases=result,status=status,
        scope='The original long-tail run remains intact. These captures discriminate host causes,not primary performance.')
    stem='coordinator_delivery_diagnostic' if plan_name=='coordinator_delivery_io_diagnostic_v1.json' else 'coordinator_GC_diagnostic'
    write(ROOT/'analysis'/(stem+'.json'),value)
    csv_save(ROOT/'figures'/(stem+'_summary.csv'),[
        dict(name=r['name'],deferred=r['deferred'],client_max_ms=r['client_ITL_ms']['maximum'],
            worker_max_ms=r['worker_emission_gap_ms']['maximum'],
            longest_loop_gap=r['loop_gaps'][0] if r['loop_gaps'] else None,
            host_union_ms=r['measured_host_union_ms']) for r in result])
    print('COORDINATOR_DIAGNOSTIC_ANALYSIS',[(r['name'],r['client_ITL_ms']['maximum'],
        r['loop_gaps'][0] if r['loop_gaps'] else None) for r in result],flush=True)
    return value


def test():
    assert gc_intervals([dict(generation=0,phase='start',time=1),dict(generation=0,phase='stop',time=3)])==[(1,3)]
    timeline=Timeline([(1,3),(2,4)])
    assert timeline.area(0,5)==3
    # The exact delivery identity must not attribute host delay to the worker.
    assert abs((1.851-.0072)-1.8438)<1e-12
    print('COORDINATOR_ANALYSIS_CPU_PASS')


if __name__=='__main__':
    import sys
    test() if len(sys.argv)>1 and sys.argv[1]=='test' else analyze(sys.argv[1] if len(sys.argv)>1 else 'coordinator_delivery_io_diagnostic_v1.json')
