"""Keep measured idle lower bounds separate from predicted safe budgets."""
import bisect
import csv
import statistics
import random
from .common import *
from ..d_local_cross_depth_batching.analyze import pct,table,requests
from ..d_local_cross_depth_batching.profile_analysis import union


def jsonlines(path):
    return [json.loads(v) for v in path.read_text().splitlines()] if path.exists() else []


def coverage_check(capture,activity):
    if not capture:return dict(passed=False,reason='MISSING_CAPTURE')
    owners=capture.get('native_graph_owners',[])
    expected=sum(v['replays'] for v in owners)
    graphs=sum(int(v['kind'])==51 for v in activity)
    activated=capture.get('activation')
    context_valid=bool(activated and activated['model_device']==f'cuda:{activated["current_device"]}')
    counters=capture.get('collector_counters',{})
    all_buffers=(counters.get('requests')==counters.get('completions')) if counters else None
    passed=bool(activity and context_valid and owners and graphs>=expected and
        capture['status']==capture['dropped']==capture['invalid']==0 and all_buffers is not False)
    return dict(passed=passed,records=len(activity),graph_records=graphs,expected_graph_replays=expected,
        context_valid=context_valid,all_requested_buffers_returned=all_buffers,
        scope='Conservative native graph-counter and buffer-return coverage screen. It does not establish negligible interference.')


def latency(rows,origin,left,right):
    itls=[];tpots=[]
    for row in rows:
        if row['first_token']-origin<=right and row['finish']-origin>=left:tpots.append(row['tpot_ms'])
        for event in row['token_events']:
            if (not event['first'] and event['new_tokens']==1
                    and event['previous_time']-origin<=right and event['time']-origin>=left):
                itls.append(1000*(event['time']-event['previous_time']))
    return dict(TPOT_p95_ms=pct(tpots,95),ITL_p50_ms=pct(itls,50),ITL_p95_ms=pct(itls,95),ITL_p99_ms=pct(itls,99),
        max_ITL_ms=max(itls,default=None),ITL_samples=len(itls))


def matched(candidate,reference):
    folder,base=ROOT/'raw'/candidate,ROOT/'raw'/reference
    cfg=json.loads((ROOT/'configs'/(candidate+'.json')).read_text())
    rows,control=requests(folder),requests(base)
    a,b=json.loads((folder/'window.json').read_text()),json.loads((base/'window.json').read_text())
    span=b['last_arrival']-b['start'];central=.35*span
    windows=[(0.,b['end']-b['start']),(.1*span,.4*span),(.4*span,.7*span),(.7*span,b['end']-b['start']),
        (max(0,central-.5),central+cfg.get('idle_trace_seconds',8.)+.5)]
    records=[];ratios=[]
    for left,right in windows:
        x,y=latency(rows,a['start'],left,right),latency(control,b['start'],left,right)
        values={key:x[key]/y[key] for key in ('TPOT_p95_ms','ITL_p95_ms','ITL_p99_ms','max_ITL_ms')
            if x[key] is not None and y[key] is not None and y[key]>0}
        records.append(dict(left_s=left,right_s=right,observed=x,reference=y,ratios=values));ratios.extend(values.values())
    indexed=lambda data:{r['sample_id']:(r['input_ids'],r['output_ids'],r['exit_depths'],r['finish_reason']) for r in data}
    exact=len(rows)==cfg['count'] and indexed(rows)==indexed(control)
    duration=(a['end']-a['start'])/(b['end']-b['start'])
    coverage=[]
    for file in folder.glob('idle_observer_*.json'):
        value=json.loads(file.read_text());capture=value['capture']
        if cfg['idle_observer']=='graph':
            activity=list(csv.DictReader((folder/capture['file']).open())) if capture else []
            coverage.append(coverage_check(capture,activity))
    complete_windows=all(len(v['ratios'])==4 for v in records)
    qualified=bool(exact and complete_windows and ratios and min(ratios)>=.95 and max(ratios)<=1.05
        and .95<=duration<=1.05 and (all(v['passed'] for v in coverage) and len(coverage)==len(cfg.get('idle_observer_roles',('prefill','decode')))
            if cfg['idle_observer']=='graph' else True))
    value=dict(name=candidate,reference=reference,model=cfg['model'],observer=cfg['idle_observer'],qualified=qualified,
        output_exact=exact,minimum_ratio=min(ratios,default=None),maximum_ratio=max(ratios,default=None),
        cohort_duration_ratio=duration,windows=records,
        activity_coverage=coverage,
        scope='Frozen 5% absolute local-tail screen, full cohort, three fixed arrival windows, and the central capture region. Graph envelopes are not exclusive kernels.')
    write(folder/'observer_qualification.json',value)
    return value


def pressure(name):
    folder=ROOT/'raw'/name
    from ..looped_pd_headroom_recovery import queues
    queues.ROOT=ROOT
    window=json.loads((folder/'window.json').read_text());_,series=queues.analyze(name,window)
    events=[]
    for file in folder.glob('P_submission_readiness_*.jsonl'):events.extend(jsonlines(file))
    events.sort(key=lambda x:x['time']);times=[v['time'] for v in events];points=[]
    starts={}
    for path in folder.glob('host_execution_*.json'):
        value=json.loads(path.read_text())
        if value['role']=='prefill':
            starts.update({v['request_id']:v['time'] for v in value.get('frontiers',[]) if v['event']=='P_FIRST_SUBMISSION'})
    spans=[]
    for row in json.loads((folder/'closed_stats.json').read_text())['metrics']:
        end=row['timings'].get('prefill_compute_completed')
        if row['request_id'].startswith('r0_') and row['transfer_id'] in starts and end is not None:
            spans.append((starts[row['transfer_id']],row['created_monotonic']+end))
    for row in series:
        index=bisect.bisect_right(times,row['time'])-1
        event=events[index] if index>=0 and row['time']-events[index]['time']<=.15 else None
        pending=bool(event and any(not t['ready'] and 'PREFILL' in t['submission_phase'] for t in event['tickets']))
        admitted=row['admitted_owner_queue_not_inflight_or_connector_blocked']
        host_service=any(left<=row['time']<=right for left,right in spans)
        count=admitted+row['unadmitted_P_slot_wait_D_credit_clear'] if admitted is not None and (pending or host_service) else admitted
        points.append(dict(time=row['time'],P_backlog_lower_bound=count,pending_prefill_event=pending,
            native_P_host_service_enclosure=host_service,
            D_credit_blocked=row['coordinator_D_credit_blocked'],
            scope='D-credit-clear P-slot pressure during a pending event or measured native host service enclosure, plus admitted owner queue. Not physical GPU saturation.'))
    table(folder/'P_pressure_timeline.csv',points)
    return points


def distribution(data):
    times=[v['duration_ms'] for v in data]
    result=dict(count=len(times),**{f'p{q}_ms':pct(times,q) for q in (50,90,95,99)},maximum_ms=max(times,default=None))
    result.update({f'fraction_at_least_{ms}ms':sum(t>=ms for t in times)/len(times) if times else None for ms in (1,2,4,8,16,32)})
    return result


def nonempty_core_upper_envelopes(intervals):
    """Bound possible gaps using one unknown kernel position per core enclosure."""
    chain=[]
    for left,right in sorted(intervals,key=lambda x:(x[1],x[0])):
        if right<=left:raise ValueError('Invalid core enclosure')
        if not chain or left>=chain[-1][1]:chain.append((left,right))
    envelopes=[(chain[0][0],chain[0][1])] if chain else []
    envelopes.extend((a[0],b[1]) for a,b in zip(chain,chain[1:]))
    if chain:envelopes.append(chain[-1])
    return chain,envelopes


def complements(name,intervals,lower,upper,qualified,evidence):
    occupied=union(intervals);gaps=[];cursor=lower
    points=pressure(name);ptimes=[v['time'] for v in points]
    def add(left,right):
        if right<=left:return
        center=(left+right)/2;index=bisect.bisect_right(ptimes,center)-1
        p=points[index] if index>=0 and center-points[index]['time']<=1. else None
        gaps.append(dict(start=left,end=right,duration_ms=1000*(right-left),
            P_backlog_lower_bound=p['P_backlog_lower_bound'] if p else None,
            same_run_P_backlog=bool(p and (p['P_backlog_lower_bound'] or 0)>0),
            qualification=qualified,evidence=evidence,
            scope='Conservative absence of observed model compute. DMA can remain active. Future Decode safety is not established.'))
    for left,right in occupied:add(cursor,left);cursor=max(cursor,right)
    add(cursor,upper)
    return gaps


def model_enclosure_gaps(name,qualification):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    window=json.loads((folder/'window.json').read_text())
    value=json.loads(next(folder.glob('idle_observer_decode_*.json')).read_text());capture=value['capture']
    left,right=capture['anchor_host'],capture['end_anchor_host']
    duration=capture['CUDA_anchor_interval_ms']
    def host(ms):return left['midpoint']+ms/duration*(right['midpoint']-left['midpoint'])
    lower=window['start']+.35*(window['last_arrival']-window['start']);upper=lower+cfg.get('idle_trace_seconds',8.)
    operations=value['operations']
    assert capture['submission_samples']==capture['all_captured_operations']==len(operations)
    intervals=[(max(lower,host(v['begin_device_ms'])),min(upper,host(v['end_device_ms'])))
        for v in operations if host(v['end_device_ms'])>lower and host(v['begin_device_ms'])<upper]
    assert intervals,'Missing complete model submission enclosures'
    # The two boundary submissions can cross the capture edge. Exclude their
    # uncovered edges. Do not turn a trace-entry offset into a long idle gap.
    lower=max(lower,min(a for a,b in intervals));upper=min(upper,max(b for a,b in intervals))
    gaps=complements(name,intervals,lower,upper,qualification['qualified'],
        'MEASURED_ALL_MODEL_SUBMISSION_ENCLOSURE_COMPLEMENT_LOWER_BOUND')
    table(folder/'model_compute_free_lower_bound_gaps.csv',gaps)
    core=[(host(v['begin_device_ms']),host(v['end_device_ms'])) for v in operations
        if v['operation']=='recurrent' and v['loops']>0 and v['rows']>0
        and host(v['begin_device_ms'])>=lower and host(v['end_device_ms'])<=upper]
    chain,envelopes=nonempty_core_upper_envelopes(core)
    anchor_rows=[dict(start=a,end=b,duration_ms=1000*(b-a),qualification=qualification['qualified'],
        evidence='STRUCTURAL_UPPER_ENVELOPE_FROM_MEASURED_NONEMPTY_CORE_ENCLOSURES',
        scope='Each selected disjoint recurrent enclosure contains native GPU computation. Allow its kernel anywhere in the enclosure. Adjacent enclosure endpoints bound possible idle. These overlapping envelopes are not observed idle opportunities.') for a,b in envelopes]
    table(folder/'core_presence_idle_upper_envelopes.csv',anchor_rows)
    result=dict(name=name,model=cfg['model'],qualified_instrumentation=qualification['qualified'],
        observation_start=lower,observation_end=upper,captured_model_submissions=len(operations),
        all_recorded_complement_gaps=distribution(gaps),
        same_run_P_backlog_gaps=distribution([g for g in gaps if g['same_run_P_backlog']]),
        nonempty_core_presence_upper_envelopes=distribution(anchor_rows),
        nonempty_core_chain_count=len(chain),
        upper_envelope_covered_start=chain[0][0] if chain else None,
        upper_envelope_covered_end=chain[-1][1] if chain else None,
        core_presence_contract='Every nonempty native recurrent submission executes GPU core kernels. The bound needs presence, not an exclusive kernel trace. It applies only inside the retained chain endpoints. Unknown work is not treated as absent.',
        physical_exclusive_kernel_idle='NOT_MEASURED',
        safe_deadline_opportunity_distribution='NOT_MEASURED_NO_ACTUAL_INSERTION_OR_VALIDATED_PREDICTOR',
        scope='CUDA-event model-submission enclosures include CPU submission and stream waits. Their complement is a conservative no-model-compute lower bound. Not an exclusive kernel union or an online-safe budget.')
    write(folder/'model_gap_summary.json',result)
    return result


def activity_gaps(name,qualification):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    window=json.loads((folder/'window.json').read_text());mapping=json.loads((folder/'worker_map.json').read_text())
    device=next(v['visible_device'] for v in mapping.values() if v['role']=='decode')
    marker=next(folder.glob('idle_observer_decode_*.json'));value=json.loads(marker.read_text());capture=value['capture']
    assert capture is not None
    start,end=capture['start'],capture['stop']
    def host(stamp):
        weight=(stamp-start['device_clock_ns'])/(end['device_clock_ns']-start['device_clock_ns'])
        return start['monotonic_s']+weight*(end['monotonic_s']-start['monotonic_s'])
    duration=cfg.get('idle_trace_seconds',8.);lower=window['start']+.35*(window['last_arrival']-window['start']);upper=lower+duration
    activity=list(csv.DictReader((folder/capture['file']).open()))
    intervals=[(max(lower,host(int(v['start_ns']))),min(upper,host(int(v['end_ns'])))) for v in activity
        if int(v['device'])==device and host(int(v['end_ns']))>lower and host(int(v['start_ns']))<upper]
    coverage=coverage_check(capture,activity)
    if not intervals or not coverage['passed']:
        import shutil
        stale=folder/'compute_free_lower_bound_gaps.csv'
        history=folder/'invalid_missing_coverage_gap_table_v1.csv'
        if stale.exists() and not history.exists():shutil.copy2(stale,history)
        table(stale,[])
        result=dict(name=name,model=cfg['model'],qualified_instrumentation=False,
            status='INVALID_ACTIVITY_COVERAGE',records=len(activity),coverage=coverage,
            all_recorded_complement_gaps='NOT_MEASURED',same_run_P_backlog_gaps='NOT_MEASURED',
            scope='An empty or partial trace is not an idle GPU. Do not construct idle gaps from incomplete coverage.')
        write(folder/'idle_gap_summary.json',result)
        return result
    occupied=union(intervals);gaps=[];cursor=lower
    points=pressure(name);ptimes=[v['time'] for v in points]
    def add(left,right):
        if right<=left:return
        center=(left+right)/2;index=bisect.bisect_right(ptimes,center)-1
        p=points[index] if index>=0 and center-points[index]['time']<=1. else None
        gaps.append(dict(start=left,end=right,duration_ms=1000*(right-left),
            P_backlog_lower_bound=p['P_backlog_lower_bound'] if p else None,
            same_run_P_backlog=bool(p and (p['P_backlog_lower_bound'] or 0)>0),
            qualification=qualification['qualified'],
            evidence='MEASURED_GRAPH_ENVELOPE_COMPLEMENT_LOWER_BOUND',
            scope='No graph envelope or observed non-graph kernel. DMA can remain active. Future decode safety is not established.'))
    for left,right in occupied:add(cursor,left);cursor=max(cursor,right)
    add(cursor,upper)
    table(folder/'compute_free_lower_bound_gaps.csv',gaps)
    eager_intervals=[(max(lower,host(int(v['start_ns']))),min(upper,host(int(v['end_ns'])))) for v in activity
        if int(v['device'])==device and int(v['kind'])==10 and int(v['node'])==0
        and host(int(v['end_ns']))>lower and host(int(v['start_ns']))<upper]
    # Ignore every graph-body kernel. The remaining known eager kernels still
    # split any true compute-free interval. Their complement is an optimistic
    # upper envelope, not measured idle. This avoids false absence claims from
    # graph-envelope complements alone.
    upper_gaps=complements(name,eager_intervals,lower,upper,qualification['qualified'],
        'OPTIMISTIC_IDLE_UPPER_ENVELOPE_ALL_GRAPH_COMPUTE_REMOVED') if eager_intervals else []
    table(folder/'compute_free_upper_envelope_gaps.csv',upper_gaps)
    selected=[v for v in gaps if v['same_run_P_backlog']]
    def summary(data):
        return distribution(data)
    result=dict(name=name,model=cfg['model'],qualified_instrumentation=qualification['qualified'],
        observation_start=lower,observation_end=upper,all_recorded_complement_gaps=summary(gaps),
        same_run_P_backlog_gaps=summary(selected),
        optimistic_idle_upper_envelope_all_graph_work_removed=distribution(upper_gaps),
        optimistic_idle_upper_envelope_with_P_pressure=distribution([v for v in upper_gaps if v['same_run_P_backlog']]),
        upper_envelope_scope='Every true physical kernel-idle interval lies inside an eager-kernel complement if eager coverage is complete. All graph compute is deliberately ignored. Durations are optimistic bounds, not observed idle or validated future slack.',
        safe_deadline_opportunity_distribution='NOT_MEASURED_NO_ACTUAL_INSERTION_OR_VALIDATED_PREDICTOR',
        P_compute_saturation='NOT_MEASURED_EXCLUSIVE_KERNEL_UNION_UNAVAILABLE',
        scope='Physical no-compute lower-bound intervals only. They are not exclusive idle completeness, SM slack, bandwidth headroom, or safe online budgets.')
    write(folder/'idle_gap_summary.json',result)
    return result


def main():
    runs=[];qualifications=[];gaps=[]
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())
    for file in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        name=file.parent.name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());receipt=json.loads(file.read_text())
        data=requests(file.parent)
        if not receipt.get('complete') or not data:continue
        window=json.loads((file.parent/'window.json').read_text());key=f'{cfg["model"]}_L{cfg["length"]}';slo=slos[key]
        row=dict(name=name,model=cfg['model'],observer=cfg.get('idle_observer','off'),load=cfg.get('load'),
            offered=cfg['count'],completed=len(data),nominal_QPS=cfg.get('arrival_rate'),
            finite_completed_QPS=len(data)/(window['end']-window['start']))
        row['evidence_class']=('COLLECTOR_PLUMBING_OR_MISROUTED_NOT_SERVING' if name.startswith('collector_') else
            'STATE_QUALIFICATION_NOT_CLEAN' if cfg.get('state_probe') or cfg.get('verify_transfer') else
            'SERIAL_CUDA_EVENT_CALIBRATION_NOT_CLEAN' if cfg.get('calibration_serial') else
            'INSTRUMENTED_CONTROL_SEPARATE_QUALIFICATION_REQUIRED' if cfg.get('idle_observer','off')!='off' else
            'MARKER_FREE_NATIVE_HOST_OBSERVED_CONTROL')
        row['state_probe']=bool(cfg.get('state_probe') or cfg.get('verify_transfer'))
        offers=jsonlines(file.parent/'offered_requests.jsonl')
        scheduled=[r['arrival'] for r in offers];actual=[r['actual_send_start'] for r in offers]
        row.update(scheduled_trace_QPS=(len(scheduled)-1)/(max(scheduled)-min(scheduled))
            if len(scheduled)>1 and max(scheduled)>min(scheduled) else None,
            realized_QPS=(len(actual)-1)/(max(actual)-min(actual))
            if len(actual)>1 and max(actual)>min(actual) else None,
            arrival_send_lag_p95_ms=pct([1000*(r['actual_send_start']-r['arrival']) for r in offers],95),
            output_tokens_per_s=sum(r['output_tokens'] for r in data)/(window['end']-window['start']),
            input_tokens_per_s=sum(r['input_tokens'] for r in data)/(window['end']-window['start']),
            total_tokens_per_s=sum(r['input_tokens']+r['output_tokens'] for r in data)/(window['end']-window['start']))
        for field,label in (('ttft_ms','TTFT'),('tpot_ms','TPOT'),('e2e_ms','E2E')):
            for q in (50,95,99):row[f'{label}_p{q}_ms']=pct([v[field] for v in data],q)
        row.update(latency(data,window['start'],0,window['end']-window['start']))
        row['Decode_screen']=row['TPOT_p95_ms']<=.9*slo['TPOT_ms'] and row['ITL_p99_ms']<=slo['ITL_ms']
        token_intervals=[1000*(e['time']-e['previous_time']) for r in data for e in r['token_events']
            if not e['first'] and e['new_tokens']==1]
        row.update(ITL_intervals_above_frozen_threshold=sum(v>slo['ITL_ms'] for v in token_intervals),
            ITL_interval_count=len(token_intervals),TTFT_SLO_pass_count=sum(r['ttft_ms']<=slo['TTFT_ms'] for r in data),
            TPOT_SLO_pass_count=sum(r['tpot_ms']<=slo['TPOT_ms'] for r in data),SLO_denominator=cfg['count'],
            service_TTFT='NOT_MEASURED_WITHOUT_QUALIFIED_FIRST_MODEL_ENTRY_MARKER',
            interval_threshold_scope='Counts above the frozen ITL p99 threshold. The p99 contract is not a hard per-token deadline guarantee.')
        row['joint_goodput']=sum(r['ttft_ms']<=slo['TTFT_ms'] and r['tpot_ms']<=slo['TPOT_ms'] for r in data)/(window['end']-window['start'])
        runs.append(row)
        if cfg.get('idle_observer','off')!='off' and name.startswith('a_'):
            reference=name.replace('_'+cfg['idle_observer']+'_','_off_')
            if not (ROOT/'raw'/reference/'dispatcher_result.json').exists():continue
            q=matched(name,reference);qualifications.append(q)
            if cfg['idle_observer']=='graph' and list(file.parent.glob('idle_observer_decode_*.json')):
                gaps.append(activity_gaps(name,q))
            if cfg['idle_observer']=='submit_events' and list(file.parent.glob('idle_observer_decode_*.json')):
                gaps.append(model_enclosure_gaps(name,q))
    table(ROOT/'serving_runs.csv',runs)
    table(ROOT/'observer_qualification.csv',[{k:v for k,v in q.items() if k!='windows'} for q in qualifications])
    write(ROOT/'idle_gap_distributions.json',gaps)
    intervals=[]
    for model in ('ouro','huginn'):
        selected=[q for q in qualifications if q['name'] in
            {f'a_{model}_heavy_submit_events_submit_s{i}_n128' for i in (1,2,3)}]
        for window_index,window_name in ((0,'WHOLE_COHORT'),(4,'BOUNDED_CAPTURE_REGION')):
            for metric in ('ITL_p99_ms','max_ITL_ms'):
                values=[q['windows'][window_index]['ratios'][metric] for q in selected]
                record=dict(model=model,window=window_name,metric=metric,independent_starts=len(values),
                    point_ratios=values,mean_ratio=statistics.mean(values) if values else None,
                    CI95='NOT_MEASURED_FEWER_THAN_THREE_STARTS',
                    scope='Percentile bootstrap of start-level ratios. No token pooling. Three starts provide limited uncertainty evidence. This interval never overrides the frozen pointwise observer gate.')
                if len(values)>=3:
                    rng=random.Random(2026110601)
                    bootstrap=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(2000)]
                    record['CI95']=[pct(bootstrap,2.5),pct(bootstrap,97.5)]
                intervals.append(record)
    write(ROOT/'observer_start_intervals.json',intervals)
    table(ROOT/'observer_start_intervals.csv',intervals)
    print(json.dumps(dict(serving=len(runs),observers=len(qualifications),
        qualified=[v['name'] for v in qualifications if v['qualified']],
        gap_summaries=len(gaps),
        scope='Full records remain in per-run JSON and root CSV files.')))


if __name__=='__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('idle_activity_and_local_tail_analysis'):main()
