"""Keep request-level SLO accounting separate from percentile safety gates."""
import math,collections
from .common import *


def records(path):return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def completed_D_layers(row,cfg):
    if 'completed_D_prefill_layers' in row:return int(row['completed_D_prefill_layers'])
    # Fixed static uses the qualified fixed executor, not dynamic request fields.
    # An unused Native configuration cut must not create fictitious migration.
    return 36-cfg['layer_cut'] if cfg.get('layer_mode')=='MIGRATE' else 0


def percentile(values,q):
    values=sorted(values)
    if not values:return None
    position=(len(values)-1)*q;a=int(position);b=math.ceil(position)
    return values[a]+(values[b]-values[a])*(position-a)


def compare(reference,candidate,prefix=False):
    old={r['sample_id']:r for r in records(ROOT/'raw'/reference/'requests.jsonl')}
    new={r['sample_id']:r for r in records(ROOT/'raw'/candidate/'requests.jsonl')};failures=[]
    for index,row in new.items():
        a=old.get(index)
        if a is None or any(a[k]!=row[k] for k in ('input_ids','seed')):
            failures.append(dict(sample_id=index,reason='INCOMPARABLE_INPUT_OR_SEED'));continue
        expected=a['output_ids'][:len(row['output_ids'])] if prefix else a['output_ids']
        if expected!=row['output_ids']:failures.append(dict(sample_id=index,reason='EXACT_OUTPUT_MISMATCH'))
    return dict(reference=reference,candidate=candidate,compared=len(new),failures=failures,
        passed=bool(new) and not failures and (prefix or len(old)==len(new)),evidence='CORRECTNESS_ONLY',
        scope='Exact generated tokens at matched inputs and seeds. Separate state and actual-layer checks are required for migration.')


def metrics(name):
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):return dict(name=name,complete=False)
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());rows=records(folder/'requests.jsonl')
    assert len(rows)==cfg['count']
    interval_rows=[];first=[];steady=[];coalesced=0
    for row in rows:
        values=[]
        for j,(a,b) in enumerate(zip(row['token_events'],row['token_events'][1:])):
            assert b['count']>a['count'];jump=b['count']-a['count'];coalesced+=jump!=1
            value=1000*(b['time']-a['time'])/jump;values.append(value)
            (first if j==0 else steady).append(value)
        interval_rows.append(values)
    all_itl=[value for values in interval_rows for value in values]
    span=max(r['finish'] for r in rows)-min(r['arrival'] for r in rows)
    arrival_span=max(r['arrival'] for r in rows)-min(r['arrival'] for r in rows)
    offered=records(folder/'offered_requests.jsonl')
    observed=[r['coordinator_observed_arrival_time'] for r in offered if 'coordinator_observed_arrival_time' in r]
    realized=(len(observed)-1)/(max(observed)-min(observed)) if len(observed)==len(rows) and max(observed)>min(observed) else None
    out=dict(name=name,complete=True,arm=cfg['arm'],output_cap=cfg['cap'],count=len(rows),
        distinct_prompt_ids=len({r['prompt_id'] for r in rows}),offered_QPS=cfg.get('arrival_rate'),
        scheduled_trace_QPS=(len(rows)-1)/arrival_span if arrival_span else None,
        realized_arrival_QPS=realized,
        completed_QPS=len(rows)/span,input_tokens_s=sum(r['input_tokens'] for r in rows)/span,
        output_tokens_s=sum(r['output_tokens'] for r in rows)/span,observation_seconds=span,
        maximum_ITL_ms=max(all_itl,default=None),ITL_intervals=len(all_itl),coalesced_intervals=coalesced,
        Service_TTFT='NOT_MEASURED',physical_GPU_idle='NOT_MEASURED',
        scope='Coordinator delivery timestamps. First ITL is included. Finite-cohort drain rates are not sustainable capacity.')
    for label,key in (('TTFT','ttft_ms'),('TPOT','tpot_ms'),('E2E','e2e_ms')):
        for q in (.5,.95,.99):out[f'{label}_p{int(100*q)}_ms']=percentile([r[key] for r in rows],q)
    for label,values in (('ITL',all_itl),('first_ITL',first),('steady_ITL',steady)):
        for q in (.5,.95,.99):out[f'{label}_p{int(100*q)}_ms']=percentile(values,q)
    for q in (.5,.95,.99):out[f'request_max_ITL_p{int(100*q)}_ms']=percentile([max(v,default=0) for v in interval_rows],q)
    worker_intervals=[];paired_intervals=[];delivery_lags=[]
    for row in rows:
        events=row['token_events']
        delivery_lags.extend(1000*(e['time']-e['worker_time']) for e in events if 'worker_time' in e)
        for a,b in zip(events,events[1:]):
            if 'worker_time' not in a or 'worker_time' not in b:continue
            jump=b['count']-a['count'];worker=1000*(b['worker_time']-a['worker_time'])/jump
            parent=1000*(b['time']-a['time'])/jump
            worker_intervals.append(worker);paired_intervals.append((parent,worker,parent-worker))
    out['worker_emission_ITL_p99_ms']=percentile(worker_intervals,.99)
    out['coordinator_delivery_lag_p95_ms']=percentile(delivery_lags,.95)
    selected=[v for v in paired_intervals if v[0]>=out['ITL_p99_ms']]
    out['parent_tail_interval_identity']=dict(samples=len(selected),
        parent_interval_mean_ms=sum(v[0] for v in selected)/len(selected) if selected else None,
        worker_interval_mean_ms=sum(v[1] for v in selected)/len(selected) if selected else None,
        receiver_lag_change_mean_ms=sum(v[2] for v in selected)/len(selected) if selected else None,
        scope='Select parent intervals at or above its p99. Parent interval equals worker interval plus the change in receive lag. The worker interval still includes GPU and Host scheduling. This is not an additive decomposition of p99 or between-method regression.')
    slo_file=ROOT/'configs/frozen_SLO_regimes.json'
    regimes={}
    if slo_file.exists():
        thresholds=json.loads(slo_file.read_text())['output_caps'].get(str(cfg['cap']),{})
        for label,slo in thresholds.items():
            ttft=[r['ttft_ms']<=slo['TTFT_ms'] for r in rows]
            decode=[r['tpot_ms']<=slo['TPOT_ms'] and max(values,default=0)<=slo['ITL_ms'] for r,values in zip(rows,interval_rows)]
            joint=[a and b for a,b in zip(ttft,decode)]
            regimes[label]=dict(thresholds=slo,TTFT_attainment=sum(ttft)/len(rows),
                Decode_attainment=sum(decode)/len(rows),Joint_attainment=sum(joint)/len(rows),
                Request_SLO_goodput=sum(joint)/span,
                Output_token_SLO_goodput=sum(r['output_tokens'] for r,p in zip(rows,joint) if p)/span,
                ITL_threshold_exceedances=sum(x>slo['ITL_ms'] for x in all_itl),
                percentile_Decode_absolute_pass=out['ITL_p99_ms']<=slo['ITL_ms'] and out['TPOT_p95_ms']<=slo['TPOT_ms'])
    out['SLO_regimes']=regimes
    native={}
    for path in folder.glob('native_*.json'):
        value=json.loads(path.read_text())
        if 'role' not in value or 'queue_samples' not in value:continue
        role=value['role'];samples=value['queue_samples']
        window=json.loads((folder/'window.json').read_text());samples=[r for r in samples if window['start']<=r['time']<=window['end']]
        middle=(window['start']+window['last_arrival'])/2
        halves=[[s['waiting'] for s in samples if s['time']<=middle],[s['waiting'] for s in samples if middle<s['time']<=window['last_arrival']]]
        compute_waiting=[s['waiting']-sum(n for status,n in s.get('waiting_statuses',{}).items() if 'REMOTE' in status) for s in samples]
        native[role]=dict(queue_samples=len(samples),waiting_max=max((s['waiting'] for s in samples),default=0),
            waiting_excluding_remote_KV_max=max(compute_waiting,default=0),
            waiting_half_means=[sum(v)/len(v) if v else None for v in halves],
            KV_free_min=min((s['KV_free_blocks'] for s in samples),default=None),
            KV_usage_max=max((s['KV_usage'] for s in samples),default=None),
            KV_allocation_refusals=len(value['KV_allocation_refusals']),
            scope='Native scheduler observations. WAITING_REMOTE_KVS is not compute-ready work. Detailed status counts remain in raw evidence.')
    out['native_observations']=native
    sends=records(folder/'actual_submissions.jsonl')
    actual=[r['actual_send_start'] for r in sends if 'actual_send_start' in r]
    out['actual_compute_submission_QPS']=(len(actual)-1)/(max(actual)-min(actual)) if len(actual)>1 and max(actual)>min(actual) else None
    out['compute_submission_roles']=sorted({r['role'] for r in sends if 'role' in r})
    handoffs=records(folder/'native_handoffs.jsonl')
    if handoffs:
        window=json.loads((folder/'window.json').read_text());last=max(r['time'] for r in handoffs)
        out['P_source_production_span_s']=last-window['start']
        out['P_clearance_after_last_arrival_s']=last-window['last_arrival']
        out['finite_P_source_ready_rate']=len(handoffs)/(last-window['start'])
    else:
        out['P_source_production_span_s']=None;out['P_clearance_after_last_arrival_s']=None;out['finite_P_source_ready_rate']=None
    out['P_production_scope']='Host observation start to final Source-ready notification. Includes compute, queue, preparation, and backpressure. This is not exclusive GPU compute or sustainable capacity.'
    out['arrival_observation_lag_ms']=dict(samples=len(observed),
        p95=percentile([1000*(r['coordinator_observed_arrival_time']-r['arrival']) for r in offered if 'coordinator_observed_arrival_time' in r],.95),
        scope='Coordinator arrival observation is separate from a later Source submission. Older discovery logs do not contain this boundary.')
    observations=records(folder/'queue_observations.jsonl')
    if observations:
        window=json.loads((folder/'window.json').read_text());middle=(window['start']+window['last_arrival'])/2
        halves=[[r['pending_phases'].get('P_HOST_COMPUTE_PENDING',0) for r in observations if r['time']<=middle],
            [r['pending_phases'].get('P_HOST_COMPUTE_PENDING',0) for r in observations if middle<r['time']<=window['last_arrival']]]
        out['coordinator_queue_observations']=dict(
            P_host_compute_pending_max=max(r['pending_phases'].get('P_HOST_COMPUTE_PENDING',0) for r in observations),
            P_host_compute_pending_half_means=[sum(v)/len(v) if v else None for v in halves],
            D_host_pending_max=max(r['pending_phases'].get('D_HOST_PENDING',0) for r in observations),
            scope='Dynamic holds its compute queue in the coordinator. Native holds its queue in the worker. Neither coordinator outstanding nor D_HOST_PENDING is P compute-ready work.')
    admissions=records(folder/'D_host_admission.jsonl')
    out['D_cohort_request_admissions_by_batch_size']=dict(collections.Counter(r['cohort_size'] for r in admissions))
    if admissions:out['D_cohort_count']=len({r['cohort'] for r in admissions})
    write(folder/'metrics.json',out);return out
