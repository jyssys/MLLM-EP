"""Keep measured request, service, queue, and profile evidence separate."""
import collections
import csv
import math
import statistics
from .common import *
from .evidence import records,rows,status
from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals
from ..looped_pd_bottleneck_crossover.summary_tables import percentile,wilson


def save_csv(name,values):
    values=list(values);path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in values for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)


def distribution(values,label):
    values=list(values)
    return {label+'_'+name:(statistics.mean(values) if name=='mean' else percentile(values,q)) if values else None
        for name,q in [('mean',None),('p50',.5),('p95',.95),('p99',.99)]}


def capacity():
    values=[]
    for cfgpath in sorted((ROOT/'configs').glob('capacity_*.json')):
        cfg=json.loads(cfgpath.read_text())
        if not isinstance(cfg,dict) or not cfg.get('name') or not cfg.get('capacity_mode'):continue
        folder=ROOT/'raw'/cfg['name']
        geometry=json.loads((folder/'geometry.json').read_text()) if (folder/'geometry.json').exists() else {}
        for r in records(folder/'cohorts.jsonl'):
            job=r['job'];batch=job['batch'];cap=job['cap'];mode=cfg['capacity_mode']
            row=dict(name=cfg['name'],model=cfg['model'],length=cfg['length'],chunk=cfg['chunk'],chunk_label=cfg['chunk_label'],
                capacity_mode=mode,batch=batch,output_cap=cap,repeat=r.get('repeat'),status=r.get('status','COMPLETE'),
                capacity_preparation=r.get('capacity_preparation','NATIVE_COHORT_FULL_PREFILL'),
                run_status=status(cfg['name']),evidence='MEASURED_LOCAL_FINITE_SERVICE',
                measured_KV_blocks=geometry.get('KV_blocks'),measured_KV_bytes=geometry.get('KV_bytes',geometry.get('KV_tensor_bytes')),
                scope='Independent finite service cohort. Not measured sustainable open-loop capacity.')
            if row['status']=='COMPLETE':
                p=r.get('prefill_host_service_s',r.get('P_plus_first_sample_s'))
                d=r.get('decode_native_boundary_service_s',r.get('prepared_D_rest_s'))
                elapsed=p if mode=='P_ONLY' else d
                row.update(service_s=elapsed,finite_requests_per_s=batch/elapsed,
                    timed_output_tokens_per_s=(batch*(cap if cfg['model']!='qwen8' else cap-1)/elapsed) if mode!='P_ONLY' else None,
                    service_boundary='P excludes first Coda' if mode=='P_ONLY' and cfg['model']!='qwen8' else
                        'P includes first sample' if mode=='P_ONLY' else 'D includes first Coda' if cfg['model']!='qwen8' else 'D excludes first sample',
                    exact_repeat_output_parity=r.get('native_output_parity',r.get('repeat_output_parity')))
            else:row.update(required_blocks=r.get('required_blocks'),available_blocks=r.get('available_blocks'))
            values.append(row)
    save_csv('capacity_samples.csv',values)
    groups=collections.defaultdict(list)
    for r in values:
        if r['status']=='COMPLETE' and r['run_status']=='COMPLETE':
            groups[(r['model'],r['length'],r['chunk'],r['capacity_mode'],r['batch'],r['output_cap'],r['capacity_preparation'])].append(r)
    summary=[]
    for key,group in groups.items():
        m,n,c,mode,b,cap,preparation=key;rates=[r['finite_requests_per_s'] for r in group]
        summary.append(dict(model=m,length=n,chunk=c,capacity_mode=mode,batch=b,output_cap=cap,
            capacity_preparation=preparation,
            measured_KV_blocks=group[0].get('measured_KV_blocks'),measured_KV_bytes=group[0].get('measured_KV_bytes'),
            repeats=len(group),finite_requests_per_s_mean=statistics.mean(rates),finite_requests_per_s_p50=statistics.median(rates),
            service_s_mean=statistics.mean(r['service_s'] for r in group),service_boundary=group[0]['service_boundary'],
            GPU= json.loads((ROOT/'configs'/(group[0]['name']+'.json')).read_text())['gpus'],
            evidence='MEASURED_FINITE_INDEPENDENT_SERVICE',uncertainty='Three within-start repeats. Independent-start CI not yet available.'))
    save_csv('capacity_summary.csv',summary);write(ROOT/'capacity_summary.json',summary)
    return summary


def slope(points):
    if len(points)<2:return None
    x=statistics.mean(t for t,v in points);y=statistics.mean(v for t,v in points)
    den=sum((t-x)**2 for t,v in points)
    return sum((t-x)*(v-y) for t,v in points)/den if den else None


def host_phases(folder,request_rows):
    path=folder/'closed_stats.json'
    if not path.exists():
        handoffs={r['request_id']:r for r in records(folder/'native_handoffs.jsonl')};starts={};shares=[]
        for p in folder.glob('native_prefill_*.json'):
            for span in json.loads(p.read_text())['prompt_spans']:
                rid=span['request_id'].split('-',1)[0];starts[rid]=min(starts.get(rid,float('inf')),span['time'])
        for r in request_rows:
            rid='q'+str(r['sample_id']);handoff=handoffs.get(rid)
            if not handoff or rid not in starts:continue
            begin=r['arrival'];end=r['finish'];last=begin
            for component,bound in [('delivery',r['admission']),('P_queue_to_first_scheduler_span',starts[rid]),
                ('P_service_host_with_first_sample',handoff['time']),('handoff_D_queue_import_first_sample_host',r['first_token']),('generation_host',end)]:
                bound=max(last,min(end,bound));duration=bound-last;last=bound
                shares.append(dict(name=folder.name,sample_id=r['sample_id'],component=component,component_ms=1000*duration,
                    E2E_ms=1000*(end-begin),share_percent=100*duration/(end-begin),
                    evidence='MEASURED_HOST_EXCLUSIVE_REQUEST_INTERVALS_NOT_GPU_KERNEL_SHARES'))
        return [],shares
    metrics=[r for r in json.loads(path.read_text()).get('metrics',[]) if r['request_id'].startswith('r0_')]
    byid={r['request_id']:r for r in metrics};frontier={};shares=[]
    for p in folder.glob('host_execution_*.json'):
        for event in json.loads(p.read_text())['frontiers']:
            if event['event']=='P_FIRST_SUBMISSION':frontier[event['request_id']]=event['time']
    for r in request_rows:
        rid=f'r{r.get("run",0)}_q{r["sample_id"]}';m=byid.get(rid)
        if not m:continue
        created=m['created_monotonic'];t=m['timings'];begin=r['arrival'];last=begin;end=r['finish']
        bounds=[('delivery',r['admission']),('KV_or_slot_admission',created+t.get('admitted',0)),
            ('reserve_and_P_receive',created+t.get('prefill_started',0))]
        if m['transfer_id'] in frontier:bounds.append(('P_queue_to_first_submission',frontier[m['transfer_id']]))
        bounds.extend([('P_service_host_not_kernel',created+t.get('prefill_compute_completed',t.get('prefill_submitted',0))),
            ('exposed_handoff_host_not_DMA',created+t.get('d_received',0)),
            ('D_activation',created+t.get('d_activated',0)),('first_D_sample',r['first_token']),('generation_host',end)])
        for component,bound in bounds:
            bound=max(last,min(end,bound));duration=bound-last;last=bound
            shares.append(dict(name=folder.name,sample_id=r['sample_id'],component=component,component_ms=1000*duration,
                E2E_ms=1000*(end-begin),share_percent=100*duration/(end-begin),
                evidence='MEASURED_HOST_EXCLUSIVE_REQUEST_INTERVALS_NOT_GPU_KERNEL_SHARES'))
    return metrics,shares


def serving_one(cfg):
    name=cfg['name'];folder=ROOT/'raw'/name;requests=rows(name)
    if not requests:return None,[],[]
    path=folder/'window.json';window=json.loads(path.read_text()) if path.exists() else None
    begin=window['start'] if window else min(r['arrival'] for r in requests)
    end=window['end'] if window else max(r['finish'] for r in requests)
    last=window['last_arrival'] if window else max(r['arrival'] for r in requests)
    elapsed=end-begin;arrival_duration=last-begin
    slopath=ROOT/'configs/frozen_SLO.json';slos=json.loads(slopath.read_text()) if slopath.exists() else {}
    slo=slos.get(cfg['model']+'_L'+str(cfg['length']))
    itls=[];bundles=0
    for r in requests:
        values,coalesced=token_intervals(r);itls.extend(values);bundles+=coalesced
    row=dict(name=name,model=cfg['model'],length=cfg['length'],output_cap=cfg['cap'],arm=cfg['arm'],
        chunk=cfg['chunk'],chunk_label=cfg['chunk_label'],phase=cfg['phase'],independent_start=cfg.get('independent_start',0),
        status=status(name),offered=cfg['count'],completed=len(requests),failed_or_missing=cfg['count']-len(requests),
        target_QPS=cfg.get('arrival_rate'),realized_QPS=(cfg['count']-1)/arrival_duration if arrival_duration>0 else None,
        elapsed_s=elapsed,arrival_window_s=arrival_duration,drain_s=end-last,
        requests_per_s=len(requests)/elapsed,output_tokens_per_s=sum(r['output_tokens'] for r in requests)/elapsed,
        total_tokens_per_s=sum(r['input_tokens']+r['output_tokens'] for r in requests)/elapsed,
        coalesced_delivery_events=bundles,gpus=cfg['gpus'],ar_batch_invariant=cfg.get('ar_batch_invariant',False),
        native_transport_limits=cfg.get('native_transport_limits',{}),
        D_token_budget=cfg.get('D_token_budget',cfg['token_budget']),D_KV_blocks=cfg.get('D_kv_blocks',cfg['kv_blocks']) if cfg['model']!='qwen8' else 'NATIVE_AUTO_ALLOCATION',
        transport_control=('NOT_APPLICABLE_COLOCATED' if cfg['arm'].startswith('DP') else
            'INFLIGHT_'+str(cfg.get('native_transport_limits',{}).get('max_inflight_bytes',256*1024**2)) if cfg['model']!='qwen8' else 'NATIVE_AR_NIXL'),
        timing_scope='Finite arrival-to-drain observation. Not established sustainable capacity.',
        eligible_clean=status(name)=='COMPLETE' and len(requests)==cfg['count'] and window is not None and cfg['phase'] in ('qps_pilot','final_repeat','paper_budget_control') and not any(cfg.get(k) for k in ('state_probe','verify_transfer','nsys')),
        actual_P_kernel_busy='NOT_MEASURED',actual_D_kernel_busy='NOT_MEASURED')
    if cfg.get('D_token_budget') or cfg.get('D_kv_blocks'):
        row['transport_control']+='_DB'+str(cfg.get('D_token_budget',cfg['token_budget']))+'_KV'+str(cfg.get('D_kv_blocks',cfg['kv_blocks']))
    if cfg['arm'].startswith('DP'):
        row['transport_control']+='_B'+str(cfg['token_budget'])+'_KV'+str(cfg['kv_blocks'])+'_W'+str(int(cfg.get('wavefront',False)))
    row['timing_isolation_clean']=row['eligible_clean']
    row['GPU_role_mapping']=dict(P_column='replica0',D_column='replica1') if cfg['arm'].startswith('DP') else dict(P_column='prefill',D_column='decode')
    from .live_audit import blocked
    row['fidelity_gate_blocked']=bool(blocked(cfg['model'],cfg));row['eligible_clean'] &= not row['fidelity_gate_blocked']
    for field,label in [('ttft_ms','TTFT_ms'),('tpot_ms','request_TPOT_ms'),('e2e_ms','E2E_ms')]:
        row.update(distribution((r[field] for r in requests),label))
    row.update(distribution(itls,'token_ITL_ms'));row['token_ITL_ms_max']=max(itls,default=None)
    offers=records(folder/'offered_requests.jsonl')
    if offers and all('actual_send_start' in r for r in offers):
        first_sent=min(r['actual_send_start'] for r in offers);last_sent=max(r['actual_send_start'] for r in offers)
        row['realized_scheduled_QPS']=row['realized_QPS']
        row['realized_QPS']=(len(offers)-1)/(last_sent-first_sent) if last_sent>first_sent else None
        row.update(distribution((1000*(r['actual_send_start']-r['arrival']) for r in offers),'load_generator_lag_ms'))
    else:row['realized_QPS_scope']='Scheduled Poisson span only. Actual send timestamps were not recorded in this diagnostic.'
    if slo:
        passed=sum(r['ttft_ms']<=slo['TTFT_ms'] and r['tpot_ms']<=slo['TPOT_ms'] for r in requests)
        low,high=wilson(passed,cfg['count'])
        row.update(SLO_passed=passed,SLO_attainment=passed/cfg['count'],SLO_Wilson95_low=low,SLO_Wilson95_high=high,
            SLO_goodput=passed/elapsed,TTFT_threshold_ms=slo['TTFT_ms'],TPOT_threshold_ms=slo['TPOT_ms'],ITL_threshold_ms=slo['ITL_ms'],
            D_TPOT_margin_pass=row['request_TPOT_ms_p95']<=.9*slo['TPOT_ms'])
    else:row.update(SLO_goodput='NOT_SCORED_NO_FROZEN_SLO',D_TPOT_margin_pass=False)
    samples=[r for p in folder.glob('NVML-*.jsonl') for r in records(p) if begin<=r['time']<=end]
    for role,gpu in zip(('P','D'),cfg['gpus']):
        values=[r for r in samples if r['physical_GPU']==gpu]
        arrival_values=[r for r in values if r['time']<=last]
        if values:row.update({role+'_NVML_util_mean':statistics.mean(r['sampled_utilization_percent'] for r in arrival_values) if arrival_values else None,
            role+'_sampled_peak_HBM_bytes':max(r['used_HBM_bytes'] for r in values)})
    row['NVML_scope']='Utilization mean covers the arrival window. Sampled peak HBM covers arrival through drain. Not SM occupancy or exclusive kernel busy.'
    metrics,shares=host_phases(folder,requests)
    left=begin+.2*arrival_duration;right=begin+.9*arrival_duration
    central=[r for r in records(folder/'admission_reasons.jsonl') if left<=r['time']<=right]
    for p in folder.glob('atomic_spans_*.json'):
        d=json.loads(p.read_text())
        if d['role']!='decode':continue
        batches=[r for r in d.get('time_histogram',[]) if left<=r['time']<=right and r['operation']=='recurrent']
        if batches:
            total=sum(r['submissions'] for r in batches)
            row['D_recurrent_batch_rows_mean']=sum(r['logical_rows']*r['submissions'] for r in batches)/total
            row['D_recurrent_batch_requests_mean']=sum(r['requests']*r['submissions'] for r in batches)/total
            row['D_recurrent_submission_count']=total
    if central:
        p_reasons=('PREFILL_NO_SLOT','PREFILL_NO_KV_CREDIT')
        row['D_raw_KV_refusal_sample_fraction']=sum(any('DECODE_NO_KV_CREDIT' in w['reasons'] for w in r['waiting']) for r in central)/len(central)
        row['D_raw_slot_refusal_sample_fraction']=sum(any('DECODE_NO_SLOT' in w['reasons'] for w in r['waiting']) for r in central)/len(central)
        row['D_KV_refusal_sample_fraction']=sum(any('DECODE_NO_KV_CREDIT' in w['reasons'] and not any(k in w['reasons'] for k in p_reasons) for w in r['waiting']) for r in central)/len(central)
        row['D_slot_refusal_sample_fraction']=sum(any('DECODE_NO_SLOT' in w['reasons'] and not any(k in w['reasons'] for k in p_reasons) for w in r['waiting']) for r in central)/len(central)
        row['D_refusal_scope']='Binding refusal: D blocks while P has a slot and KV credit. Raw simultaneous P/D refusals are reported separately.'
    if metrics:
        firsts={}
        for p in folder.glob('host_execution_*.json'):
            host=json.loads(p.read_text())
            if host['role']=='prefill':
                firsts.update({r['request_id']:r['time'] for r in host['frontiers'] if r['event']=='P_FIRST_SUBMISSION'})
            if host['role']=='decode':
                points=[]
                for s in host['queue_samples']:
                    if not left<=s['time']<=right:continue
                    count=sum(v for k,v in s['queued'].items() if any(word in k.lower() for word in ('recurrent','coda','prelude')))
                    points.append((s['time'],count))
                if points:row['D_ready_queue_slope_per_s']=slope(points)
        intervals=[];finished=[]
        for m in metrics:
            t=m['timings'];created=m['created_monotonic'];entry=created+t.get('prefill_started',t.get('admitted',0))
            first=firsts.get(m['transfer_id']);done=created+t['prefill_compute_completed'] if 'prefill_compute_completed' in t else None
            if first is not None:intervals.append((entry,first))
            if done is not None:finished.append(done)
        points=[(left+(right-left)*i/100,sum(a<=left+(right-left)*i/100<b for a,b in intervals)) for i in range(101)] if right>left else []
        # P-slot waiting is part of the P backlog only while D credit is clear.
        # Never relabel D-reservation waiting as P compute waiting.
        for index,(timepoint,value) in enumerate(points):
            snapshots=[r for r in central if r['time']<=timepoint]
            if snapshots:
                observed=snapshots[-1]
                count=sum('PREFILL_NO_SLOT' in w['reasons'] and 'PREFILL_NO_KV_CREDIT' not in w['reasons'] for w in observed['waiting'])
                points[index]=(timepoint,value+count)
        row.update(P_queue_slope_per_s=slope(points),P_queue_half_growth=(statistics.mean(v for t,v in points[51:])-statistics.mean(v for t,v in points[:50])) if points else None,
            P_queue_max=max((v for t,v in points),default=None),P_completion_rate_central=sum(left<=t<=right for t in finished)/(right-left) if right>left else None)
        row['P_queue_scope']='Admitted P wait plus P-slot backlog. Simultaneous D refusals remain explicit. D-only reservation waiting is excluded. P kernel saturation and matched P service are mandatory.'
        a=folder/'before_measured.json';b=folder/'after_measured.json'
        if a.exists() and b.exists():
            before=json.loads(a.read_text());after=json.loads(b.read_text())
            for field,label in [('transport_bytes_sent','KV_sent_bytes'),('transport_completed_count','KV_fragment_completions'),('transport_completion_observation_seconds','host_cumulative_transfer_observation_s')]:
                row[label]=sum(s[field]-before[k][field] for k,s in after.items())
    else:
        # Native AR exposes actual scheduler waiting and running samples.
        for p in folder.glob('native_prefill_*.json'):
            points=[(r['time'],r['waiting']) for r in json.loads(p.read_text())['queue_samples'] if left<=r['time']<=right]
            if len(points)>1:
                mid=len(points)//2
                row.update(P_queue_slope_per_s=slope(points),P_queue_half_growth=statistics.mean(v for t,v in points[mid:])-statistics.mean(v for t,v in points[:mid]),P_queue_max=max(v for t,v in points))
        for p in folder.glob('native_decode_*.json'):
            points=[r for r in json.loads(p.read_text())['queue_samples'] if left<=r['time']<=right]
            if points:
                row.update(D_scheduler_waiting_max=max(r['waiting'] for r in points),D_KV_usage_max=max(r['KV_usage'] for r in points),
                    D_running_mean=statistics.mean(r['running'] for r in points))
                data=json.loads(p.read_text())
                if 'KV_allocation_refusals' in data:
                    refusals=[r for r in data['KV_allocation_refusals'] if left<=r['time']<=right]
                    row['D_KV_allocation_refusals']=len(refusals)
                    row['D_KV_refusal_sample_fraction']=sum(any(t['time']<=r['time']<t['time']+.25 for r in refusals) for t in points)/len(points)
                else:row['D_KV_refusal_sample_fraction']='NOT_MEASURED_NATIVE_NO_REFUSAL_CALLBACK'
                if all('waiting_statuses' in r for r in points):
                    row['D_slot_refusal_sample_fraction']=sum(r['running']>=r['max_running_requests'] and any('REMOTE' not in k and v>0 for k,v in r['waiting_statuses'].items()) for r in points)/len(points)
                    ready=[(r['time'],sum(v for k,v in r['waiting_statuses'].items() if 'REMOTE' not in k)) for r in points]
                    row['D_ready_queue_slope_per_s']=slope(ready)
        handoffs=records(folder/'native_handoffs.jsonl')
        row['P_completion_rate_central']=sum(left<=r['time']<=right for r in handoffs)/(right-left) if right>left else None
        for p in folder.glob('native_decode_*.json'):
            d=json.loads(p.read_text())
            if not d.get('NIXL_measured_boundary_flushed'):continue
            measurements=[r['data'] for r in d['NIXL_stats_intervals']]
            for field,label in [('bytes_transferred','KV_received_bytes'),('num_descriptors','NIXL_descriptors'),
                ('transfer_duration','NIXL_telemetry_cumulative_transfer_s'),('post_duration','NIXL_telemetry_cumulative_post_s')]:
                row[label]=sum(sum(r.get(field,[])) for r in measurements)
            row['NIXL_telemetry_scope']='Native completion telemetry after an explicit warm-up flush. Cumulative overlapping transfer time is not exclusive E2E or pure GPU DMA.'
    return row,requests,shares


def serving():
    runs=[];share_rows=[]
    for p in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(p.read_text())
        if not isinstance(cfg,dict) or not cfg.get('name') or cfg.get('capacity_mode'):continue
        row,requests,shares=serving_one(cfg)
        if row:runs.append(row);share_rows.extend(shares)
    save_csv('serving_runs.csv',runs);write(ROOT/'serving_runs.json',runs);save_csv('e2e_component_shares.csv',share_rows)
    groups=collections.defaultdict(list)
    for r in share_rows:groups[(r['name'],r['component'])].append(r)
    save_csv('e2e_component_shares_summary.csv',[dict(name=k[0],component=k[1],requests=len(v),
        component_ms_mean=statistics.mean(r['component_ms'] for r in v),
        mean_request_share_percent=statistics.mean(r['share_percent'] for r in v),
        duration_weighted_share_percent=100*sum(r['component_ms'] for r in v)/sum(r['E2E_ms'] for r in v),
        evidence='MEASURED_HOST_EXCLUSIVE_INTERVALS_ONLY',scope='Both share columns sum to 100% within a run. Neither column is an exclusive GPU/DMA component breakdown.') for k,v in groups.items()])
    return runs


if __name__=='__main__':print('CAPACITY',len(capacity()),'SERVING',len(serving()))
