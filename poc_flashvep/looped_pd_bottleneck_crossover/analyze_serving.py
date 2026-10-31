"""Report request latency and non-overlapping host intervals separately from kernels."""
import collections
import statistics
from .common import *
from .summary_tables import records,percentile,save_csv,wilson

def token_intervals(row):
    values=[];coalesced=0;events=row['token_events']
    if events and 'previous_time' in events[0]:
        for e in events:
            if e['first']:continue
            if e['new_tokens']==1:values.append(1000*(e['time']-e['previous_time']))
            else:coalesced+=1
    else:
        for a,b in zip(events,events[1:]):
            if b['count']-a['count']==1:values.append(1000*(b['time']-a['time']))
            else:coalesced+=1
    return values,coalesced

def analyze():
    slo_path=ROOT/'configs/frozen_SLO.json';slos=json.loads(slo_path.read_text()) if slo_path.exists() else {}
    ar_slo=ROOT/'configs/frozen_SLO_qwen.json'
    if ar_slo.exists():slos.update(json.loads(ar_slo.read_text()))
    runs=[];requests=[];shares=[];admission=[];work=[]
    configs={}
    for p in (ROOT/'configs').glob('*.json'):
        value=json.loads(p.read_text())
        if isinstance(value,dict) and value.get('name'):configs[value['name']]=value
    for folder in sorted((ROOT/'raw').iterdir()):
        cfg=configs.get(folder.name)
        if not cfg or not (folder/'requests.jsonl').exists():continue
        rows=[r for r in records(folder/'requests.jsonl') if not r.get('warmup')]
        if not rows:continue
        status=json.loads((folder/'status.json').read_text()).get('status') if (folder/'status.json').exists() else 'NO_TERMINAL_STATUS'
        window=json.loads((folder/'window.json').read_text()) if (folder/'window.json').exists() else None
        elapsed=window['end']-window['start'] if window else max(r['finish'] for r in rows)-min(r['arrival'] for r in rows)
        slo=slos.get(cfg['model']);itls=[];coalesced=0
        for r in rows:
            values,bundles=token_intervals(r);itls.extend(values);coalesced+=bundles
            r=dict(r,name=folder.name,cell=cfg.get('workload'),phase=cfg.get('phase'),ITL_single_token_ms=values,coalesced_delivery_events=bundles)
            if slo:
                r['TTFT_pass']=r['ttft_ms']<=slo['TTFT_ms'];r['TPOT_pass']=r['tpot_ms']<=slo['TPOT_ms']
                r['joint_SLO_pass']=r['TTFT_pass'] and r['TPOT_pass']
                r['all_observed_ITL_pass']=not bundles and all(v<=slo['ITL_ms'] for v in values)
            requests.append(r)
        row=dict(name=folder.name,model=cfg['model'],arm=cfg.get('arm','Native_vLLM_NIXL_PD'),cell=cfg.get('workload'),
            phase=cfg.get('phase'),status=status,offered_requests=cfg['count'],completed_requests=len(rows),
            failed_or_missing=cfg['count']-len(rows),arrival_rate=cfg.get('arrival_rate'),rate_factor=cfg.get('rate_factor'),
            elapsed_s=elapsed,completed_requests_per_s=len(rows)/elapsed,output_tokens_per_s=sum(r['output_tokens'] for r in rows)/elapsed,
            coalesced_delivery_events=coalesced,gpus=cfg['gpus'],chunk=cfg.get('chunk'),KV_blocks=cfg.get('kv_blocks'),
            compute_slots=cfg.get('capacity',32 if cfg['model']=='qwen' else None),
            observation_scope='FINITE_COHORT_ARRIVAL_TO_DRAIN_NOT_PROVEN_SUSTAINABLE_CAPACITY',independent_start=cfg.get('start',0))
        if cfg['model']=='qwen':
            for role,label in (('prefill','P'),('decode','D')):
                path=folder/f'{role}_ready.json'
                if path.exists():
                    ready=json.loads(path.read_text());row[label+'_KV_blocks']=ready['KV_blocks']
                    row[label+'_KV_tensor_bytes']=ready['KV_tensor_bytes']
                    row['KV_blocks']=ready['KV_blocks'];row['KV_dtype']=ready['KV_dtype']
        if window:
            duration=window['last_arrival']-window['start']
            row['realized_arrival_rate']=(cfg['count']-1)/duration if duration>0 else None
            row['drain_after_last_arrival_s']=window['end']-window['last_arrival']
        samples=[r for path in folder.glob('NVML-*.jsonl') for r in records(path)]
        for role,gpu in zip(('P','D'),cfg['gpus']):
            active=[r for r in samples if r['physical_GPU']==gpu and window and window['start']<=r['time']<=window['last_arrival']]
            if active:
                row[role+'_arrival_window_NVML_utilization_mean_percent']=statistics.mean(r['sampled_utilization_percent'] for r in active)
                row[role+'_arrival_window_peak_sampled_HBM_bytes']=max(r['used_HBM_bytes'] for r in active)
            row['GPU_utilization_scope']='NVML averaging-window samples, not CUPTI kernel busy or SM occupancy. DP roles are replica indices.'
        row['measurement_mode']='HEAVY_CORRECTNESS' if cfg.get('state_probe') or cfg.get('verify_transfer') else 'HEAVY_PROFILE' if cfg.get('profile') else 'HOST_COUNTERS_AND_NVML_NO_CUPTI'
        row['eligible_for_serving_comparison']=status=='COMPLETE' and len(rows)==cfg['count'] and window is not None and \
            row['measurement_mode']=='HOST_COUNTERS_AND_NVML_NO_CUPTI' and not cfg.get('prepare_D_only') and cfg.get('phase')!='calibration'
        row['eligible_for_serving_comparison'] &= cfg.get('phase') in ('crossover','focused_crossover','critical_repeat','primary','repeat','protocol_control','static_tuning')
        for field,label in (('ttft_ms','TTFT_ms'),('tpot_ms','request_TPOT_ms'),('e2e_ms','E2E_ms')):
            v=[r[field] for r in rows];row[label+'_mean']=statistics.mean(v)
            for q in (.5,.95,.99):row[label+'_p'+str(int(100*q))]=percentile(v,q)
        for q in (.5,.95,.99):row['single_token_delivery_ITL_ms_p'+str(int(100*q))]=percentile(itls,q)
        row['single_token_delivery_ITL_ms_max']=max(itls,default=None)
        if slo:
            passed=sum(r['ttft_ms']<=slo['TTFT_ms'] and r['tpot_ms']<=slo['TPOT_ms'] for r in rows)
            row.update(joint_SLO_passed=passed,joint_SLO_attainment=passed/cfg['count'],SLO_goodput_requests_per_s=passed/elapsed,
                TTFT_SLO_ms=slo['TTFT_ms'],TPOT_SLO_ms=slo['TPOT_ms'],secondary_ITL_threshold_ms=slo['ITL_ms'])
            bounds=wilson(passed,cfg['count'])
            row.update(joint_SLO_Wilson95_low=bounds[0],joint_SLO_Wilson95_high=bounds[1],
                Wilson_scope='Descriptive binomial interval. Queued requests are dependent. Use independent-start comparisons for method inference.')
        else:row['SLO_goodput_requests_per_s']='NOT_SCORED_NO_FROZEN_MODEL_SLO'
        if not row['eligible_for_serving_comparison']:
            row['SLO_goodput_requests_per_s']='NOT_ELIGIBLE_CLEAN_SERVING_RUN'
        stats=json.loads((folder/'closed_stats.json').read_text()) if (folder/'closed_stats.json').exists() else {}
        metrics={r['request_id']:r for r in stats.get('metrics',[]) if not r['request_id'].startswith('r-1_')}
        gpu_first={};core_first={}
        for path in folder.glob('host_execution_*.json'):
            host=json.loads(path.read_text())
            for event in host['frontiers']:
                if event['event']=='P_FIRST_SUBMISSION':gpu_first[event['request_id']]=event['time']
                if event['event']=='P_FIRST_RECURRENT_SUBMISSION':core_first[event['request_id']]=event['time']
            work.append(dict(name=folder.name,model=cfg['model'],role=host['role'],**host['CPU'],**host['counts'],
                scope='Process lifetime includes warm-up. CPU spans are nested, not additive.'))
        for r in rows:
            rid=f'r{r.get("run",0)}_q{r["sample_id"]}';m=metrics.get(rid)
            if not m:continue
            created=m['created_monotonic'];t=m['timings'];end=r['finish'];last=r['arrival'];den=end-last
            first_submit=gpu_first.get(m['transfer_id'])
            boundary=[('harness_delivery',r['admission']),('KV_or_slot_admission_wait',created+t.get('admitted',0)),
                ('reserve_and_P_receive',created+t.get('prefill_started',t.get('admitted',0)))]
            if first_submit is not None:boundary.append(('actual_P_queue_to_first_submission',first_submit))
            if m['transfer_id'] in core_first:boundary.append(('P_prelude_and_pre_core_queue_observed',core_first[m['transfer_id']]))
            boundary.extend([('P_service_observed_not_kernel',created+t.get('prefill_compute_completed',t.get('prefill_submitted',0))),
                ('exposed_handoff_observed',created+t.get('d_received',t.get('prefill_compute_completed',0))),
                ('D_activation_observed',created+t.get('d_activated',t.get('d_received',0))),
                ('first_D_sample_or_continuation',r['first_token']),('generation_observed',end)])
            for label,value in boundary:
                boundary_time=max(last,min(end,value));duration=boundary_time-last;last=boundary_time
                shares.append(dict(name=folder.name,model=cfg['model'],arm=cfg.get('arm'),cell=cfg.get('workload'),
                    run=r.get('run',0),sample_id=r['sample_id'],component=label,component_ms=duration*1000,request_E2E_ms=den*1000,
                    share_percent=100*duration/den,evidence='MEASURED_HOST_EXCLUSIVE_INTERVALS_NOT_GPU_KERNEL_OR_DMA_SHARES'))
        observations=records(folder/'admission_reasons.jsonl');counts=collections.Counter();samples=0
        for obs in observations:
            if window and not window['start']<=obs['time']<=window['end']:continue
            samples+=1
            reasons={reason for w in obs['waiting'] for reason in w['reasons']}
            counts.update(reasons)
        for reason,count in counts.items():admission.append(dict(name=folder.name,model=cfg['model'],reason=reason,observations=count,
            total_observations=samples,fraction=count/samples,scope='Sampled reason occupancy. Not a time-exact E2E decomposition.'))
        runs.append(row)
    save_csv(ROOT/'serving_runs.csv',runs);save_csv(ROOT/'e2e_component_shares.csv',shares)
    grouped={}
    for r in shares:
        key=r['name'],r['component'];g=grouped.setdefault(key,dict(name=r['name'],model=r['model'],arm=r['arm'],cell=r['cell'],component=r['component'],
            component_ms=0.,summed_request_E2E_ms=0.,requests=0,evidence='HOST_REQUEST_TIME_SHARES_NOT_COHORT_CRITICAL_PATH'))
        g['component_ms']+=r['component_ms'];g['summed_request_E2E_ms']+=r['request_E2E_ms'];g['requests']+=1
    for g in grouped.values():g['share_percent']=100*g['component_ms']/g['summed_request_E2E_ms']
    save_csv(ROOT/'e2e_component_shares_aggregate.csv',list(grouped.values()))
    save_csv(ROOT/'admission_reason_summary.csv',admission);save_csv(ROOT/'host_execution_summary.csv',work)
    write(ROOT/'per_request_analysis.json',requests)
    return runs

if __name__=='__main__':print('SERVING_RUNS',len(analyze()))
