"""Measured request statistics. No simulated speedups or capacity interpolation."""
import csv
import json
import math
from .common import *
from .calibration import percentile

def read(path):
    if not path.exists():return []
    text=path.read_text();values=[];items=text.splitlines()
    for i,line in enumerate(items):
        if not line:continue
        try:values.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(items)-1 and not text.endswith('\n'):break
            raise
    return values

def csv_write(path,rows):
    rows=list(rows)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)

def wilson(k,n):
    if not n:return (None,None)
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-delta),min(1.,center+delta)

def metric(cfg,directory):
    rows=[r for r in read(directory/'requests.jsonl') if not r['warmup']]
    result=dict(name=cfg['name'],model=cfg['model'],arm=cfg['arm'],workload=cfg.get('workload'),
        phase=cfg.get('phase','gate'),start=cfg.get('start',0),expected_N=cfg['count'],completed_N=len(rows),
        rate=cfg.get('rate','NOT_MEASURED'),GPU_ids=str(cfg['gpus']),
        protocol='native_chunk_streaming' if cfg['arm']=='PD_G1' else 'snapshot' if cfg['arm'] not in ('DP','DP_Chunked') else 'none')
    status=directory/'status.json'
    result['status']=json.loads(status.read_text())['status'] if status.exists() else 'NOT_RUN'
    arrivals=cfg.get('arrival_offsets',[])
    if len(arrivals)>1 and arrivals[-1]>arrivals[0]:
        result['scheduled_arrival_span_s']=arrivals[-1]-arrivals[0]
        result['realized_scheduled_arrival_rate_per_s']=(len(arrivals)-1)/(arrivals[-1]-arrivals[0])
    result['offered_rate_scope']='NORMAL_POISSON_LAMBDA. W3 has an explicit5x long-prompt burst.' if cfg.get('phase') in ('primary','repeat') else 'DIAGNOSTIC_OR_CALIBRATION_SCHEDULE'
    result['burst_rate_multiplier']=5 if cfg.get('workload')=='W3' and cfg.get('phase') in ('primary','repeat') else 1
    for key in ('ttft_ms','tpot_ms','e2e_ms'):
        values=[r[key] for r in rows]
        if not values:continue
        result[key+'_mean']=sum(values)/len(values)
        for p in (.5,.95,.99):result[key+f'_p{round(p*100)}']=percentile(values,p)
    itls=[1000*(e['time']-e['previous_time']) for r in rows for e in r['token_events'] if not e['first'] and e['new_tokens']==1]
    result['ITL_observed_single_token_intervals']=len(itls)
    result['multi_token_delivery_events']=sum(e['new_tokens']>1 for r in rows for e in r['token_events'])
    if itls:
        for p in (.5,.95,.99):result[f'ITL_ms_p{round(p*100)}']=percentile(itls,p)
        result['ITL_ms_max']=max(itls)
    window=directory/'window.json'
    if window.exists():
        t=json.loads(window.read_text());duration=t['end']-t['start']
        result.update(cohort_makespan_s=duration,drain_after_last_arrival_s=t['end']-t['last_arrival'],
            requests_per_s=len(rows)/duration,output_tokens_per_s=sum(r['output_tokens'] for r in rows)/duration)
    if 'ttft_slo_ms' in cfg:
        passed=sum(r['ttft_ms']<=cfg['ttft_slo_ms'] and r['tpot_ms']<=cfg['tpot_slo_ms'] for r in rows)
        result.update(SLO_passed_N=passed,SLO_denominator=cfg['count'],SLO_attainment=passed/cfg['count'],
            SLO_Wilson95_low=wilson(passed,cfg['count'])[0],SLO_Wilson95_high=wilson(passed,cfg['count'])[1],
            failed_or_missing_N=cfg['count']-len(rows),ttft_slo_ms=cfg['ttft_slo_ms'],tpot_slo_ms=cfg['tpot_slo_ms'])
        if window.exists():result['SLO_goodput_per_s']=passed/duration
    if rows:
        result.update(input_tokens_mean=sum(r['input_tokens'] for r in rows)/len(rows),
            output_tokens_mean=sum(r['output_tokens'] for r in rows)/len(rows),
            cap_stops=sum(r['finish_reason']=='length' for r in rows),
            distinct_prompt_ids=len({r['prompt_id'] for r in rows}))
    a=directory/'before_measured.json';b=directory/'after_measured.json'
    if a.exists() and b.exists():
        before=json.loads(a.read_text());after=json.loads(b.read_text())
        for field in ('transport_bytes_sent','transport_completed_count','issued_loop_rows','issued_submissions','transport_completion_observation_seconds'):
            result[field]=sum(after[k].get(field,0)-before[k].get(field,0) for k in before)
        result['required_loop_rows']=sum((r['input_tokens']+r['output_tokens']-1)*cfg['depth'] for r in rows)
        result['full_depth_work_exact']=result['issued_loop_rows']==result['required_loop_rows']
        for i,k in enumerate(before):
            for field in ('weight_bytes','reserved_kv_bytes','peak_allocated_bytes'):
                result[f'worker{i}_{field}']=after[k].get(field,'NOT_MEASURED')
    decisions=read(directory/'online_decisions.jsonl')
    planned=[r for r in decisions if r['event']=='INITIAL_PLAN' and r['request_id'].startswith('r0_')]
    result['initial_loop_cuts']=json.dumps({str(k):sum(r['cut']==k for r in planned) for k in sorted({r['cut'] for r in planned})})
    result['frontier_accepts']=sum(r['event']=='FRONTIER_RECHECK' and r['accept'] for r in decisions)
    result['frontier_refusals']=sum(r['event']=='FRONTIER_RECHECK' and not r['accept'] for r in decisions)
    result['stability_status']='NOT_ESTABLISHED_FINITE_COHORT'
    result['GPU_kernel_time']='NOT_MEASURED_CLEAN'
    result['pure_DMA_time']='NOT_MEASURED'
    for field in ('exclusive_prefill_queue_ms','exclusive_decode_queue_ms','online_selector_overhead_ms',
                  'clean_cuda_kernel_count','clean_recurrent_gpu_ms','clean_attention_gpu_ms',
                  'clean_ffn_gpu_ms','exact_GPU_idle_fraction'):
        result[field]='NOT_MEASURED'
    for path in directory.glob('NVML-summary-*.json'):
        summary=json.loads(path.read_text())
        for gpu,value in summary.get('measured_cohort_peak_sampled_HBM_bytes',{}).items():
            result[f'physical_GPU{gpu}_sampled_peak_HBM_bytes']=value
    sampled=[r for path in directory.glob('NVML-*.jsonl') for r in read(path) if r.get('measured_cohort')]
    for gpu in cfg['gpus']:
        values=[r['sampled_utilization_percent'] for r in sampled if r['physical_GPU']==gpu]
        if values:result[f'physical_GPU{gpu}_mean_NVML_utilization_percent']=sum(values)/len(values)
    if cfg.get('phase') in ('primary','repeat','protocol_control'):
        if (directory/'live_output_mismatch.json').exists():result['status']='BLOCKED_CORRECTNESS'
        parity=ROOT/'state'/f'primary_parity_{cfg["name"]}.json'
        if cfg['arm']!='PD_G1' and result['status']=='COMPLETE':
            if not parity.exists():result['status']='PENDING_OUTPUT_PARITY'
            elif not json.loads(parity.read_text())['passed']:result['status']='BLOCKED_CORRECTNESS'
        if result.get('full_depth_work_exact') is False:result['status']='BLOCKED_CORRECTNESS_WORK_ACCOUNTING'
    return result,rows

def main():
    setup();all_metrics=[];requests=[];matrix=[];online=[];shares=[];communication=[];cut_outcomes=[];policy_reasons=[]
    skipped={r['name']:r['reason'] for r in read(ROOT/'state/skipped.jsonl')+read(ROOT/'state/primary_skipped.jsonl')+read(ROOT/'state/secondary_skipped.jsonl')}
    all_cells=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if 'arm' not in cfg:continue
        directory=ROOT/'raw'/cfg['name']
        status=directory/'status.json'
        actual=[r for r in read(directory/'requests.jsonl') if not r['warmup']]
        all_cells.append(dict(name=cfg['name'],model=cfg['model'],arm=cfg['arm'],
            phase=cfg.get('phase','static_tuning' if cfg.get('tune_jobs') else 'calibration' if cfg.get('calibration') else 'qualification'),
            planned_measured_requests=sum(j['count'] for j in cfg['tune_jobs']) if cfg.get('tune_jobs') else cfg['count'],
            measured_requests=len(actual),tuning_cohorts=len(cfg.get('tune_jobs',[])),gpus=str(cfg['gpus']),
            status=json.loads(status.read_text())['status'] if status.exists() else skipped.get(cfg['name'],'NOT_RUN')))
        if cfg.get('tune_jobs'):continue
        if not directory.exists():continue
        result,rows=metric(cfg,directory);all_metrics.append(result)
        if cfg.get('phase') in ('primary','repeat','protocol_control'):
            requests.extend(dict(r,phase=cfg['phase'],start=cfg.get('start',0),rate=cfg.get('rate')) for r in rows)
            online.extend(dict(name=cfg['name'],phase=cfg['phase'],**r) for r in read(directory/'online_decisions.jsonl'))
            closed=directory/'closed_stats.json'
            intervals={}
            if closed.exists():
                for m in json.loads(closed.read_text()).get('metrics',[]):intervals[m['request_id']]=m
            for r in rows:
                m=intervals.get(f'r{r["run"]}_q{r["sample_id"]}')
                end=r['finish']
                if not m:
                    boundaries=[('harness_delivery',r['admission']),
                        ('colocated_queue_and_prefill_unseparated',r['first_token']),('generation_observed',end)]
                else:
                    t=m['timings'];created=m['created_monotonic']
                    boundaries=[('harness_delivery',created),
                    ('reservation_queue',created+t.get('admitted',0)),
                    ('reservation_and_P_notification',created+t.get('prefill_started',t.get('admitted',0))),
                    ('P_queue_and_compute_observed',created+t.get('prefill_compute_completed',t.get('prefill_started',0))),
                    ('exposed_handoff_observed',created+t.get('d_received',t.get('prefill_compute_completed',0))),
                    ('D_admission_observed',created+t.get('d_activated',t.get('d_received',0))),
                    ('D_continuation_and_first_token',r['first_token']),('generation_observed',end)]
                last=r['arrival'];den=end-last
                for label,boundary in boundaries:
                    boundary=max(last,min(end,boundary));duration=boundary-last;last=boundary
                    shares.append(dict(name=cfg['name'],model=cfg['model'],arm=cfg['arm'],workload=cfg['workload'],
                        sample_id=r['sample_id'],component=label,component_ms=1000*duration,
                        E2E_ms=1000*den,share_percent=100*duration/den,
                        evidence='MEASURED_HOST_EXCLUSIVE_INTERVAL_ACCOUNTING_NOT_GPU_KERNEL_SHARES'))
            window=directory/'window.json'
            if window.exists():
                t=json.loads(window.read_text())
                plans={r['request_id']:r for r in read(directory/'online_decisions.jsonl') if r['event']=='INITIAL_PLAN'}
                refused={r['seed'] for p in directory.glob('fallback_*.jsonl') for r in read(p) if t['start']<=r['time']<=t['end']}
                for r in rows:
                    plan=plans.get(f'r{r["run"]}_q{r["sample_id"]}')
                    if plan is None:continue
                    if cfg.get('dynamic'):
                        from .policy import token_candidates
                        state=plan['state'];n=r['input_tokens'];depth=cfg['depth']
                        full_cost=cfg['prefill_ms_per_token']*n/1000.
                        slack=state['tpot_slo_s']-state['max_decode_age_s']
                        allowed=max(0.,slack)*cfg.get('slack_fraction',.5)
                        fractions=([(depth-c)/depth for c in cfg['cut_candidates'] if 0<c<depth]
                            if cfg['axis']=='loop' else [(n-k)/n for k in token_candidates(n,cfg['chunk'],depth,cfg['cut_candidates']) if 0<k<n])
                        minimum=min(fractions,default=0.)
                        selected_fraction=((depth-plan['cut'])/depth if cfg['axis']=='loop' else (n-plan['prefix_k'])/n)
                        reason=('NO_P_BACKLOG' if state['p_backlog']<1 else
                            'D_IDLE_HALF_WORK_FLOOR' if state['active_decode']==0 and selected_fraction>0 else
                            'NON_FULL_BOUNDARY_SELECTED' if selected_fraction>0 else
                            'D_AGE_SLACK_EXHAUSTED' if slack<=0 else
                            'MINIMUM_LEGAL_CONTINUATION_EXCEEDS_CONSERVATIVE_ALLOWANCE' if minimum*full_cost>allowed else
                            'OTHER_LEGAL_ALIGNMENT_CONSTRAINT')
                        policy_reasons.append(dict(name=cfg['name'],model=cfg['model'],workload=cfg['workload'],arm=cfg['arm'],
                            sample_id=r['sample_id'],reason=reason,observed_P_backlog=state['p_backlog'],
                            observed_active_Decode=state['active_decode'],predicted_full_prefill_ms=1000*full_cost,
                            minimum_nonzero_D_fraction=minimum,minimum_predicted_continuation_ms=1000*minimum*full_cost,
                            allowed_continuation_ms=1000*allowed,initial_D_fraction=selected_fraction,
                            evidence='DETERMINISTIC_POLICY_RECONSTRUCTION_FROM_RECORDED_STATE. Cost is a host-calibration model, not measured GPU time.'))
                    fallback=r['seed'] in refused
                    cut=cfg['depth'] if fallback else plan['cut'];prefix=r['input_tokens'] if fallback else plan['prefix_k']
                    cut_outcomes.append(dict(name=cfg['name'],model=cfg['model'],workload=cfg['workload'],arm=cfg['arm'],
                        sample_id=r['sample_id'],input_tokens=r['input_tokens'],initial_cut=plan['cut'],actual_cut=cut,
                        initial_prefix_k=plan['prefix_k'],actual_prefix_k=prefix,full_PD_fallback=fallback,
                        P_work_fraction=cut/cfg['depth'] if cfg['axis']=='loop' else prefix/r['input_tokens'],
                        actual_D_continuation_fraction=(cfg['depth']-cut)/cfg['depth'] if cfg['axis']=='loop' else (r['input_tokens']-prefix)/r['input_tokens'],
                        scope='INITIAL_COORDINATOR_PLAN_PLUS_MEASURED_SOURCE_FALLBACK_AND_COMPLETED_REQUEST'))
                payload=[r for p in directory.glob('migration_*.jsonl') for r in read(p) if t['start']<=r['time']<=t['end']]
                kv=sum(r['KV_bytes'] for r in payload);hidden=sum(r['hidden_bytes'] for r in payload)
                total=result.get('transport_bytes_sent',0)
                if cfg['arm']=='PD_G1':
                    hidden=len(rows)*(2048 if cfg['model']=='ouro' else 10560)*2;kv=total-hidden
                    scope='MEASURED_TOTAL_PLUS_STRUCTURAL_SINGLE_CARRIER_SPLIT'
                elif cfg['arm'] in ('DP','DP_Chunked'):scope='NO_REQUEST_STATE_TRANSFER_BY_DESIGN'
                else:scope='MEASURED_DESCRIPTOR_PAYLOAD_ACCOUNTING'
                communication.append(dict(name=cfg['name'],model=cfg['model'],arm=cfg['arm'],workload=cfg['workload'],
                    KV_bytes=kv,hidden_and_initial_state_bytes=hidden,measured_total_payload_bytes=total,
                    reconciles=kv+hidden==total,completed_NIXL_transfers=result.get('transport_completed_count','NOT_MEASURED'),
                    descriptors=sum(r['descriptors'] for r in payload) if payload else 'NOT_MEASURED',
                    host_cumulative_transfer_completion_observation_s=result.get('transport_completion_observation_seconds','NOT_MEASURED'),
                    pure_DMA_s='NOT_MEASURED',control_metadata_bytes='NOT_MEASURED',evidence=scope))
    for model in ('ouro','huginn'):
        for w in ('W1','W2','W3','W4'):
            for arm in ARMS:
                values=[r for r in all_metrics if r['model']==model and r['workload']==w and r['arm']==arm and r['phase']=='primary']
                expected_name=f'primary_{model}_{w.lower()}_{arm.lower()}_start0'
                state=values[-1]['status'] if values else skipped.get(expected_name,'NOT_RUN')
                matrix.append(dict(model=model,workload=w,arm=arm,status=state,
                    completed_N=sum(r['completed_N'] for r in values),run_names=';'.join(r['name'] for r in values)))
    csv_write(ROOT/'per_run_metrics.csv',all_metrics);csv_write(ROOT/'experiment_matrix.csv',matrix)
    csv_write(ROOT/'all_experiment_matrix.csv',all_cells)
    csv_write(ROOT/'e2e_component_shares.csv',shares)
    groups={}
    for r in shares:
        key=(r['name'],r['component']);g=groups.setdefault(key,dict(name=r['name'],model=r['model'],arm=r['arm'],
            workload=r['workload'],component=r['component'],component_ms=0.,summed_request_E2E_ms=0.,requests=0,
            evidence='HOST_REQUEST_TIME_SHARES_NOT_COHORT_CRITICAL_PATH_OR_GPU_SHARES'))
        g['component_ms']+=r['component_ms'];g['summed_request_E2E_ms']+=r['E2E_ms'];g['requests']+=1
    for g in groups.values():g['share_percent']=100*g['component_ms']/g['summed_request_E2E_ms']
    csv_write(ROOT/'e2e_component_shares_aggregate.csv',groups.values())
    csv_write(ROOT/'communication_payloads.csv',communication)
    csv_write(ROOT/'online_cut_outcomes.csv',cut_outcomes)
    csv_write(ROOT/'online_policy_reason_accounting.csv',policy_reasons)
    with (ROOT/'per_request.jsonl').open('w') as stream:
        for row in requests:stream.write(json.dumps(row)+'\n')
    with (ROOT/'online_decisions.jsonl').open('w') as stream:
        for row in online:stream.write(json.dumps(row)+'\n')
    write(ROOT/'status.json',dict(updated=time.time(),primary_complete=sum(r['status']=='COMPLETE' for r in matrix),
        primary_total=56,completed_primary_requests=sum(r['phase']=='primary' for r in requests),
        completed_repeat_requests=sum(r['phase']=='repeat' for r in requests),elapsed_hours=(time.time()-START_EPOCH)/3600))
    print(json.loads((ROOT/'status.json').read_text()))

if __name__=='__main__':main()
