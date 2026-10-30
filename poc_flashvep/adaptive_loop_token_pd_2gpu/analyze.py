"""Measured request statistics. No simulated speedups or capacity interpolation."""
import csv
import json
import math
from .common import *
from .calibration import percentile

def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []

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
    return center-delta,center+delta

def metric(cfg,directory):
    rows=[r for r in read(directory/'requests.jsonl') if not r['warmup']]
    result=dict(name=cfg['name'],model=cfg['model'],arm=cfg['arm'],workload=cfg.get('workload'),
        phase=cfg.get('phase','gate'),start=cfg.get('start',0),expected_N=cfg['count'],completed_N=len(rows),
        rate=cfg.get('rate','NOT_MEASURED'),GPU_ids=str(cfg['gpus']),
        protocol='native_chunk_streaming' if cfg['arm']=='PD_G1' else 'snapshot' if cfg['arm'] not in ('DP','DP_Chunked') else 'none')
    status=directory/'status.json'
    result['status']=json.loads(status.read_text())['status'] if status.exists() else 'NOT_RUN'
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
    return result,rows

def main():
    setup();all_metrics=[];requests=[];matrix=[];online=[];shares=[]
    skipped={r['name']:r['reason'] for r in read(ROOT/'state/skipped.jsonl')+read(ROOT/'state/primary_skipped.jsonl')}
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if 'arm' not in cfg or cfg.get('tune_jobs'):continue
        directory=ROOT/'raw'/cfg['name']
        if not directory.exists():continue
        result,rows=metric(cfg,directory);all_metrics.append(result)
        if cfg.get('phase')=='primary':
            requests.extend(rows);online.extend(dict(name=cfg['name'],**r) for r in read(directory/'online_decisions.jsonl'))
            closed=directory/'closed_stats.json'
            intervals={}
            if closed.exists():
                for m in json.loads(closed.read_text()).get('metrics',[]):intervals[m['request_id']]=m
            for r in rows:
                m=intervals.get(f'r{r["run"]}_q{r["sample_id"]}')
                if not m:continue
                t=m['timings'];created=m['created_monotonic'];end=r['finish']
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
    for model in ('ouro','huginn'):
        for w in ('W1','W2','W3','W4'):
            for arm in ARMS:
                values=[r for r in all_metrics if r['model']==model and r['workload']==w and r['arm']==arm and r['phase']=='primary']
                expected_name=f'primary_{model}_{w.lower()}_{arm.lower()}_start0'
                state=values[-1]['status'] if values else skipped.get(expected_name,'NOT_RUN')
                matrix.append(dict(model=model,workload=w,arm=arm,status=state,
                    completed_N=sum(r['completed_N'] for r in values),run_names=';'.join(r['name'] for r in values)))
    csv_write(ROOT/'per_run_metrics.csv',all_metrics);csv_write(ROOT/'experiment_matrix.csv',matrix)
    csv_write(ROOT/'e2e_component_shares.csv',shares)
    with (ROOT/'per_request.jsonl').open('w') as stream:
        for row in requests:stream.write(json.dumps(row)+'\n')
    with (ROOT/'online_decisions.jsonl').open('w') as stream:
        for row in online:stream.write(json.dumps(row)+'\n')
    write(ROOT/'status.json',dict(updated=time.time(),primary_complete=sum(r['status']=='COMPLETE' for r in matrix),
        primary_total=56,completed_primary_requests=len(requests),elapsed_hours=(time.time()-START_EPOCH)/3600))
    print(json.loads((ROOT/'status.json').read_text()))

if __name__=='__main__':main()
