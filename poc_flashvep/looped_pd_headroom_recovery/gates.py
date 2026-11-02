"""Evaluate same-start evidence. Missing GPU intervals never become idle time."""
import collections
import statistics
from .common import *
from .analyze import records,table
from .queues import analyze as queues
from .profile_analysis import union


def percentile(values,q):
    if not values:return None
    values=sorted(values);index=(len(values)-1)*q
    lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]+(index-lo)*(values[hi]-values[lo])


def actual_P_reference(cfg):
    values=[]
    for path in (ROOT/'configs').glob('capacity_*.json'):
        source=json.loads(path.read_text())
        if source.get('model')!=cfg['model'] or source.get('length')!=cfg['length']:continue
        # Same precision, numerical release and native local service path.
        # Both full and C2048 receive the same B1/2/4 tuning budget.
        for row in records(ROOT/'raw'/source['name']/'cohorts.jsonl'):
            if row.get('status')=='COMPLETE':values.append(row['job']['batch']/row['prefill_host_service_s'])
    return max(values,default=None)


def one(cfg):
    from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals
    folder=ROOT/'raw'/cfg['name']
    if not (folder/'window.json').exists():return None
    window=json.loads((folder/'window.json').read_text())
    if window['last_arrival']<=window['start']:return None
    rows=[r for r in records(folder/'requests.jsonl') if not r.get('warmup')]
    if not rows:return None
    left=window['start']+.2*(window['last_arrival']-window['start'])
    right=window['start']+.9*(window['last_arrival']-window['start'])
    if (folder/'profile_analysis.json').exists():
        profile_bounds=json.loads((folder/'profile_analysis.json').read_text())
        if profile_bounds.get('bounded_capture'):
            left=max(left,profile_bounds['window_start_monotonic'])
            right=min(right,profile_bounds['window_end_monotonic'])
            other=folder/'profile_analysis_decode.json'
            if other.exists():
                d=json.loads(other.read_text())
                left=max(left,d['window_start_monotonic']);right=min(right,d['window_end_monotonic'])
            if right<=left:return dict(name=cfg['name'],model=cfg['model'],classification='NO_VALID_SHARED_CAPTURE_WINDOW',same_start_joint_success=False)
    queue,timeline=queues(cfg['name'],window,observation=(left,right))
    central=[r for r in rows if any(left<=v['time']<=right for v in r['token_events'])]
    # Only intervals entirely contained in the shared central interval.
    itls=[]
    for row in rows:
        for event in row['token_events']:
            if not event['first'] and left<=event['previous_time']<=event['time']<=right:
                itls.append(1000*(event['time']-event['previous_time'])/event['new_tokens'])
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())
    slo=slos.get(cfg['model']+'_L'+str(cfg['length']))
    # Whole-request TPOT and global ITL remain in serving_runs.csv. This
    # table also exposes the aligned interval instead of stitching starts.
    tp=percentile([r['tpot_ms'] for r in central],.95)
    all_itls=[v for r in rows for v in token_intervals(r)[0]]
    tail=percentile(itls,.99)
    metrics=json.loads((folder/'closed_stats.json').read_text()).get('metrics',[]) if (folder/'closed_stats.json').exists() else []
    completed=[]
    for m in metrics:
        if not m['request_id'].startswith('r0_'):continue
        value=m['timings'].get('prefill_compute_completed')
        if value is not None:
            at=m['created_monotonic']+value
            if left<=at<=right:completed.append(at)
    rate=len(completed)/(right-left)
    reference=actual_P_reference(cfg)
    kernels=folder/'aligned_kernels.json'
    profile=folder/'profile_analysis.json'
    busy=None;profile_eligible=False
    if kernels.exists() and profile.exists() and cfg.get('capture_role','prefill')=='prefill':
        p=json.loads(profile.read_text());profile_eligible=p['eligible_for_matched_clean']
        value=json.loads(kernels.read_text())
        overlap=[(max(left,r['start_monotonic']),min(right,r['end_monotonic'])) for r in value['kernels'] if r['end_monotonic']>left and r['start_monotonic']<right]
        busy=union(overlap)/(right-left)
    credit=[r for r in records(folder/'queue_observations.jsonl') if left<=r['time']<=right]
    free=[]
    for sample in credit:
        for role,peer in sample['worker_credits'].items():
            if peer['role']=='decode':free.append(peer['total_blocks']-peer['reserved_blocks'])
    # Native block size is 16 tokens, with one block table per depth.
    # This is structural credit accounting, not a performed borrowing test.
    probe_blocks=cfg['depth']*4  # a 64-token, one-output probe
    admission=[r for r in records(folder/'admission_reasons.jsonl') if left<=r['time']<=right]
    refusals=sum(any(any(x.startswith('DECODE_') for x in q['reasons']) for q in r['waiting']) for r in admission)
    ratio=refusals/len(admission) if admission else None
    qpass=(queue['P_ready_lower_bound_half_growth'] is not None and queue['P_ready_lower_bound_half_growth']>0 and queue['P_ready_lower_bound_max']>=2)
    ppass=busy is not None and profile_eligible and busy>=.8 and reference is not None and rate>=.85*reference and qpass
    dpass=slo is not None and tp is not None and tail is not None and tp<=.9*slo['TPOT_ms'] and tail<=slo['ITL_ms']
    memory=bool(free) and min(free)>=probe_blocks
    if ratio is not None and ratio>.5:classification='KV_ADMISSION_BOUND'
    elif ppass:classification='P_COMPUTE_BOUND_BUT_D_SLACK_UNPROVEN'
    elif qpass and busy is not None and profile_eligible and busy<.8:classification='P_SERVICE_QUEUE_BUT_NOT_GPU_COMPUTE_BOUND'
    elif qpass:classification='PROFILING_OR_MEASUREMENT_BLOCKED'
    elif slo and not dpass:classification='DECODE_BOUND_OR_TAIL_LIMITED'
    else:classification='MIXED_OR_INCONCLUSIVE'
    return dict(name=cfg['name'],model=cfg['model'],length=cfg['length'],cap=cfg['cap'],chunk=cfg['chunk'],
        QPS=cfg.get('arrival_rate'),offered=cfg['count'],completed=len(rows),
        observation_start=left,observation_end=right,**queue,
        P_compute_completed_in_interval=len(completed),P_completion_rate_in_interval=rate,
        strongest_new_finite_P_reference=reference,P_fraction_reference=rate/reference if reference else None,
        P_actual_kernel_busy_central=busy,profile_perturbation_qualified=profile_eligible,
        D_existing_requests_with_tokens_in_interval=len(central),D_request_TPOT_p95_ms=tp,
        D_token_ITL_p99_central_ms=tail,D_token_ITL_p99_whole_run_ms=percentile(all_itls,.99),
        D_token_ITL_max_whole_run_ms=max(all_itls,default=None),
        frozen_TPOT_margin_ms=.9*slo['TPOT_ms'] if slo else None,frozen_ITL_limit_ms=slo['ITL_ms'] if slo else None,
        D_min_coordinator_free_blocks=min(free,default=None),structural_64_token_probe_blocks=probe_blocks,
        D_credit_refusal_sample_fraction=ratio,P_compute_gate=ppass,D_SLO_gate=dpass,D_credit_gate=memory,
        actual_D_borrowing='NOT_MEASURED',same_start_joint_success=False,
        classification=classification,evidence='MEASURED_EVENT_TRACE_ANALYSIS_WITH_EXPLICIT_MISSING_GPU_GATES',
        scope='Central interval is 20% to 90% of the actual arrival span. TPOT covers whole requests active in that interval. No clean/profile or P/D start stitching. No sustained-capacity or borrowing claim.')


def main():
    output=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or 'name' not in cfg or cfg.get('arm','').startswith('DP'):continue
        if cfg.get('state_probe') or cfg.get('verify_transfer') or cfg.get('calibration_serial') or cfg.get('prepare_D_only'):continue
        result=one(cfg)
        if result:output.append(result)
    table('p_d_gate_results.csv',output)
    write(ROOT/'p_d_gate_results.json',output)
    print(json.dumps(dict(runs=len(output),strict_successes=sum(r['same_start_joint_success'] for r in output))))


if __name__=='__main__':main()
