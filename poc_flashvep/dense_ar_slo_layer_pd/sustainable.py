"""Evaluate fixed windows without removing late or unfinished arrivals."""
import bisect,collections,math,random,fcntl
from .common import *
from .configuration import base
from .analyze import records,percentile


def duration_config(name,arm,rate,cap=16,seed=2026111441,warm=60.,measure=240.,follow=60.,**changes):
    cfg=base(name,arm,count=1,cap=cap,rate=rate,seed=seed,**changes)
    rng=random.Random(seed);arrivals=[];at=0.;horizon=warm+measure+follow
    while at<horizon:
        arrivals.append(at);at+=rng.expovariate(rate)
    cfg.update(count=len(arrivals),request_caps=[cap]*len(arrivals),arrival_offsets=arrivals,
        dataset='duration_discovery_'+str(len(arrivals))+'.json',
        timeout=horizon+600,wall_timeout=horizon+1200,
        fixed_measurement=dict(warm_seconds=warm,measurement_seconds=measure,
            follow_through_seconds=follow,arrival_horizon_seconds=horizon,
            target_joint_attainment=.95,completion_rate_fraction=.98,
            outstanding_slope_max=.01,outstanding_slope_upper_max=.025,
            queue_block_seconds=30.,finite_operational_screen=True))
    pool_name=cfg.get('duration_prompt_pool','discovery.json')
    source=json.loads((ROOT/'data'/pool_name).read_text())
    if pool_name!='discovery.json':cfg['dataset']=Path(pool_name).stem+'_duration_N'+str(len(arrivals))+'.json'
    pool=[source[i%len(source)] for i in range(len(arrivals))]
    path=ROOT/'data'/cfg['dataset']
    # Expanded real-token datasets can be large. Do not serialize them while
    # another launcher's measured GPU window holds this same resource lease.
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if path.exists():assert json.loads(path.read_text())==pool
        else:write(path,pool)
    cfg['prompt_reuse_scope']='Cycle a declared64-prompt real-text discovery pool. Prefix caching is disabled. Cycles are not independent requests for inference.'
    if pool_name!='discovery.json':
        cfg['prompt_reuse_scope']=f'Cycle the declared{len(source)}-prompt pool {pool_name}. Prefix caching is disabled. Repetitions are not independent inference samples.'
    return cfg


def ols_slope(points):
    if len(points)<2:return None
    xm=sum(x for x,y in points)/len(points);ym=sum(y for x,y in points)/len(points)
    denominator=sum((x-xm)**2 for x,y in points)
    return sum((x-xm)*(y-ym) for x,y in points)/denominator if denominator else None


def assess_rows(cfg,rows,offered,anchor,queue_samples=()):
    fixed=cfg['fixed_measurement'];begin=anchor+fixed['warm_seconds'];end=begin+fixed['measurement_seconds']
    duration=end-begin;byid={r['sample_id']:r for r in rows}
    arrivals=sorted(o['arrival'] for o in offered);finishes=sorted(r['finish'] for r in rows)
    cohort=[o for o in offered if begin<=o['arrival']<end]
    completions=[r for r in rows if begin<=r['finish']<end]
    outstanding=lambda t:bisect.bisect_right(arrivals,t)-bisect.bisect_right(finishes,t)
    q=[(q['time']-begin,q['outstanding']) for q in queue_samples if begin<=q['time']<end]
    # Reconstruct at regular half-second boundaries if the run failed before its
    # queue log flushed. Every offered request still remains in the ledger.
    if not q:q=[(j/2,outstanding(begin+j/2)) for j in range(int(duration*2))]
    blocks=[];width=fixed['queue_block_seconds']
    for j in range(int(duration/width)):
        values=[v for t,v in q if j*width<=t<(j+1)*width]
        if values:blocks.append(((j+.5)*width,sum(values)/len(values)))
    slope=ols_slope(q);block_slope=ols_slope(blocks)
    bootstrap=[];rng=random.Random(1414)
    if len(blocks)>=4:
        # Resample two adjacent30-second residual blocks. Independent first
        # differences destroy the negative correlation of bounded queues.
        # This remains an empirical finite-window screen, not a stability proof.
        mean_x=sum(x for x,y in blocks)/len(blocks);mean_y=sum(y for x,y in blocks)/len(blocks)
        intercept=mean_y-block_slope*mean_x
        residuals=[y-(intercept+block_slope*x) for x,y in blocks]
        for _ in range(2000):
            selected=[]
            while len(selected)<len(blocks):
                begin_block=rng.randrange(len(blocks))
                selected.extend(residuals[(begin_block+j)%len(blocks)] for j in range(2))
            bootstrap.append(ols_slope([(x,intercept+block_slope*x+error)
                for (x,y),error in zip(blocks,selected)]))
    upper=percentile(bootstrap,.95)
    arrival_rate=len(cohort)/duration;completion_rate=len(completions)/duration
    initial=outstanding(begin);terminal=outstanding(end)
    stability=dict(arrival_rate=arrival_rate,completion_rate=completion_rate,
        completion_arrival_ratio=completion_rate/arrival_rate if arrival_rate else None,
        outstanding_start=initial,outstanding_end=terminal,outstanding_delta=terminal-initial,
        outstanding_slope_per_s=slope,block_mean_slope_per_s=block_slope,
        empirical_slope_upper_95=upper,queue_blocks=blocks,
        uncertainty_estimator='CIRCULAR_RESIDUAL_BLOCK_OLS_V2_TWO_ADJACENT_30S_BLOCKS')
    reasons=[]
    if not cohort:reasons.append('NO_MEASUREMENT_ARRIVALS')
    if completion_rate<fixed['completion_rate_fraction']*arrival_rate:reasons.append('COMPLETION_RATE_BELOW_ARRIVALS')
    if slope is None or slope>fixed['outstanding_slope_max']:reasons.append('OUTSTANDING_QUEUE_GROWS')
    if upper is None or upper>fixed['outstanding_slope_upper_max']:reasons.append('QUEUE_GROWTH_UNCERTAINTY')
    if terminal>initial+max(4,.01*len(cohort)):reasons.append('END_OUTSTANDING_INCREASE')
    thresholds=json.loads((ROOT/'configs/frozen_SLO_regimes.json').read_text())['output_caps']
    def intervals(row):
        return [1000*(b['time']-a['time'])/(b['count']-a['count']) for a,b in zip(row['token_events'],row['token_events'][1:])]
    def passes(row,regime):
        limits=thresholds[str(cfg['request_caps'][row['sample_id']])][regime]
        return row['ttft_ms']<=limits['TTFT_ms'] and row['tpot_ms']<=limits['TPOT_ms'] and max(intervals(row),default=0)<=limits['ITL_ms']
    regimes={}
    for regime in ('TTFT_priority','Balanced','Decode_priority'):
        # Follow all measurement arrivals until final outcome. Missing requests
        # remain in this denominator and fail the SLO test.
        cohort_pass=sum(o['sample_id'] in byid and passes(byid[o['sample_id']],regime) for o in cohort)
        eligible=[r for r in completions if passes(r,regime)]
        attain=cohort_pass/len(cohort) if cohort else 0.
        window_attain=len(eligible)/len(completions) if completions else 0.
        rejection=list(reasons)
        if attain<fixed['target_joint_attainment']:rejection.append('ARRIVAL_COHORT_JOINT_SLO_BELOW_TARGET')
        if window_attain<fixed['target_joint_attainment']:rejection.append('COMPLETION_WINDOW_JOINT_SLO_BELOW_TARGET')
        regimes[regime]=dict(arrival_cohort_joint_attainment=attain,completion_window_joint_attainment=window_attain,
            SLO_qualified_output_tokens_s=sum(r['output_tokens'] for r in eligible)/duration,
            SLO_qualified_output_requests_s=len(eligible)/duration,qualified=not rejection,rejections=rejection)
    result=dict(name=cfg['name'],window_begin=begin,window_end=end,measurement_seconds=duration,
        measurement_arrivals=len(cohort),measurement_completions=len(completions),
        unfinished_measurement_arrivals=sum(o['sample_id'] not in byid for o in cohort),
        raw_completed_output_tokens_s=sum(r['output_tokens'] for r in completions)/duration,
        completed_output_requests_s=completion_rate,
        actual_token_events_in_window=sum(b['count']-(a['count'] if a else 0) for r in rows
            for a,b in zip([None]+r['token_events'][:-1],r['token_events']) if begin<=b['time']<end)/duration,
        stability=stability,SLO_regimes=regimes,evidence='MEASURED',
        scope='Fixed completion window and fully followed measurement-arrival cohort. Queue stability is a finite operational screen, not a mathematical proof of sustainability.')
    selected=[byid[o['sample_id']] for o in cohort if o['sample_id'] in byid]
    for label,key in (('TTFT','ttft_ms'),('TPOT','tpot_ms'),('E2E','e2e_ms')):
        for quantile in (.5,.95,.99):result[f'{label}_p{int(quantile*100)}_ms']=percentile([r[key] for r in selected],quantile)
    all_itl=[x for r in selected for x in intervals(r)]
    for quantile in (.5,.95,.99):result[f'ITL_p{int(quantile*100)}_ms']=percentile(all_itl,quantile)
    result['maximum_ITL_ms']=max(all_itl,default=None)
    return result


def assess(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    anchor=json.loads((folder/'measurement_begin.json').read_text())['time']
    result=assess_rows(cfg,records(folder/'requests.jsonl'),records(folder/'offered_requests.jsonl'),anchor,
        records(folder/'queue_observations.jsonl'))
    receipt=json.loads((folder/'dispatcher_result.json').read_text())
    result['runtime_complete']=receipt.get('complete',False)
    gate_path=ROOT/'analysis'/(name+'_output_gate.json')
    output_gate=json.loads(gate_path.read_text()).get('passed',False) if gate_path.exists() else False
    lifetime={p.name:json.loads(p.read_text()).get('passed',False) for p in folder.glob('*lifetime_*.json')}
    result['output_correctness_passed']=output_gate
    result['normal_lifetime_checks']=lifetime
    coalesced=sum(b['count']-a['count']!=1 for row in records(folder/'requests.jsonl')
        if result['window_begin']<=row['arrival']<result['window_end']
        for a,b in zip(row['token_events'],row['token_events'][1:]))
    result['measurement_arrival_coalesced_intervals']=coalesced
    resource_failures=[]
    if not output_gate:resource_failures.append('EXACT_OUTPUT_GATE_NOT_PASSED')
    if not lifetime or not all(lifetime.values()):resource_failures.append('NORMAL_RESOURCE_LIFETIME_NOT_PASSED')
    if coalesced:resource_failures.append('UNRESOLVED_COALESCED_TOKEN_INTERVALS')
    for v in result['SLO_regimes'].values():
        if not result['runtime_complete']:v['qualified']=False;v['rejections'].append('RUNTIME_NOT_COMPLETE')
        if resource_failures:v['qualified']=False;v['rejections'].extend(resource_failures)
    write(ROOT/'analysis'/(name+'_sustainable_v2.json'),result);return result
