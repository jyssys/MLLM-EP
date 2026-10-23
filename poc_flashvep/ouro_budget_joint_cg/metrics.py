"""Client clocks, complete-response denominators, and cohort uncertainty."""
import numpy as np

def percentile(values,q):
    return float(np.percentile(values,q)) if values else None

def summarize(data,ttft_s=5.,tbt_s=.2):
    requests=data['requests'];gaps=[g for r in requests for g in r.get('ITLs_s',[])]
    ttft=[r['TTFT_s'] for r in requests if r.get('TTFT_s') is not None]
    ends=[r.get('completion',data['end']) for r in requests]
    start=data['start'];drain=max(ends,default=data['end'])-start
    tokens=sum(len(r['output_ids']) for r in requests)
    complete=[r for r in requests if r['status']=='COMPLETE' and len(r['output_ids'])<r['cap']]
    capped=sum(len(r['output_ids'])>=r['cap'] for r in requests)
    def good(r):
        return r in complete and r.get('TTFT_s',float('inf'))<=ttft_s and max(r.get('ITLs_s',[]),default=0.)<=tbt_s
    passed=sum(good(r) for r in requests)
    return dict(offered=len(requests),complete_responses=len(complete),capped=capped,
        error_or_timeout=sum(r['status'] in ('ERROR','TIMEOUT') for r in requests),
        output_tokens=tokens,drained_wall_s=drain,drained_output_tokens_s=tokens/max(drain,1e-9),
        drained_completed_requests_s=len(complete)/max(drain,1e-9),
        TTFT_P50_ms=1000*percentile(ttft,50) if ttft else None,
        TTFT_P95_ms=1000*percentile(ttft,95) if ttft else None,
        ITL_pooled_P50_ms=1000*percentile(gaps,50) if gaps else None,
        ITL_pooled_P95_ms=1000*percentile(gaps,95) if gaps else None,
        ITL_pooled_P99_ms=1000*percentile(gaps,99) if gaps else None,
        E2E_P50_ms=1000*percentile([r['E2E_s'] for r in requests],50),
        joint_complete_attainment_pct=100*passed/max(1,len(requests)),
        joint_declared_drain_goodput_rps=passed/max(drain,1e-9),
        SLO_TTFT_s=ttft_s,SLO_every_gap_TBT_s=tbt_s,
        capacity='UNQUALIFIED_SMALL_BOUNDED_COHORT')

def prefill_exposures(data):
    rows=data.get('service_observations',[]);result=[]
    all_prefills=[q for q in rows if any(s['phase'].startswith('prefill') for s in q['spans'])]
    for r in data['requests']:
        for left,right in zip(r['token_times'],r['token_times'][1:]):
            prefills=[q for q in all_prefills if left<=q['host_dispatch_begin_s']<right]
            overlapping=[q for q in all_prefills if q['aligned_service_begin_s']<right and q['aligned_service_end_s']>left]
            intervals=sorted((max(left,q['aligned_service_begin_s']),min(right,q['aligned_service_end_s']))
                for q in overlapping)
            union=0.;end=-float('inf')
            for begin,finish in intervals:
                union+=max(0.,finish-max(begin,end));end=max(end,finish)
            result.append(dict(id=r['id'],left_s=left,right_s=right,ITL_ms=1000*(right-left),
                submitted_prefills=len(prefills),overlapping_prefill_intervals=len(overlapping),prefill_interval_union_ms=1000*union,
                evidence='MEASURED_OBSERVER_SAME_GPU_ALIGNMENT_NOT_ADDITIVE_E2E_ATTRIBUTION'))
    return result

def paired_block_interval(pairs,seed=20261023):
    # Pairs contain independent process/GPU block aggregates, not token IID rows.
    if not pairs:
        return None
    differences=np.array([b-a for a,b in pairs],dtype=float)
    rng=np.random.default_rng(seed)
    samples=[np.mean(rng.choice(differences,len(differences),replace=True)) for _ in range(5000)]
    return dict(mean_difference=float(differences.mean()),CI95_low=float(np.percentile(samples,2.5)),
        CI95_high=float(np.percentile(samples,97.5)),independent_blocks=len(pairs))
