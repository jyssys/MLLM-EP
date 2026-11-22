"""Separate point winners, start-level uncertainty and decode safety."""
import itertools
import numpy as np

def paired_difference(left,right):
    a=np.asarray(left,float);b=np.asarray(right,float);assert len(a)==len(b)>0
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    delta=a-b;result=dict(estimate=float(delta.mean()),CI95=None,starts=len(delta),
        method='Paired server-start difference of means. Pointwise,unadjusted.')
    if len(delta)>=3:
        n=len(delta);indices=np.asarray(list(itertools.product(range(n),repeat=n))) if n<=5 else np.random.default_rng(2026112201).integers(0,n,(20000,n))
        result.update(CI95=np.percentile(delta[indices].mean(axis=1),[2.5,97.5]).tolist(),resamples=len(indices))
    return result

def ratio(left,right):
    a=np.asarray(left,float);b=np.asarray(right,float);assert len(a)==len(b)>0
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    if np.any(b<=0):return dict(estimate=None,CI95=None,starts=len(a),reason='ZERO_DENOMINATOR')
    result=dict(estimate=float(a.mean()/b.mean()),CI95=None,starts=len(a),
        method='Paired independent server-start empirical bootstrap; pointwise,unadjusted')
    if len(a)>=3:
        indices=np.asarray(list(itertools.product(range(len(a)),repeat=len(a)))) if len(a)<=5 else np.random.default_rng(2026112201).integers(0,len(a),(20000,len(a)))
        values=a[indices].mean(axis=1)/b[indices].mean(axis=1)
        result.update(CI95=np.percentile(values,[2.5,97.5]).tolist(),resamples=len(indices))
    return result

def point_winner(test):
    value=test['estimate']
    return 'NOT_RUN' if value is None else 'LAYER_POINT_WIN' if value<1 else 'TOKEN_POINT_WIN' if value>1 else 'EQUAL'

def confidence(test):
    ci=test['CI95']
    return 'INSUFFICIENT_STARTS' if ci is None else 'LAYER_STATISTICALLY_CONFIRMED' if ci[1]<1 else 'TOKEN_STATISTICALLY_CONFIRMED' if ci[0]>1 else 'TIE_OR_INCONCLUSIVE'

def ratio_of_ratios(left_high,right_high,left_low,right_low):
    arrays=[np.asarray(v,float) for v in (left_high,right_high,left_low,right_low)]
    assert len({len(v) for v in arrays})==1 and len(arrays[0])>0 and all(np.all(v>0) for v in arrays)
    n=len(arrays[0]);a,b,c,d=arrays;result=dict(estimate=float((a.mean()/b.mean())/(c.mean()/d.mean())),CI95=None,starts=n,
        method='Paired endpoint contrast of ratios of start means. Pointwise,unadjusted.')
    if n>=3:
        indices=np.asarray(list(itertools.product(range(n),repeat=n))) if n<=5 else np.random.default_rng(2026112201).integers(0,n,(20000,n))
        values=(a[indices].mean(axis=1)/b[indices].mean(axis=1))/(c[indices].mean(axis=1)/d[indices].mean(axis=1))
        result.update(CI95=np.percentile(values,[2.5,97.5]).tolist(),resamples=len(indices))
    return result

def safety(arm,references):
    tests={name:{field:ratio([x[field] for x in arm],[x[field] for x in values]) for field in ('ITL_p99_ms','TPOT_p95_ms')}
        for name,values in references.items()}
    complete=set(references)=={'NativeStrong','NativeCommon'} and len(arm)>=3
    relative=complete and all(t['CI95'] is not None and t['CI95'][1]<=1.05 for fields in tests.values() for t in fields.values())
    absolute=all(x['decode_SLO_attainment']>=.95 for x in arm)
    return dict(tests=tests,both_native_references_complete=complete,relative_5percent_pass=bool(relative),
        absolute_decode_attainment_95percent_pass=absolute,qualified=bool(relative and absolute),
        thresholds=dict(every_client_ITL_ms=34,request_TPOT_ms=33),
        per_start_decode_attainment=[x['decode_SLO_attainment'] for x in arm],
        per_start_all_ITL_request_attainment=[x['every_ITL_SLO_attainment'] for x in arm],
        per_start_max_ITL_ms=[x['max_ITL_ms'] for x in arm])
