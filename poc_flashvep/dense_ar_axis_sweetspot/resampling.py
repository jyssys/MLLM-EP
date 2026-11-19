"""Use empirical-CDF quantiles for equiprobable server-start bootstrap draws."""
import itertools
import math
import random
import statistics


METHOD='paired_start_percentile_bootstrap_inverse_empirical_CDF_v2'


def draws(n):
    if n<=6:
        return itertools.product(range(n),repeat=n)
    rng=random.Random(2026111788)
    return (tuple(rng.randrange(n) for _ in range(n)) for _ in range(20000))


def empirical_quantile(values,q):
    ordered=sorted(values)
    assert ordered and 0<=q<=1
    # Each enumerated draw has probability1/n**n. Do not interpolate away
    # endpoint mass when n=3 gives only27 distinct equiprobable index draws.
    index=max(0,min(len(ordered)-1,math.ceil(q*len(ordered))-1))
    return ordered[index]


def mean_interval(values):
    if len(values)<2:return None
    sample=[statistics.mean(values[i] for i in indices) for indices in draws(len(values))]
    return [empirical_quantile(sample,.025),empirical_quantile(sample,.975)]


def ratio_interval(reference,candidate):
    assert len(reference)==len(candidate)>0
    n=len(reference);a=statistics.mean(reference);b=statistics.mean(candidate)
    difference=[y-x for x,y in zip(reference,candidate)]
    result=dict(starts=n,ratio=b/a if a else None,difference=statistics.mean(difference),
        difference_CI95=mean_interval(difference),ratio_CI95=None,
        unit='Paired independent server start',reference_mean=a,candidate_mean=b,
        ratio_status='DEFINED' if a else 'UNDEFINED_ZERO_REFERENCE_MEAN',
        ratio_CI_status='NOT_ESTIMATED_SINGLE_START' if n<2 else 'PENDING',
        CI_method=METHOD,
        scope='Nominal95% percentile bootstrap. Exact enumeration for n≤6. Empirical-CDF quantiles preserve draw mass. Three starts still provide limited population-tail precision. No simultaneous/multiplicity guarantee.')
    if n<2:return result
    sample=[];zero=0
    for indices in draws(n):
        x=statistics.mean(reference[i] for i in indices)
        y=statistics.mean(candidate[i] for i in indices)
        if x==0:zero+=1
        else:sample.append(y/x)
    result['zero_denominator_draws']=zero
    result['ratio_CI_status']='UNDEFINED_ZERO_DENOMINATOR_DRAWS' if zero else 'ESTIMATED'
    if not zero:
        result['ratio_CI95']=[empirical_quantile(sample,.025),empirical_quantile(sample,.975)]
    return result
