"""Use independent paired starts as the bootstrap sampling unit."""
import itertools,statistics
from .analyze import percentile


def paired_ratio(numerator,denominator):
    assert len(numerator)==len(denominator) and len(numerator)>=2
    n=len(numerator);ratios=[]
    for indices in itertools.product(range(n),repeat=n):
        a=sum(numerator[i] for i in indices);b=sum(denominator[i] for i in indices)
        if b:ratios.append(a/b)
    return dict(independent_start_pairs=n,ratio_of_means=statistics.mean(numerator)/statistics.mean(denominator),
        paired_bootstrap_95_percentile_CI=[percentile(ratios,.025),percentile(ratios,.975)],
        bootstrap_draws=len(ratios),method='Enumerate every empirical paired bootstrap sample of n independent starts. Percentile intervals use no request or token resampling.',
        limitation=f'{n} starts produce a finite empirical interval. This interval does not establish a production latency guarantee.')


def paired_difference(numerator,denominator):
    assert len(numerator)==len(denominator) and len(numerator)>=2
    differences=[a-b for a,b in zip(numerator,denominator)];n=len(differences)
    samples=[sum(differences[i] for i in indices)/n for indices in itertools.product(range(n),repeat=n)]
    return dict(independent_start_pairs=n,mean_difference=statistics.mean(differences),
        paired_bootstrap_95_percentile_CI=[percentile(samples,.025),percentile(samples,.975)],bootstrap_draws=len(samples))
