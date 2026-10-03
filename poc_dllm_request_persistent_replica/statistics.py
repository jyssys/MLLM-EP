"""Request-clustered bootstrap for the specification's aggregate byte ratios."""
import numpy as np


def bootstrap_ratio(numerator,denominator,repeats=2000):
    n=np.asarray(numerator,dtype=float);d=np.asarray(denominator,dtype=float)
    valid=np.isfinite(n)&np.isfinite(d);n=n[valid];d=d[valid]
    if not len(n) or d.sum()<=0:return None
    indices=np.random.default_rng(8611).integers(0,len(n),size=(repeats,len(n)))
    dn=d[indices].sum(axis=1);nn=n[indices].sum(axis=1)
    ratios=nn[dn>0]/dn[dn>0]
    return dict(n=len(n),mean=float(n.sum()/d.sum()),ci95=np.quantile(ratios,[.025,.975]).tolist(),
        estimator='ratio of aggregate byte gains; paired request resampling',valid_bootstrap_repeats=len(ratios))
