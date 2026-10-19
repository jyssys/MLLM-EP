"""Recompute historical metrics without changing historical evidence."""
import numpy as np
from poc_depth_cdb_factorial.contract import *
from poc_depth_cdb_factorial.scripts.analyze import paired_ratio

def main():
    old=read(V3_ROOT/'artifacts/live_hindsight_confirmation.json')
    rows=[]
    for C in [16,32,64]:
        pairs={arm:sorted([r for r in old['metrics'] if r['concurrency']==C and r['arm']==arm],key=lambda r:r['repeat'])
            for arm in ['generic','hindsight']}
        record=dict(C=C,source_sha256=sha(V3_ROOT/'artifacts/live_hindsight_confirmation.json'))
        for metric in ['makespan_ms','E2E_P95_ms']:
            a=[r[metric] for r in pairs['generic']]
            b=[r[metric] for r in pairs['hindsight']]
            record[metric]=dict(generic=a,hindsight=b,ratio_of_means=float(np.mean(a)/np.mean(b)),
                reduction_of_means_pct=100*(1-float(np.mean(b)/np.mean(a))),
                paired_repeat_median_speedup=paired_ratio(a,b))
        rows.append(record)
    write(ROOT/'artifacts/historical_v3_recomputed.json',dict(rows=rows,
        old_claim='4.41% is a P95 latency reduction of means, not cohort makespan or the mean of request speedups.',
        new_primary='Measured cohort makespan and deterministic depth-sum LPT.',
        CI_caution='The old request-conditional bootstrap differs from the repeat-level bootstrap used here.'))

if __name__=='__main__':main()
