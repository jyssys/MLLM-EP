"""Pair independent starts. Request-level intervals do not remove queue dependence."""
import random
import statistics
from .common import *
from .analyze import serving,save_csv,percentile
from .evidence import rows


def bootstrap(values,seed=20261101):
    values=list(values)
    if len(values)<2:return None,None
    rng=random.Random(seed);means=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(4000)]
    return percentile(means,.025),percentile(means,.975)


def main():
    runs=[r for r in serving() if r['phase']=='final_repeat' and r['eligible_clean']];groups={}
    for r in runs:groups.setdefault((r['model'],r['length'],r['output_cap'],r['target_QPS']),[]).append(r)
    output=[]
    for cell,group in groups.items():
        methods=sorted({(r['arm'],r['chunk']) for r in group})
        for ai,a in enumerate(methods):
            for b in methods[ai+1:]:
                x={r['independent_start']:r for r in group if (r['arm'],r['chunk'])==a};y={r['independent_start']:r for r in group if (r['arm'],r['chunk'])==b}
                starts=sorted(set(x)&set(y))
                if not starts:continue
                for metric in ('TTFT_ms_p95','request_TPOT_ms_p95','E2E_ms_p95','requests_per_s','output_tokens_per_s','SLO_goodput'):
                    paired=[(x[s][metric],y[s][metric]) for s in starts if isinstance(x[s].get(metric),(int,float)) and isinstance(y[s].get(metric),(int,float))]
                    if not paired:continue
                    difference=[v-u for u,v in paired];lo,hi=bootstrap(difference)
                    higher=metric in ('requests_per_s','output_tokens_per_s','SLO_goodput')
                    ratios=[v/u if higher else u/v for u,v in paired if u>0 and v>0];rlo,rhi=bootstrap(ratios)
                    output.append(dict(model=cell[0],length=cell[1],output_cap=cell[2],target_QPS=cell[3],method_A=a,method_B=b,
                        metric=metric,independent_paired_starts=len(paired),B_minus_A_mean=statistics.mean(difference),
                        paired_bootstrap95_low=lo,paired_bootstrap95_high=hi,
                        B_relative_performance_ratio_mean=statistics.mean(ratios) if ratios else None,
                        ratio_bootstrap95_low=rlo,ratio_bootstrap95_high=rhi,
                        ratio_scope='B/A for throughput and goodput. A/B for latency. Zero-valued pairs have no ratio.',
                        scope='Paired resampling of server starts. N=3 remains a small inferential sample. Do not treat queued requests as independent starts.'))
    save_csv('paired_start_comparisons.csv',output);print('PAIRED_START_METRICS',len(output));return output


if __name__=='__main__':main()
