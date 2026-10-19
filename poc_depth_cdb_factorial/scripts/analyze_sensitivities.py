"""Report matched historical placements, one shuffled order, and quality work drift."""
import numpy as np
from poc_depth_cdb_factorial.contract import *
from poc_depth_cdb_factorial.scripts.analyze import paired_ratio,records,parity

def mean_ratio(a,b):
    a,b=np.asarray(a),np.asarray(b)
    rng=np.random.default_rng(20261019)
    idx=rng.integers(0,len(a),size=(2000,len(a)))
    draws=a[idx].mean(axis=1)/b[idx].mean(axis=1)
    return dict(speedup=float(a.mean()/b.mean()),CI95=np.quantile(draws,[.025,.975]).tolist(),
        statistic='Ratio of means of per-repeat P95 request latency',n_paired_repeats=len(a))

def main():
    metrics=read(ROOT/'artifacts/metrics_raw.json')
    summaries=[]
    for kind in ['sensitivity','order_control']:
        for C in sorted({r['C'] for r in metrics if r['kind']==kind}):
            selected={a:sorted([r for r in metrics if r['kind']==kind and r['C']==C and r['arm']==a],key=lambda r:r['repeat']) for a in ['A1','A2','A3','A4']}
            if not all(selected.values()):continue
            row=dict(kind=kind,C=C)
            for a,b,label in [('A1','A2','no_refill'),('A3','A4','refill')]:
                row[label]=dict(makespan=paired_ratio([r['makespan_s'] for r in selected[a]],[r['makespan_s'] for r in selected[b]]),
                    P95_ratio_of_means=mean_ratio([r['e2e_P95_ms'] for r in selected[a]],[r['e2e_P95_ms'] for r in selected[b]]),
                    control_P95_ms=[r['e2e_P95_ms'] for r in selected[a]],hindsight_P95_ms=[r['e2e_P95_ms'] for r in selected[b]],
                    control_output_tokens=[r['output_tokens'] for r in selected[a]],hindsight_output_tokens=[r['output_tokens'] for r in selected[b]],
                    control_actual_work=[r['actual_recurrent_steps'] for r in selected[a]],hindsight_actual_work=[r['actual_recurrent_steps'] for r in selected[b]])
            summaries.append(row)
    write(ROOT/'artifacts/sensitivity_summary.json',summaries)
    quality=[]
    for arm in ARMS:
        selected=[r for r in metrics if r['kind']=='quality' and r['arm']==arm]
        if not selected:continue
        total=sum(r['makespan_s'] for r in selected)
        tokens=sum(r['output_tokens'] for r in selected)
        quality.append(dict(arm=arm,cohorts=len(selected),C_per_cohort=64,questions=sum(r['N'] for r in selected),
            sum_separate_C64_serving_s=total,output_tokens=tokens,scheduled_loop_work=sum(r['actual_recurrent_steps'] for r in selected),
            tokens_per_sum_serving_s=tokens/total,
            warning='Sum of four distinct C64 serving intervals. Not a C256 burst or wall time including intervening other-arm cases.'))
    write(ROOT/'artifacts/quality_work_and_service.json',quality)
    natural=[]
    for a,b,label in [('A1','A2','no_refill_hindsight'),('A3','A4','refill_hindsight')]:
        values={arm:sorted([r for r in metrics if r['kind']=='quality' and r['arm']==arm],key=lambda r:r['case']) for arm in [a,b]}
        if len(values[a])!=4 or len(values[b])!=4:continue
        left=np.asarray([r['makespan_s'] for r in values[a]])
        right=np.asarray([r['makespan_s'] for r in values[b]])
        rng=np.random.default_rng(20261019)
        index=rng.integers(0,4,size=(2000,4))
        draws=left[index].sum(axis=1)/right[index].sum(axis=1)
        natural.append(dict(comparison=label,control=a,hindsight=b,speedup=float(left.sum()/right.sum()),
            CI95=np.quantile(draws,[.025,.975]).tolist(),control_C64_s=left.tolist(),hindsight_C64_s=right.tolist(),
            bootstrap_unit='Four matched, different 64-question cohorts. One timing measurement per cohort.',
            statistic='Ratio of summed serving intervals, not median C64 burst makespan or a C256 burst.',
            label='EXPLORATORY_MEASURED_NATURAL_WORKLOAD_DIAGNOSTIC',quality_matched=False,
            control_tokens=sum(r['output_tokens'] for r in values[a]),hindsight_tokens=sum(r['output_tokens'] for r in values[b])))
    write(ROOT/'artifacts/natural_quality_timing_diagnostic.json',natural)
    print('SENSITIVITY_SUMMARY',len(summaries),'comparisons')

if __name__=='__main__':main()
