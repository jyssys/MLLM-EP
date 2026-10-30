"""Paired bootstrap by independent start, never by individual token."""
import csv
import itertools
import statistics
from collections import defaultdict
from .common import ROOT
from .analyze import csv_write,quantile


def interval(values):
    if len(values)<3:return 'NOT_ESTIMABLE_N_LT3','NOT_ESTIMABLE_N_LT3'
    samples=[statistics.mean(values[i] for i in indices) for indices in itertools.product(range(len(values)),repeat=len(values))]
    return quantile(samples,.025),quantile(samples,.975)


def main():
    rows=[r for r in csv.DictReader((ROOT/'static_cut_metrics_by_start.csv').open())
          if int(r['prompt_tokens'])==512 and int(r['start'])<3 and r['regime']!='D_pressure' and r['handoff_mode']=='cut_snapshot']
    lookup={(r['model'],r['regime'],int(r['start']),int(r['cut'])):r for r in rows}
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['regime'],int(r['cut']))].append(r)
    paired=[];absolute=[]
    for (model,regime,cut),group in groups.items():
        depth=4 if model=='ouro' else 32
        values=[float(lookup[(model,regime,int(r['start']),depth)]['all_requests_e2e_ms_mean'])/float(r['all_requests_e2e_ms_mean']) for r in group]
        low,high=interval(values)
        paired.append(dict(model=model,regime=regime,cut=cut,independent_starts=len(values),
                           control='Matched full-cut snapshot. Not native chunk-streaming PD_G1.',
                           mean_paired_E2E_speedup_vs_full_PD=statistics.mean(values),CI95_low=low,CI95_high=high,
                           evidence='MEASURED_FINITE_COHORT_PAIRED_START_BOOTSTRAP',
                           scope='All-request mean E2E. Conditional on the same frozen eight prompts. Not stable goodput.'))
        for metric in ('ttft_ms_mean','tpot_ms_mean','all_requests_e2e_ms_mean'):
            actual=[float(r[metric]) for r in group];low,high=interval(actual)
            absolute.append(dict(model=model,regime=regime,cut=cut,metric=metric,independent_starts=len(actual),
                                 mean=statistics.mean(actual),CI95_low=low,CI95_high=high,
                                 scope='Bootstrap across starts. Small N. Not a benchmark-population confidence interval.'))
    csv_write('paired_cut_vs_full_pd_ci.csv',paired);csv_write('fixed_cut_mean_latency_ci.csv',absolute)
    path=ROOT/'fixed_regime_selection_comparison.csv'
    if not path.exists():return
    held=defaultdict(list)
    for r in csv.DictReader(path.open()):
        if r['calibration']=='False':held[r['model']].append(float(r['latency_reduction_pct']))
    comparisons=[]
    for model,values in held.items():
        low,high=interval(values)
        comparisons.append(dict(model=model,held_out_starts=len(values),mean_latency_reduction_pct=statistics.mean(values),
                                CI95_low=low,CI95_high=high,evidence='OFFLINE_FIXED_REGIME_COMPOSITION',
                                limitation='Not a dynamic serving oracle, not a goodput gain, same diagnostic prompts.'))
    csv_write('holdout_regime_selection_ci.csv',comparisons)
    print('PAIRED_CELLS',len(paired),'HOLDOUT_SUMMARIES',comparisons)


if __name__=='__main__':main()
