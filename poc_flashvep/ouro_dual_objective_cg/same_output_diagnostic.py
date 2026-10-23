"""Postselected equal-output checks. These do not replace natural serving."""
import csv
from collections import defaultdict
import numpy as np
from .common import ROOT
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    cohorts=list(csv.DictReader((ROOT/'whole_cohort_output_parity.csv').open()))
    serving={(r['run_id'],r['scenario'],r['policy']):r for r in csv.DictReader((ROOT/'serving_results.csv').open())}
    result=[];summary=defaultdict(list)
    for r in cohorts:
        if r['split']!='FINAL':continue
        left=serving[(r['run_id'],r['scenario'],r['candidate'])];right=serving[(r['run_id'],r['scenario'],r['reference'])]
        equal=r['all_request_tokens_exact']=='True' and r['complete_status_equal']==r['requests']
        row=dict(r,equal_output_cohort=equal,candidate_tokens=left['output_tokens'],reference_tokens=right['output_tokens'],
            candidate_caps=left['caps'],reference_caps=right['caps'],
            candidate_drained_wall_s=left['drained_wall_s'],reference_drained_wall_s=right['drained_wall_s'],
            equal_output_throughput_ratio=float(left['output_tokens_s'])/float(right['output_tokens_s']) if equal else None,
            diagnostic_scope='LIVE_MEASURED_POSTSELECTED_SUBSET. Equal-output membership is observed after execution. No independent causal or population claim.')
        result.append(row);summary[(r['lane'],r['workload'],r['rate'],r['candidate'],r['reference'])].append(row)
    table=[]
    for key,rs in summary.items():
        chosen=[r for r in rs if r['equal_output_cohort']]
        table.append(dict(zip(('lane','workload','rate','candidate','reference'),key),
            tested_cohorts=len(rs),equal_output_cohorts=len(chosen),
            independent_starts_in_subset=len({r['run_id'] for r in chosen}),
            mean_equal_output_ratio=None if not chosen else float(np.mean([r['equal_output_throughput_ratio'] for r in chosen])),
            scope='Postselected descriptive ratio only. No confidence or quality equivalence claim. Natural all-offered table remains primary.'))
    csv_file(ROOT/'same_output_cohort_diagnostics.csv',result);csv_file(ROOT/'same_output_cohort_summary.csv',table)
    print('Equal-output diagnostics',sum(r['equal_output_cohort'] for r in result),'/',len(result))

if __name__=='__main__':main()
