"""Process-block serving summaries. This module never selects a policy."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,write
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def num(row,key):
    try:return float(row[key])
    except (KeyError,ValueError,TypeError):return None

def main():
    rows=list(csv.DictReader((ROOT/'serving_results.csv').open()));groups=defaultdict(list)
    for row in rows:
        groups[(row['split'],row['lane'],row['workload'],row['offered_rate_rps'],row['width'],row['policy'])].append(row)
    rng=np.random.default_rng(2026102317);out=[]
    metrics=['output_tokens_s','drain_goodput_TTFT5s_TBT200ms_rps','joint_attainment_TTFT5s_TBT200ms_pct',
        'TTFT_P50_ms','TTFT_P95_ms','TTFT_P99_ms','ITL_P50_ms','ITL_P95_ms','ITL_P99_ms',
        'E2E_P50_ms','E2E_P95_ms','E2E_P99_ms','complete_E2E_P95_ms','TPOT_P50_ms','TPOT_P95_ms',
        'peak_HBM_GiB','peak_reserved_GiB','queue_slope_requests_s','TTFT_attainment_5s_pct',
        'complete_every_gap_TBT_attainment_200ms_pct','arrival_window_goodput_TTFT5s_TBT200ms_rps']
    for key,vs in groups.items():
        blocks=defaultdict(list)
        for r in vs:blocks[r['run_id']].append(r)
        item=dict(zip(('split','lane','workload','rate_parameter_rps','width','policy'),key),
            independent_starts=len(blocks),cohorts=len(vs),offered=sum(int(r['offered']) for r in vs),
            completed=sum(int(r['completed']) for r in vs),caps=sum(int(r['caps']) for r in vs),
            output_tokens=sum(int(r['output_tokens']) for r in vs),
            selector_calls=sum(int(r['selector_calls']) for r in vs),fresh_calls=sum(int(r['fresh_calls']) for r in vs),
            multiple_feasible_C_fresh=sum(int(r['multiple_feasible_C_fresh']) for r in vs),
            C_differs_fixed_same_state=sum(int(r['C_differs_fixed_same_state']) for r in vs),
            all_R4=all(r['full_R4']=='True' for r in vs),
            GPU_ids=','.join(sorted({r['GPU'] for r in vs})),
            scope='LIVE_MEASURED_MEAN_OF_COHORT_QUANTILES. Bootstrap resamples process starts. No token-independent sampling. Finite drained cohorts, not capacity.')
        item['fresh_C_change_fraction']=item['C_differs_fixed_same_state']/max(1,item['fresh_calls'])
        for metric in metrics:
            b=[]
            for block in blocks.values():
                numbers=[num(r,metric) for r in block];numbers=[x for x in numbers if x is not None]
                if numbers:b.append(float(np.mean(numbers)))
            if b:
                b=np.array(b);boot=b[rng.integers(0,len(b),size=(3000,len(b)))].mean(axis=1)
                item[metric]=float(b.mean())
                item[metric+'_CI95_low']=None if len(b)<2 else float(np.percentile(boot,2.5))
                item[metric+'_CI95_high']=None if len(b)<2 else float(np.percentile(boot,97.5))
            else:
                for suffix in ('','_CI95_low','_CI95_high'):item[metric+suffix]=None
        out.append(item)
    csv_file(ROOT/'main_table.csv',out)
    final=[r for r in out if r['split']=='FINAL'];paired=list(csv.DictReader((ROOT/'paired_comparisons.csv').open()))
    primary=[]
    for r in paired:
        if r['split']!='FINAL' or r['candidate']!='O6_J' or r['reference'] not in ('O6_G','O6_G1024'):continue
        if r['metric']!='drain_goodput_TTFT5s_TBT200ms_rps':continue
        selected=next(x for x in final if x['lane']==r['lane'] and x['workload']==r['workload'] and
            str(x['rate_parameter_rps'])==r['offered_rate_rps'] and x['policy']=='O6_J')
        primary.append(dict(r,actual_C_screen=selected['fresh_C_change_fraction']>=.05,
            point_goodput_screen=num(r,'ratio') is not None and num(r,'ratio')>=1.05,
            paired_delta_CI_excludes_zero=num(r,'CI95_low')>0,
            TTFT_regression_screen='Review same-stratum TTFT CI and output parity before promotion',
            scope='Registered threshold review only. Quality, outputs, two strata and quiet confirmation remain separate requirements.'))
    csv_file(ROOT/'final_primary_gate_evidence.csv',primary)
    write(ROOT/'table_analysis_status.json',dict(cells=len(out),FINAL_cells=len(final),primary_contrasts=len(primary),
        qualified_sustainable_capacity=False,capacity_measurement_status='NOT_RUN_UNLESS_ALL_INCREMENTAL_GATES_PASS'))
    print('Serving table cells',len(out),'FINAL cells',len(final),'primary comparisons',len(primary))

if __name__=='__main__':main()
