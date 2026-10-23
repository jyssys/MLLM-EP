"""Pooled quiet metrics with paired process-block uncertainty."""
import itertools
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .cost_predictor import csv_file
from .metrics import summarize

def statistics(cohorts):
    rows=[r for d in cohorts for r in d['requests']]
    gaps=[g for r in rows for g in r.get('ITLs_s',[])]
    summaries=[summarize(d) for d in cohorts]
    wall=sum(r['drained_wall_s'] for r in summaries)
    return dict(TTFT_request_pooled_P95_ms=1000*float(np.percentile([r['TTFT_s'] for r in rows],95)),
        ITL_token_pooled_P95_ms=1000*float(np.percentile(gaps,95)),
        ITL_token_pooled_P99_ms=1000*float(np.percentile(gaps,99)),
        E2E_request_pooled_P95_ms=1000*float(np.percentile([r['E2E_s'] for r in rows],95)),
        drained_output_tokens_s=sum(r['output_tokens'] for r in summaries)/wall,
        joint_TTFT5s_everygap200ms_drain_goodput=sum(r['joint_declared_drain_goodput_rps']*r['drained_wall_s'] for r in summaries)/wall)

def main():
    groups=defaultdict(dict)
    for folder in sorted((ROOT/'raw').glob('cb_p5_confirm_*')):
        if not (folder/'completed.json').exists():
            continue
        job=json.loads((folder/'identity.json').read_text())['job']
        B=next(c['budget_ms'] for c in job['cells'] if c['label']=='J1')
        policies=defaultdict(list)
        for file in folder.glob('*_T*.json'):
            data=json.loads(file.read_text())
            if 'requests' in data:
                policies[data['label']].append(data)
        groups[(job['shape'][0],job['shape'][1],B)][folder.name]=policies
    results=[]
    for (P,N,B),by_run in sorted(groups.items()):
        runs=sorted(by_run)
        if len(runs)!=3:
            continue
        for ref in ('C1A','C1S','G1','M0','M1','F1A'):
            if not all(ref in by_run[run] and 'J1' in by_run[run] for run in runs):
                continue
            a=statistics([d for run in runs for d in by_run[run][ref]])
            b=statistics([d for run in runs for d in by_run[run]['J1']])
            draws=defaultdict(list)
            for indices in itertools.product(range(len(runs)),repeat=len(runs)):
                ra=statistics([d for i in indices for d in by_run[runs[i]][ref]])
                rb=statistics([d for i in indices for d in by_run[runs[i]]['J1']])
                for metric in a:
                    draws[metric].append(rb[metric]-ra[metric])
            for metric in a:
                results.append(dict(P=P,N=N,budget_ms=B,reference=ref,candidate='J1',metric=metric,
                    reference_value=a[metric],joint_value=b[metric],joint_over_reference=b[metric]/a[metric] if a[metric] else None,
                    difference=b[metric]-a[metric],CI95_low=float(np.percentile(draws[metric],2.5)),CI95_high=float(np.percentile(draws[metric],97.5)),
                    independent_blocks=3,ordered_block_resamples=27,
                    inference='Paired GPU/model-process blocks, both source cohorts retained. Small-block exploratory interval.',
                    evidence='MEASURED_CLEAN_FINITE_COHORT_NOT_CAPACITY'))
    csv_file(ROOT/'main_paired_pooled_CI.csv',results)
    print('Quiet pooled paired metric rows',len(results))

if __name__=='__main__':
    main()
