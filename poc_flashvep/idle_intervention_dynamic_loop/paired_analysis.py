"""Use paired independent starts for ratios. Tokens are not independent starts."""
import math
import random
from collections import defaultdict
from .common import *
from .analyze import percentile
from .campaign_report import write_csv


def paired(rows):
    grouped=defaultdict(dict)
    planned=defaultdict(dict)
    for row in rows:
        if row['name'].startswith('confirm_'):planned[row['model'],row['arrival_seed']][row['method']]=row
        if row['name'].startswith('confirm_') and row.get('complete') and row.get('evidence_class')=='MARKER_FREE':
            grouped[row['model'],row['arrival_seed']][row['method']]=row
    results=[]
    for model in ('ouro','huginn'):
        methods=sorted({method for (m,_),arms in grouped.items() if m==model for method in arms})
        for reference in ('native','zero','dynamic_token'):
            for method in methods:
                if method==reference:continue
                for metric in ('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput','completed_QPS'):
                    samples=[];sources=[];unavailable=[]
                    for (m,seed),arms in sorted(planned.items()):
                        if m==model and reference in arms and method in arms:
                            if not arms[reference].get('complete') or not arms[method].get('complete'):
                                unavailable.append(dict(seed=seed,reason='INCOMPLETE_START',candidate=arms[method]['name']))
                    for (m,seed),arms in sorted(grouped.items()):
                        if m!=model or reference not in arms or method not in arms:continue
                        a,b=arms[method][metric],arms[reference][metric]
                        if not a or not b:
                            unavailable.append(dict(seed=seed,reason='ZERO_RATIO_DENOMINATOR_OR_NUMERATOR',candidate=arms[method]['name']))
                            continue
                        samples.append(a/b);sources.append(dict(seed=seed,candidate=arms[method]['name'],reference=arms[reference]['name']))
                    if not samples:continue
                    logs=[math.log(value) for value in samples];point=math.exp(sum(logs)/len(logs))
                    rng=random.Random(2026110917);replicates=[]
                    if len(samples)>=3:
                        for _ in range(10000):replicates.append(math.exp(sum(rng.choice(logs) for _ in logs)/len(logs)))
                    results.append(dict(model=model,method=method,reference=reference,metric=metric,
                        paired_starts=len(samples),start_ratios=samples,geometric_ratio=point,
                        unavailable_starts=unavailable,eligible_complete_three_start_comparison=len(samples)>=3 and not unavailable,
                        bootstrap_95_CI=[percentile(replicates,.025),percentile(replicates,.975)] if replicates else 'NOT_ESTIMATED_FEWER_THAN_THREE_STARTS',
                        sources=sources,scope='Paired start-level bootstrap. Three starts provide limited uncertainty coverage. No token-level independence assumption.'))
    write(ROOT/'paired_start_ratios.json',results);write_csv(ROOT/'paired_start_ratios.csv',results)
    return results


if __name__=='__main__':
    result=paired(json.loads((ROOT/'serving_runs.json').read_text()));print(json.dumps(result,indent=2))
