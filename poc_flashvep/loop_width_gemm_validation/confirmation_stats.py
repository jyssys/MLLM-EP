"""Use independent process starts, not requests, as the resampling unit."""
import itertools,math,random
from collections import defaultdict
from .common import *
from .cost_analysis import quantile


def paired_ratio(values):
    if not values or any(a<=0 or b<=0 for a,b in values):return dict(status='UNDEFINED_NONPOSITIVE_METRIC')
    logs=[math.log(a/b) for a,b in values];n=len(logs)
    if n<=5:
        samples=[math.exp(sum(logs[i] for i in indices)/n) for indices in itertools.product(range(n),repeat=n)]
    else:
        rng=random.Random(2026111119)
        samples=[math.exp(sum(logs[rng.randrange(n)] for _ in range(n))/n) for _ in range(10000)]
    return dict(ratio=math.exp(sum(logs)/n),CI95=[quantile(samples,2.5),quantile(samples,97.5)],starts=n,
        per_start_ratios=[a/b for a,b in values],
        scope='Paired percentile bootstrap of the geometric mean. Resampling unit is the independent process start. Small-start uncertainty remains limited.')


def main():
    initialize();source=json.loads((ROOT/'analysis/serving_results.json').read_text())
    rows=[r for r in source if r.get('complete') and r['count']==128 and 'CONFIRMATION' in r.get('phase','')]
    groups=defaultdict(list)
    for row in rows:groups[row['model'],row['cap'],row['width_arm']].append(row)
    summaries=[]
    for (model,cap,arm),runs in sorted(groups.items()):
        if arm=='zero':continue
        baseline={r['arrival_seed']:r for r in groups[model,cap,'zero']}
        paired=[(r,baseline[r['arrival_seed']]) for r in sorted(runs,key=lambda x:x['arrival_seed']) if r['arrival_seed'] in baseline]
        result=dict(model=model,cap=cap,arm=arm,arrival_seeds=[r['arrival_seed'] for r,b in paired],
            all_exact=all(r['exact_output_work_lifetime_passed'] and b['exact_output_work_lifetime_passed'] for r,b in paired),
            absolute_Decode_SLO=all(r['Decode_SLO_screen'] for r,b in paired),
            ITL_threshold_misses=sum(r['ITL_limit_misses'] for r,b in paired),
            baseline_ITL_threshold_misses=sum(b['ITL_limit_misses'] for r,b in paired))
        for field in ('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms','SLO_goodput','completed_QPS'):
            result[field]=paired_ratio([(r[field],b[field]) for r,b in paired])
        result['relative_tail_CI_gate']=all(result[f].get('starts',0)>=3 and result[f]['CI95'][1]<=1.05 for f in ('ITL_p99_ms','TPOT_p95_ms'))
        result['TTFT_improvement_CI_gate']=result['TTFT_p95_ms'].get('starts',0)>=3 and result['TTFT_p95_ms']['CI95'][1]<1
        summaries.append(result)
    comparisons=[]
    for first,second in (('loop','token'),('loop28','token1792'),('loop28','token')):
        a={r['arrival_seed']:r for r in groups.get(('huginn',4,first),[])}
        b={r['arrival_seed']:r for r in groups.get(('huginn',4,second),[])}
        seeds=sorted(a.keys()&b.keys())
        if not seeds:continue
        record=dict(model='huginn',cap=4,numerator=first,denominator=second,arrival_seeds=seeds,
            scope='Held-out paired method comparison. Equal recurrent rows do not imply equal Attention or Prelude/Coda work.')
        for field in ('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput','completed_QPS'):
            record[field]=paired_ratio([(a[s][field],b[s][field]) for s in seeds])
        comparisons.append(record)
    write(ROOT/'analysis/paired_start_confirmation.json',summaries)
    write(ROOT/'analysis/paired_method_comparisons.json',comparisons)
    print(json.dumps(summaries))


if __name__=='__main__':main()
