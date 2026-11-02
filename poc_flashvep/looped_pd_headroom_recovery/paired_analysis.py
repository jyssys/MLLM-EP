"""Report independent-start uncertainty without treating prompts as starts."""
import collections
import math
import statistics
from .common import *
from .analyze import table


def interval(values):
    if len(values)!=3:return None,None
    mean=statistics.mean(values)
    half=4.302652729911275*statistics.stdev(values)/math.sqrt(3)
    return mean-half,mean+half


def main():
    from .primary_campaign import subset_parity
    path=ROOT/'state/confirmation_plan.json'
    if not path.exists():
        table('paired_repetitions.csv',[])
        return
    plan=json.loads(path.read_text());runs=json.loads((ROOT/'serving_runs.json').read_text());index={r['name']:r for r in runs}
    pairs=[];summaries=[];parity=[]
    for item in plan['runs']:
        if item['PD'] not in index or item['DP'] not in index:continue
        a,b=index[item['PD']],index[item['DP']]
        if a['status']!='COMPLETE' or b['status']!='COMPLETE':continue
        exact=subset_parity(item['PD'],item['DP'])
        pair=dict(model=item['model'],start=item['start'],PD=item['PD'],DP=item['DP'],
            target_QPS=a['target_QPS'],PD_offered=a['offered'],DP_offered=b['offered'],exact_output_parity=exact,
            PD_SLO_goodput=a.get('SLO_goodput'),DP_SLO_goodput=b.get('SLO_goodput'),
            paired_goodput_difference_PD_minus_DP=a['SLO_goodput']-b['SLO_goodput'],
            paired_completed_throughput_ratio_PD_over_DP=a['requests_per_s']/b['requests_per_s'],
            paired_TTFT_p95_difference_ms=a['TTFT_ms_p95']-b['TTFT_ms_p95'],
            paired_TPOT_p95_difference_ms=a['request_TPOT_ms_p95']-b['request_TPOT_ms_p95'],
            paired_ITL_p99_difference_ms=a['token_ITL_ms_p99']-b['token_ITL_ms_p99'],
            evidence='MEASURED_MATCHED_INDEPENDENT_START_PAIR',
            scope='Same prompt IDs, generation seeds, arrival offsets and GPU pair. A finite N128 cohort is not a proven sustainable capacity.')
        pairs.append(pair)
    for model in ('ouro','huginn'):
        rows=[r for r in pairs if r['model']==model]
        for field in ('paired_goodput_difference_PD_minus_DP','paired_completed_throughput_ratio_PD_over_DP',
            'paired_TTFT_p95_difference_ms','paired_TPOT_p95_difference_ms','paired_ITL_p99_difference_ms'):
            values=[r[field] for r in rows];low,high=interval(values)
            summaries.append(dict(model=model,metric=field,independent_starts=len(values),
                mean=statistics.mean(values) if values else None,CI95_low=low,CI95_high=high,
                CI_method='Two-sided Student t, df=2' if len(values)==3 else 'NOT_AVAILABLE_NEEDS_THREE_STARTS',
                scope='Uncertainty across matched process starts. Do not interpret each token or prompt as an independent server start.'))
        names=[r['PD'] for r in rows]
        for name in names[1:]:
            parity.append(dict(model=model,reference=names[0],candidate=name,passed=subset_parity(names[0],name),
                evidence='MEASURED_SAME_GENERATION_SEED_DIFFERENT_ARRIVAL_SEED_OUTPUT_PARITY'))
    table('paired_repetitions.csv',pairs)
    table('paired_start_uncertainty.csv',summaries)
    write(ROOT/'state/cross_start_output_parity.json',parity)
    print(json.dumps(dict(pairs=len(pairs),summaries=len(summaries),cross_start_checks=len(parity))))


if __name__=='__main__':main()
