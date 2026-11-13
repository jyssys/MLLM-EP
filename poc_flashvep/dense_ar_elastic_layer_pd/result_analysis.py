"""Keep independent-start uncertainty separate from per-token samples."""
import csv,itertools,statistics
from .common import *
from .analyze import metrics,records,percentile
from .statistics import paired_ratio

METRICS=('TTFT_p50_ms','TTFT_p95_ms','TTFT_p99_ms','TPOT_p50_ms','TPOT_p95_ms',
    'ITL_p50_ms','ITL_p95_ms','ITL_p99_ms','maximum_ITL_ms','E2E_p95_ms',
    'output_tokens_s','input_tokens_s','completed_QPS','worker_emission_ITL_p99_ms',
    'coordinator_delivery_lag_p95_ms')


def estimate(values):
    if not values:return dict(mean=None,independent_starts=0,CI95=None)
    n=len(values)
    draws=[sum(values[i] for i in indices)/n for indices in itertools.product(range(n),repeat=n)]
    return dict(mean=statistics.mean(values),independent_starts=n,
        CI95=[percentile(draws,.025),percentile(draws,.975)] if n>=2 else None,
        scope='Empirical independent-start bootstrap. Three starts give a small finite sampling distribution, not a production guarantee.')


def collected():
    plan_path=ROOT/'state/confirmation_plan.json'
    if not plan_path.exists():return []
    rows=[]
    for item in json.loads(plan_path.read_text())['jobs']:
        name=item['name'];m=metrics(name)
        if not m.get('complete'):continue
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        requests=records(ROOT/'raw'/name/'requests.jsonl')
        rows.append(dict(name=name,arm=item['arm'],cap=cfg['cap'],rate=cfg['arrival_rate'],seed=cfg['arrival_seed'],
            metrics=m,migrated_requests=sum(r.get('completed_D_prefill_layers',0)>0 for r in requests),
            migrated_layers=sum(r.get('completed_D_prefill_layers',0) for r in requests),
            completed_layer_distribution={str(k):sum(r.get('completed_D_prefill_layers',0)==k for r in requests) for k in (0,2,4,8,12)}))
    return rows


def main():
    rows=collected();cells=[]
    for cap,rate in sorted({(r['cap'],r['rate']) for r in rows if r['arm']!='Reference'}):
        cell=dict(cap=cap,rate=rate,arms={})
        native={r['seed']:r for r in rows if r['cap']==cap and r['rate']==rate and r['arm']=='Native'}
        for arm in ('Native','Colocated','Static','Revised'):
            selected=[r for r in rows if r['cap']==cap and r['rate']==rate and r['arm']==arm]
            value=dict(names=[r['name'] for r in selected],seeds=[r['seed'] for r in selected],
                estimates={k:estimate([r['metrics'][k] for r in selected]) for k in METRICS},
                migrated_requests=sum(r['migrated_requests'] for r in selected),migrated_layers=sum(r['migrated_layers'] for r in selected),
                total_requests=128*len(selected),layer_distribution={str(k):sum(r['completed_layer_distribution'][str(k)] for r in selected) for k in (0,2,4,8,12)},
                SLO={})
            for regime in ('TTFT_priority','Balanced','Decode_priority'):
                value['SLO'][regime]={k:estimate([r['metrics']['SLO_regimes'][regime][k] for r in selected])
                    for k in ('TTFT_attainment','Decode_attainment','Joint_attainment','Request_SLO_goodput',
                        'Output_token_SLO_goodput','ITL_threshold_exceedances')}
            paired=sorted([r for r in selected if r['seed'] in native],key=lambda r:r['seed'])
            if len(paired)>=2:
                value['ratios_vs_Native']={k:paired_ratio([r['metrics'][k] for r in paired],
                    [native[r['seed']]['metrics'][k] for r in paired]) for k in METRICS}
                for regime in ('TTFT_priority','Balanced','Decode_priority'):
                    for k in ('Request_SLO_goodput','Output_token_SLO_goodput'):
                        denominator=[native[r['seed']]['metrics']['SLO_regimes'][regime][k] for r in paired]
                        if sum(denominator):
                            value['SLO'][regime][k+'_ratio_vs_Native']=paired_ratio(
                                [r['metrics']['SLO_regimes'][regime][k] for r in paired],denominator)
                value['Decode_NI_5pct_established']=len(paired)>=3 and all(
                    value['ratios_vs_Native'][k]['paired_bootstrap_95_percentile_CI'][1]<=1.05 for k in ('ITL_p99_ms','TPOT_p95_ms'))
                value['TTFT_5pct_target_and_CI_improvement']=len(paired)>=3 and (
                    value['ratios_vs_Native']['TTFT_p95_ms']['ratio_of_means']<=.95 and
                    value['ratios_vs_Native']['TTFT_p95_ms']['paired_bootstrap_95_percentile_CI'][1]<1.)
            cell['arms'][arm]=value
        cells.append(cell)
    write(ROOT/'analysis/four_method_qps_results.json',dict(results=rows,cells=cells,evidence='MEASURED',
        scope='Metrics are start-level quantiles, not pooled token quantiles. First ITL is included. All throughput and goodput include finite-cohort drain. Missing cells remain unmeasured.'))
    flat=[]
    for r in rows:
        row={k:r[k] for k in ('name','arm','cap','rate','seed','migrated_requests','migrated_layers')}
        row.update({k:r['metrics'][k] for k in METRICS})
        for regime in ('TTFT_priority','Balanced','Decode_priority'):
            row.update({regime+'_'+k:r['metrics']['SLO_regimes'][regime][k]
                for k in ('Joint_attainment','Request_SLO_goodput','Output_token_SLO_goodput','ITL_threshold_exceedances')})
        flat.append(row)
    if flat:
        with (ROOT/'analysis/independent_start_metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    pairs=[]
    for cap,rate in sorted({(r['cap'],r['rate']) for r in rows if r['arm']!='Reference'}):
        for candidate,reference in itertools.permutations(('Native','Colocated','Static','Revised'),2):
            baseline={r['seed']:r for r in rows if r['cap']==cap and r['rate']==rate and r['arm']==reference}
            selected=[r for r in rows if r['cap']==cap and r['rate']==rate and r['arm']==candidate and r['seed'] in baseline]
            if len(selected)<2:continue
            ratios={k:paired_ratio([r['metrics'][k] for r in selected],
                [baseline[r['seed']]['metrics'][k] for r in selected]) for k in METRICS}
            for regime in ('TTFT_priority','Balanced','Decode_priority'):
                for k in ('Request_SLO_goodput','Output_token_SLO_goodput'):
                    denominator=[baseline[r['seed']]['metrics']['SLO_regimes'][regime][k] for r in selected]
                    ratios[regime+'_'+k]=paired_ratio(
                        [r['metrics']['SLO_regimes'][regime][k] for r in selected],denominator) if sum(denominator) else None
            pairs.append(dict(cap=cap,rate=rate,candidate=candidate,reference=reference,ratios=ratios))
    write(ROOT/'analysis/all_pairwise_start_results.json',dict(results=pairs,evidence='TRACE_ANALYSIS',
        scope='Paired independent starts. No retuning or selection from these comparisons. A zero denominator has no defined goodput ratio.'))
    print('INDEPENDENT_STARTS',len(rows),'CELLS',len(cells))


if __name__=='__main__':main()
