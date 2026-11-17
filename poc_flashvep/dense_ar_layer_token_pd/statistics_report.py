"""Use independent server starts, not requests, as resampling units."""
import collections
import csv
import itertools
import json
import random
import statistics
from .common import ROOT,write
from .analysis import metrics
from ..dense_ar_slo_layer_pd.analyze import percentile

KEYS=('mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms',
    'E2E_p95_ms','maximum_ITL_ms','output_tokens_s','completed_QPS','Joint_attainment',
    'Decode_attainment','Output_token_SLO_goodput')


def draws(n):
    if n<=6:
        return itertools.product(range(n),repeat=n)
    rng=random.Random(2026111788)
    return (tuple(rng.randrange(n) for _ in range(n)) for _ in range(20000))


def mean_interval(values):
    if len(values)<2:
        return None
    sample=[statistics.mean(values[i] for i in indices) for indices in draws(len(values))]
    return [percentile(sample,.025),percentile(sample,.975)]


def ratio_interval(reference,candidate):
    assert len(reference)==len(candidate)
    n=len(reference)
    difference=[b-a for a,b in zip(reference,candidate)]
    result=dict(starts=n,ratio=statistics.mean(candidate)/statistics.mean(reference)
        if statistics.mean(reference) else None,
        difference=statistics.mean(difference),difference_CI95=mean_interval(difference),
        ratio_CI95=None,unit='Paired independent server start',
        reference_mean=statistics.mean(reference),candidate_mean=statistics.mean(candidate),
        ratio_status='DEFINED' if statistics.mean(reference) else 'UNDEFINED_ZERO_REFERENCE_MEAN',
        ratio_CI_status='NOT_ESTIMATED_SINGLE_START' if n<2 else 'PENDING')
    if n<2:
        return result
    values=[];zero=0
    for indices in draws(n):
        a=statistics.mean(reference[i] for i in indices)
        b=statistics.mean(candidate[i] for i in indices)
        if a==0:
            zero+=1
        else:
            values.append(b/a)
    result['zero_denominator_draws']=zero
    result['ratio_CI_status']='UNDEFINED_ZERO_DENOMINATOR_DRAWS' if zero else 'ESTIMATED'
    if not zero:
        result['ratio_CI95']=[percentile(values,.025),percentile(values,.975)]
    return result


def collect():
    rows=[];status=[]
    for path in sorted((ROOT/'configs').glob('heldout_*.json')):
        cfg=json.loads(path.read_text())
        if not cfg.get('phase','').startswith('FROZEN_HELDOUT'):
            continue
        receipt=ROOT/'raw'/cfg['name']/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            status.append(dict(name=cfg['name'],status='FAILED' if receipt.exists() else 'NOT_RUN'))
            continue
        value=metrics(cfg['name']);assert value['output_gate']['passed']
        row=dict(name=cfg['name'],method=cfg['confirmation_method'],cap=cfg['cap'],
            QPS=cfg['arrival_rate'],seed=cfg['arrival_seed'],count=cfg['count'],
            dataset=cfg['dataset'],actual_cuts=value['actual_cut_histogram'],
            source_groups=value['source_groups'],destination_groups=value['destination_groups'],
            evidence='MEASURED',measurement='FINITE_COHORT',
            scheduled_trace_QPS=value['scheduled_trace_QPS'],realized_arrival_QPS=value['realized_arrival_QPS'])
        work=value['work_accounting']
        row['work_accounting']=work
        row['migrated_request_fraction']=work['completed_migrated_requests']/cfg['count'] if work else None
        row['offloaded_GEMM_row_fraction']=work['D_GEMM_token_layer_rows']/(
            work['P_GEMM_token_layer_rows']+work['D_GEMM_token_layer_rows']) if work else None
        row['geometry_counts']=value['batch_geometry_counts']
        row['accounting_scope']=value['accounting_scope']
        row.update({k:value[k] for k in KEYS if k in value})
        row.update({k:value['SLO_regimes']['Decode_priority'][k] for k in
                    ('Joint_attainment','Decode_attainment','Output_token_SLO_goodput')})
        row['regimes']=value['SLO_regimes']
        rows.append(row)
    return rows,status


def main():
    rows,status=collect();groups=collections.defaultdict(list)
    for row in rows:
        groups[(row['cap'],row['QPS'],row['method'])].append(row)
    aggregate=[]
    for (cap,qps,method),values in sorted(groups.items()):
        aggregate.append(dict(cap=cap,QPS=qps,method=method,starts=len(values),
            names=[r['name'] for r in values],seeds=[r['seed'] for r in values],
            metrics={key:dict(mean=statistics.mean(r[key] for r in values),
                CI95=mean_interval([r[key] for r in values])) for key in KEYS},
            SLO_regimes={regime:{key:dict(mean=statistics.mean(r['regimes'][regime][key] for r in values),
                CI95=mean_interval([r['regimes'][regime][key] for r in values])) for key in
                    ('TTFT_attainment','Decode_attainment','Joint_attainment','Output_token_SLO_goodput','Request_SLO_goodput')}
                for regime in ('TTFT_priority','Balanced','Decode_priority')},
            scope='Mean of per-start metrics. Tail percentiles are computed within each start. A single start has no resampling CI. Finite-cohort drain throughput is not sustainable capacity.'))
    comparisons=[]
    pairs=(('Native','Token'),('Native','Static'),('Native','Dynamic'),('Native','Colocated'),
        ('Token','Static'),('Token','Dynamic'),('Static','Dynamic'),
        ('Colocated','Token'),('Colocated','Static'),('Colocated','Dynamic'))
    for cap,qps in sorted({(r['cap'],r['QPS']) for r in rows}):
        for reference,candidate in pairs:
            a={r['seed']:r for r in groups.get((cap,qps,reference),[])}
            b={r['seed']:r for r in groups.get((cap,qps,candidate),[])}
            seeds=sorted(a.keys()&b.keys())
            if not seeds:
                continue
            for seed in seeds:
                ca=json.loads((ROOT/'configs'/(a[seed]['name']+'.json')).read_text())
                cb=json.loads((ROOT/'configs'/(b[seed]['name']+'.json')).read_text())
                assert all(ca[key]==cb[key] for key in ('arrival_offsets','request_caps','dataset','gpus'))
            values={key:ratio_interval([a[s][key] for s in seeds],[b[s][key] for s in seeds]) for key in KEYS}
            noninferior=(len(seeds)>=3 and all(values[k]['ratio_CI95'] is not None and
                values[k]['ratio_CI95'][1]<=1.05 for k in ('ITL_p99_ms','TPOT_p95_ms')))
            comparisons.append(dict(cap=cap,QPS=qps,reference=reference,candidate=candidate,
                paired_seeds=seeds,metrics=values,Decode_5pct_noninferiority=noninferior,
                SLO_regimes={regime:{key:ratio_interval(
                    [a[s]['regimes'][regime][key] for s in seeds],
                    [b[s]['regimes'][regime][key] for s in seeds]) for key in
                    ('TTFT_attainment','Decode_attainment','Joint_attainment','Output_token_SLO_goodput')}
                    for regime in ('TTFT_priority','Balanced','Decode_priority')},
                scope='Only comparisons against Native are the prespecified Native-relative gate. Other reference ratios are descriptive. Small-start bootstrap CIs have limited tail precision. No request/token pseudoreplication.'))
    write(ROOT/'analysis/heldout_start_statistics.json',dict(rows=rows,aggregate=aggregate,
        comparisons=comparisons,incomplete=status,confidence_level=.95,
        resampling_unit='Independent server start, paired by arrival seed',
        evidence='MEASURED',scope='Unseen controlled prompts and finite cohorts. No steady-state or sustainable-throughput estimate.'))
    flat=[]
    for r in rows:
        flat.append({key:r[key] for key in ('name','method','cap','QPS','seed','count',*KEYS)})
    target=ROOT/'analysis/heldout_per_start_metrics.csv'
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]) if flat else [])
        writer.writeheader();writer.writerows(flat)
    print('INDEPENDENT_START_ROWS',len(rows),'AGGREGATES',len(aggregate),'PAIRED_COMPARISONS',len(comparisons),flush=True)


if __name__=='__main__':
    main()
