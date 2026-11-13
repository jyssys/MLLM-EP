"""Preserve all cell outcomes and keep held-out comparisons paired."""
import collections,csv,statistics
from .common import *
from .analyze import metrics,compare,records,percentile,completed_D_layers
from .statistics import paired_ratio,paired_difference


KEYS=('TTFT_p50_ms','TTFT_p95_ms','TTFT_p99_ms','ITL_p50_ms','ITL_p95_ms','ITL_p99_ms',
    'maximum_ITL_ms','first_ITL_p95_ms','steady_ITL_p99_ms','TPOT_p50_ms','TPOT_p95_ms',
    'E2E_p50_ms','E2E_p95_ms','E2E_p99_ms','TPOT_p99_ms','request_max_ITL_p95_ms',
    'completed_QPS','output_tokens_s','scheduled_trace_QPS','realized_arrival_QPS','actual_compute_submission_QPS',
    'worker_emission_ITL_p99_ms','coordinator_delivery_lag_p95_ms')


def cells():
    rows=[]
    for p in sorted((ROOT/'configs').glob('*.json')):
        c=json.loads(p.read_text())
        if 'name' not in c:continue
        receipt=ROOT/'raw'/c['name']/'dispatcher_result.json'
        if not receipt.exists():continue
        r=json.loads(receipt.read_text());m=metrics(c['name']) if r['complete'] and c.get('module')!='capacity' else {}
        ref=f'coarse_O{c.get("cap")}_N64_q4.777_native16_s1_v1'
        output=None
        if m.get('complete') and c.get('dataset')=='discovery.json':
            output=compare(ref,c['name'],c['count']!=64)
        if not r['complete']:level='FAILED'
        elif output is not None and not output['passed']:level='FAILED_OUTPUT_GATE'
        elif c.get('module')=='capacity':level='MEASURED_LOCAL_CAPACITY'
        elif c.get('correctness_only') or c.get('numerical_native_reference') or c['name'].startswith('qual_') or 'qualification' in c['name']:
            level='CORRECTNESS_ONLY'
        elif c.get('physical_activity'):level='DIAGNOSTIC_KERNEL_CAPTURE'
        elif c.get('execution_events'):level='DIAGNOSTIC_EVENT_ENCLOSURE'
        elif c.get('calibration_serial'):level='MEASURED_SERIAL_CALIBRATION'
        elif c.get('cap')==2:level='DIAGNOSTIC_PRIMITIVE_CALIBRATION'
        else:level='MEASURED_MARKER_FREE_SERVING'
        rows.append(dict(name=c['name'],arm=c.get('evaluation_arm',c['arm']),phase=c['phase'],
            count=c['count'],input=c['length'],output=c['cap'],offered_QPS=c.get('arrival_rate'),
            dataset=c['dataset'],arrival_seed=c['arrival_seed'],complete=r['complete'],
            status=r['status'],evidence_class=level,output_gate=output,
            config_SHA256=digest(p),source_archive_SHA256=digest(receipt.parent/'source.zip'),metrics=m))
    write(ROOT/'analysis/all_cell_results.json',dict(results=rows,
        scope='Runtime completion does not imply correctness, a clean profiler, or sustainable capacity. Every failure remains in its raw namespace.'))
    csvpath=ROOT/'analysis/all_serving_cells.csv'
    with csvpath.open('w') as stream:
        fields=['name','arm','phase','evidence_class','count','input','output','offered_QPS','dataset','arrival_seed','complete','output_pass',*KEYS]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            if not r['metrics'].get('complete'):continue
            writer.writerow({**{k:r[k] for k in fields if k in r},
                'output_pass':r['output_gate']['passed'] if r['output_gate'] is not None else '',
                **{k:r['metrics'].get(k) for k in KEYS}})
    return rows


def confirmation(source_path=None,result_path=None,caps=None):
    path=Path(source_path) if source_path else ROOT/'analysis/heldout_results.json'
    if not path.exists():return
    data=json.loads(path.read_text())['results'];groups=collections.defaultdict(dict)
    extra=ROOT/'analysis/exploratory_fraction50_results.json'
    if extra.exists():data+=json.loads(extra.read_text())['results']
    for r in data:
        if caps is not None and r['cap'] not in caps:continue
        r['metrics']=metrics(r['name'])
        groups[(r['cap'],r['arm'])][r['arrival_seed']]=r
    reports=[]
    for (cap,arm),by_seed in sorted(groups.items()):
        seeds=sorted(by_seed);native=groups[(cap,'Native')]
        assert set(seeds)==set(native) and len(seeds)>=3
        ms=[by_seed[s]['metrics'] for s in seeds]
        metric_keys=KEYS+('P_source_production_span_s','P_clearance_after_last_arrival_s','finite_P_source_ready_rate') if arm!='Colocated' else KEYS
        summary={k:dict(mean=statistics.mean(m[k] for m in ms),minimum=min(m[k] for m in ms),
            maximum=max(m[k] for m in ms)) for k in metric_keys}
        ratios={k:paired_ratio([by_seed[s]['metrics'][k] for s in seeds],[native[s]['metrics'][k] for s in seeds])
            for k in metric_keys if k!='P_clearance_after_last_arrival_s'}
        regimes={}
        for label in ms[0]['SLO_regimes']:
            regimes[label]={}
            for key in ('TTFT_attainment','Decode_attainment','Joint_attainment','Request_SLO_goodput','Output_token_SLO_goodput'):
                a=[by_seed[s]['metrics']['SLO_regimes'][label][key] for s in seeds]
                b=[native[s]['metrics']['SLO_regimes'][label][key] for s in seeds]
                regimes[label][key]=dict(mean=statistics.mean(a),difference_vs_Native=paired_difference(a,b),
                    ratio_vs_Native=paired_ratio(a,b) if sum(b)>0 else None)
        distribution=collections.Counter();decisions=collections.Counter();layers=0;migrated=0
        for r in by_seed.values():
            rows=records(ROOT/'raw'/r['name']/'requests.jsonl')
            cfg=json.loads((ROOT/'configs'/(r['name']+'.json')).read_text())
            for req in rows:
                n=completed_D_layers(req,cfg);distribution[n]+=1;layers+=n;migrated+=n>0
            for d in records(ROOT/'raw'/r['name']/'online_decisions.jsonl'):decisions[d['choice']['reason']]+=1
        comparisons={}
        for baseline in ('Colocated','Static','MatchedFraction','FixedFraction50'):
            if (cap,baseline) not in groups:continue
            other=groups[(cap,baseline)]
            comparisons[baseline]={k:paired_ratio([by_seed[s]['metrics'][k] for s in seeds],
                [other[s]['metrics'][k] for s in seeds]) for k in KEYS}
            comparisons[baseline]['SLO_regimes']={}
            for label in ms[0]['SLO_regimes']:
                comparisons[baseline]['SLO_regimes'][label]={}
                for k in ('TTFT_attainment','Decode_attainment','Joint_attainment','Request_SLO_goodput','Output_token_SLO_goodput'):
                    a=[by_seed[s]['metrics']['SLO_regimes'][label][k] for s in seeds]
                    b=[other[s]['metrics']['SLO_regimes'][label][k] for s in seeds]
                    comparisons[baseline]['SLO_regimes'][label][k]=dict(difference=paired_difference(a,b),
                        ratio=paired_ratio(a,b) if sum(b)>0 else None)
        relative_pass=all(ratios[k]['paired_bootstrap_95_percentile_CI'][1]<=1.05
            for k in ('ITL_p99_ms','TPOT_p95_ms'))
        absolute_pass=all(m['SLO_regimes']['Balanced']['percentile_Decode_absolute_pass'] for m in ms)
        ttft_benefit=ratios['TTFT_p95_ms']['paired_bootstrap_95_percentile_CI'][1]<1
        reports.append(dict(cap=cap,arm=arm,seeds=seeds,independent_starts=len(seeds),exploratory=arm=='FixedFraction50',
            output_pass=all(r['output_gate']['passed'] for r in by_seed.values()),metrics=summary,
            ratios_vs_Native=ratios,ratios_vs_other_baselines=comparisons,SLO_regimes=regimes,
            completed_layer_distribution=dict(distribution),migrated_requests=migrated,completed_migrated_layers=layers,
            total_requests=sum(distribution.values()),total_prefill_layers=36*sum(distribution.values()),
            decision_reasons=dict(decisions),relative_Decode_CI_pass=relative_pass,
            Balanced_absolute_Decode_pass=absolute_pass,TTFT_improvement_CI_pass=ttft_benefit))
    destination=Path(result_path) if result_path else ROOT/'analysis/heldout_paired_statistics.json'
    write(destination,dict(results=reports,
        scope='Paired ratios use means of independent start percentiles. No pooled token percentile substitutes for a start. Report the actual start count for each workload.'))
    return reports


def main():
    all_rows=cells();reports=confirmation()
    print('AGGREGATE',len(all_rows),'heldout_groups',len(reports or []))


if __name__=='__main__':main()
