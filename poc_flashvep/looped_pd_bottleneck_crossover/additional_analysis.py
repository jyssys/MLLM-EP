"""Preserve failures, report instrumentation costs, and rescore frozen SLOs."""
import collections
import statistics
from .common import *
from .summary_tables import records,percentile,save_csv
from .analyze_serving import analyze,token_intervals


def experiment_matrix():
    rows=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or not cfg.get('name'):continue
        folder=ROOT/'raw'/cfg['name'];status=folder/'status.json';result=folder/'dispatcher_result.json'
        raw=json.loads(status.read_text()) if status.exists() else {}
        exit_status=json.loads(result.read_text()) if result.exists() else {}
        state=raw.get('status','NOT_RUN')
        if state=='RUNNING' and exit_status.get('returncode',0):state='FAILED_STALE_RAW_STATUS'
        if state=='RUNNING' and raw.get('pid') and not Path(f'/proc/{raw["pid"]}').exists():state='INCOMPLETE_PROCESS_ABSENT'
        common=dict(name=cfg['name'],model=cfg['model'],phase=cfg.get('phase'),worker=cfg['worker'],
            gpus=cfg['gpus'],status=state,raw_status=raw.get('status'),returncode=exit_status.get('returncode'),
            error=raw.get('error'),configuration=str(path.relative_to(ROOT)),
            measured_requests=len([r for r in records(folder/'requests.jsonl') if not r.get('warmup')]))
        groups=cfg.get('groups',[])
        if groups:
            cohorts=records(folder/'cohorts.jsonl')
            for gi,g in enumerate(groups):
                for ji,job in enumerate(g['jobs']):
                    stem=f'g{gi}_j{ji}'
                    evidence=[r for r in cohorts if r.get('tag')==stem or r.get('tag','').startswith(stem+'_r')]
                    row=dict(common,cell=job.get('cell'),batch=job['batch'],output_cap=job['cap'],chunk=g['chunk'],
                        KV_blocks=g['kv_blocks'],measured_cohorts=len(evidence),
                        cohort_states=sorted({r.get('status') for r in evidence if r.get('status')}))
                    rows.append(row)
        else:rows.append(dict(common,cell=cfg.get('workload'),offered_requests=cfg.get('count'),chunk=cfg.get('chunk'),KV_blocks=cfg.get('kv_blocks')))
    save_csv(ROOT/'experiment_matrix.csv',rows)
    write(ROOT/'state/namespace_status_reconciliation.json',dict(time=time.time(),
        scope='Raw status files are not overwritten. A nonzero dispatcher result reconciles a stale RUNNING status.',
        failed=[r for r in rows if str(r['status']).startswith('FAILED')]))


def instrumentation():
    rows=[]
    for model in ('ouro','huginn'):
        folder=ROOT/'raw'/f'phase_a_{model}_event_overhead_start0'
        all_rows=records(folder/'cohorts.jsonl')
        for metric in ('prefill_host_service_s','decode_native_boundary_service_s'):
            off=[r[metric] for r in all_rows if not r['job']['instrument']]
            on=[r[metric] for r in all_rows if r['job']['instrument']]
            if not off or not on:continue
            a=statistics.median(off);b=statistics.median(on)
            rows.append(dict(model=model,metric=metric,off_median_s=a,on_median_s=b,
                on_over_off=b/a,relative_overhead_percent=100*(b/a-1),off_repeats=len(off),on_repeats=len(on),
                independent_starts=1,order='OFF_ON_OFF',evidence='MEASURED_INSTRUMENTATION_ABLATION_NOT_A_POPULATION_BOUND'))
    save_csv(ROOT/'instrumentation_overhead.csv',rows)
    rows=[]
    for path in (ROOT/'raw').glob('*/D_only_window.json'):
        cfg=json.loads((ROOT/'configs'/(path.parent.name+'.json')).read_text())
        rows.append(dict(name=path.parent.name,model=cfg['model'],cell=cfg.get('workload'),
            chunk=cfg.get('chunk'),KV_blocks=cfg.get('kv_blocks'),compute_slots=cfg.get('slots',cfg.get('capacity')),
            **json.loads(path.read_text())))
    save_csv(ROOT/'prepared_native_pd_decode_capacity.csv',rows)
    rows=[]
    for path in (ROOT/'raw').glob('*/projection_geometry_microbenchmark.json'):
        cfg=json.loads((ROOT/'configs'/(path.parent.name+'.json')).read_text())
        for row in json.loads(path.read_text()):
            rows.append(dict(name=path.parent.name,model=cfg['model'],**row,
                prepared_fixed_over_native=row['fixed_256_rows_prepared']/row['native_rows']))
    save_csv(ROOT/'projection_geometry_microbenchmark.csv',rows)


def sensitivity():
    slos={}
    for name in ('frozen_SLO.json','frozen_SLO_qwen.json'):
        path=ROOT/'configs'/name
        if path.exists():slos.update(json.loads(path.read_text()))
    out=[]
    for run in analyze():
        if not run['eligible_for_serving_comparison']:continue
        model=run['model'];slo=slos[model];folder=ROOT/'raw'/run['name']
        rows=[r for r in records(folder/'requests.jsonl') if not r.get('warmup')]
        for tf in (1.5,2.,4.):
            for pf in (1.5,2.,4.):
                ttft=slo['TTFT_ms']*tf/2;tpot=slo['TPOT_ms']*pf/2
                passed=sum(r['ttft_ms']<=ttft and r['tpot_ms']<=tpot for r in rows)
                out.append(dict(name=run['name'],model=model,cell=run['cell'],TTFT_ms=ttft,TPOT_ms=tpot,
                    TTFT_multiplier=tf,TPOT_multiplier=pf,primary=tf==pf==2.,passed=passed,N=run['offered_requests'],
                    joint_attainment=passed/run['offered_requests'],goodput_requests_s=passed/run['elapsed_s'],
                    evidence='TRACE_RESCORE_MEASURED_LATENCY_NOT_A_NEW_GPU_RUN'))
        for threshold in (50,100,150,200,250):
            intervals=[token_intervals(r) for r in rows]
            complete=sum(not bundled and all(v<=threshold for v in values) for values,bundled in intervals)
            population=[v for values,bundled in intervals for v in values]
            out.append(dict(name=run['name'],model=model,cell=run['cell'],absolute_ITL_ms=threshold,
                all_observed_ITL_pass=complete,N=run['offered_requests'],
                ambiguous_coalesced_requests=sum(bool(b) for v,b in intervals),population_single_token_ITL_p99_ms=percentile(population,.99),
                evidence='TRACE_RESCORE_MEASURED_SINGLE_TOKEN_DELIVERY_INTERVALS',
                scope='Coalesced delivery cannot establish every-token latency. Missing requests do not pass.'))
    save_csv(ROOT/'slo_sensitivity.csv',out)


def validate_shares():
    import csv
    path=ROOT/'e2e_component_shares.csv'
    if not path.exists():return
    totals=collections.defaultdict(float)
    for row in csv.DictReader(path.open()):totals[(row['name'],row['run'],row['sample_id'])]+=float(row['share_percent'])
    errors={str(k):v for k,v in totals.items() if abs(v-100)>1e-6}
    write(ROOT/'state/e2e_percentage_validation.json',dict(time=time.time(),requests_with_breakdown=len(totals),
        maximum_sum_error_percent=max((abs(v-100) for v in totals.values()),default=0.),errors=errors,
        scope='Host-exclusive request intervals. This is not a pure GPU or DMA decomposition.'))
    assert not errors


def main():
    experiment_matrix();instrumentation();sensitivity();validate_shares()
    print('ADDITIONAL_ANALYSIS_COMPLETE')


if __name__=='__main__':main()
