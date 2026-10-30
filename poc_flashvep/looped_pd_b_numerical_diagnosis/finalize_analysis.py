"""CPU-only tables. Keep host phases, GPU activity, and clean latency separate."""
import csv
import json
import math
from statistics import mean,stdev
import numpy as np
from .common import ROOT,write
from .slowdown import summarize
from .summarize import rows,comparison


def csv_write(name,records):
    if not records:return
    fields=list(dict.fromkeys(key for record in records for key in record))
    with (ROOT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in records)


def completed(name):
    path=ROOT/'raw'/name/'status.json'
    return path.exists() and json.loads(path.read_text()).get('status')=='COMPLETE'


def mean_ci(values):
    n=len(values)
    if n<2:return [None,None]
    # Three independent server starts. Use Student t, not a token-level resample.
    critical={2:12.7062047364,3:4.30265272991}.get(n)
    if critical is None:
        from scipy.stats import t
        critical=float(t.ppf(.975,n-1))
    half=critical*stdev(values)/math.sqrt(n)
    return [mean(values)-half,mean(values)+half]


def main():
    clean=[];phases=[];shares=[];contrast=[];host=[];gpu=[];order=[];kv=[];teacher=[];isolation=[]
    plan_path=ROOT/'configs/quiet_plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else []
    generic=ROOT/'configs/generic_queued_plan.json'
    if generic.exists():plan+=json.loads(generic.read_text())
    for job in plan:
        name=job['name'];folder=ROOT/'raw'/name
        if not completed(name):continue
        data,summary,details=summarize(folder)
        config=json.loads((folder/'serving_contract.json').read_text())
        if 'independent_start' in job:
            item=dict(model=job['model'],method=job['method_label'],repeat=job['independent_start'],
                 run=name,N=len(data),GPUs=job['gpus'],input_tokens=sum(r['input_tokens'] for r in data.values()),
                 output_tokens=sum(r['output_tokens'] for r in data.values()),
                 truncated=sum(r['truncated'] for r in data.values()),evidence='LIVE_MEASURED_FINITE_COHORT',
                 timing_scope='Scheduled harness arrival to final output. N8, safety cap8. Not sustainable capacity.')
            for field,label in [('ttft_ms','TTFT'),('tpot_ms','TPOT'),('e2e_ms','E2E')]:
                values=[r[field] for r in data.values() if r.get(field) is not None]
                item[label+'_mean_ms']=mean(values)
                for pct in (50,95,99):item[label+f'_p{pct}_ms']=float(np.percentile(values,pct))
            item['cohort_makespan_ms']=1000*(max(r['finish'] for r in data.values())-min(r['arrival'] for r in data.values()))
            item.update(summary['counters'])
            item['exposed_handoff_mean_ms']=summary['host_phases']['exposed_handoff_ms']['mean']
            clean.append(item)
            phases.extend(dict(model=job['model'],method=job['method_label'],repeat=job['independent_start'],run=name,**r) for r in details)
            terms=['harness_submission_ms','coordinator_queue_ms','reservation_ms','prefill_with_backpressure_ms',
                   'exposed_handoff_ms','resume_first_token_ms','harness_first_token_observation_ms']
            # Use the observed first token as the shared boundary. Do not count its delivery twice.
            residuals=[]
            for r in details:
                residual=r['E2E_ms']-r['TTFT_ms']
                assert abs(sum(r[t] for t in terms)+residual-r['E2E_ms'])<1e-4
                residuals.append(residual)
            e2e=mean(r['E2E_ms'] for r in details)
            for term in terms+['post_first_token_wall_time_ms']:
                value=mean(residuals) if term=='post_first_token_wall_time_ms' else mean(r[term] for r in details)
                shares.append(dict(model=job['model'],method=job['method_label'],repeat=job['independent_start'],
                     run=name,N=len(details),component=term,mean_ms=value,percent_of_mean_request_E2E=100*value/e2e,
                     total_mean_E2E_ms=e2e,evidence='TRACE_ANALYSIS_OF_MEASURED_HOST_PHASES',
                     scope='Serial request-time partition. Prefill includes backpressure. Not exclusive GPU kernel or DMA shares.'))
        samples=folder/'fabric_process_samples.jsonl'
        if samples.exists():
            records=[json.loads(line) for line in samples.read_text().splitlines()]
            window=json.loads((folder/'window.json').read_text())
            selected=[r for r in records if window['start']<=r['time']<=window['end']]
            outside=[p for row in selected for p in row['processes'] if p['outside_assigned_pair']]
            isolation.append(dict(run=name,model=job['model'],GPUs=job['gpus'],whole_process_samples=len(records),
                  measured_window_samples=len(selected),outside_pair_process_observations=len(outside),
                  evidence='MEASURED_LOW_RATE_NVML_SAMPLES',scope='One-second process samples. Subsecond processes are not excluded.'))
        for path in folder.glob('transfer_host_probe_*.json'):
            record=json.loads(path.read_text());host.append(dict(run=name,model=job['model'],method=job['method_label'],
                 pid=int(path.stem.split('_')[-1]),**record['totals'],scope=record['scope']))
        path=folder/'timeline_summary.json'
        if path.exists():
            a=json.loads(path.read_text());gpu.append(a)
    for model in ('ouro','huginn'):
        group=[r for r in clean if r['model']==model]
        for method in ('B','B_queued','PD_G1_queued'):
            paired=[]
            for repeat in range(3):
                a=next((r for r in group if r['method']=='PD_G1' and r['repeat']==repeat),None)
                b=next((r for r in group if r['method']==method and r['repeat']==repeat),None)
                if a and b:paired.append((a,b))
            if not paired:continue
            checks=[comparison(a['run'],b['run']) for a,b in paired]
            for metric in ('TTFT_mean_ms','TTFT_p95_ms','TPOT_mean_ms','E2E_mean_ms','E2E_p95_ms','cohort_makespan_ms','exposed_handoff_mean_ms'):
                deltas=[b[metric]-a[metric] for a,b in paired];ratios=[a[metric]/b[metric] for a,b in paired]
                contrast.append(dict(model=model,method=method,baseline='PD_G1',metric=metric,
                    independent_paired_starts=len(paired),control_mean=mean(a[metric] for a,b in paired),
                    method_mean=mean(b[metric] for a,b in paired),mean_delta=mean(deltas),
                    mean_delta_t95=mean_ci(deltas),mean_paired_ratio=mean(ratios),ratio_t95=mean_ci(ratios),
                    output_agreement=sum(x['exact'] for x in checks),comparison_N=sum(x['N'] for x in checks),
                    evidence='LIVE_MEASURED_FINITE_COHORT',
                    CI_scope='Paired independent-start Student-t interval. Three starts give weak precision. Not a capacity result.'))
        for metric in ('TTFT_p95_ms','E2E_mean_ms','cohort_makespan_ms'):
            pairs=[(a,b) for a in group for b in group if a['method']=='PD_G1_queued' and b['method']=='B_queued' and a['repeat']==b['repeat']]
            if pairs:
                values=[a[metric]/b[metric] for a,b in pairs]
                checks=[comparison(a['run'],b['run']) for a,b in pairs]
                contrast.append(dict(model=model,method='B_queued',baseline='PD_G1_queued',metric=metric,
                    independent_paired_starts=len(pairs),control_mean=mean(a[metric] for a,b in pairs),
                    method_mean=mean(b[metric] for a,b in pairs),mean_paired_ratio=mean(values),ratio_t95=mean_ci(values),
                    output_agreement=sum(x['exact'] for x in checks),comparison_N=sum(x['N'] for x in checks),
                    evidence='LIVE_MEASURED_FINITE_COHORT',CI_scope='Separate finite-control campaign. Same pairs and contract, but controls ran later. No capacity claim.'))
        baseline=f'{model}_down_norm_qual_N8_seed2026103001_PD_G1_0'
        a=rows(ROOT/'raw'/baseline)
        for method,index in [('PD_G1',0),('B',1)]:
            name=f'{model}_order_reversed8_capacity3_{method}_{index}'
            b={r['source_sample_id']:r for r in rows(ROOT/'raw'/name).values()}
            ids=sorted(a.keys()&b.keys())
            order.append(dict(model=model,method=method,run=name,N=len(ids),baseline=baseline,
                seed_exact=sum(a[i]['seed']==b[i]['seed'] for i in ids),input_exact=sum(a[i]['input_ids']==b[i]['input_ids'] for i in ids),
                output_exact=sum(a[i]['output_ids']==b[i]['output_ids'] for i in ids),
                evidence='LIVE_MEASURED_DIFFERENT_ADMISSION_ORDER',capacity=3))
    adapter_reference='huginn_down_norm_adapter_qual_N8_seed2026103001_PD_G1_0'
    a=rows(ROOT/'raw'/adapter_reference)
    for method,index in [('PD_G1',0),('B',1)]:
        name=f'huginn_adapter_order8_capacity3_{method}_{index}'
        b={r['source_sample_id']:r for r in rows(ROOT/'raw'/name).values()}
        ids=sorted(a.keys()&b.keys())
        if ids:
            order.append(dict(model='huginn',variant='down_norm_adapter',method=method,run=name,N=len(ids),
                baseline=adapter_reference,seed_exact=sum(a[i]['seed']==b[i]['seed'] for i in ids),
                input_exact=sum(a[i]['input_ids']==b[i]['input_ids'] for i in ids),
                output_exact=sum(a[i]['output_ids']==b[i]['output_ids'] for i in ids),
                evidence='LIVE_MEASURED_DIFFERENT_ADMISSION_ORDER',capacity=3))
    for folder in sorted((ROOT/'raw').glob('teacher_*')):
        path=folder/'result.json'
        if not path.exists():continue
        value=json.loads(path.read_text())
        first=value.get('first_observed_difference',{})
        teacher.append(dict(run=folder.name,variant=value.get('variant'),sample=value.get('sample'),
                 outputs=value.get('outputs'),first_observed_difference=first,scope=value.get('scope'),
                 trace_count=len(value.get('comparisons',[]))))
        hashes=value.get('prefill_KV_hashes',[])
        for record in hashes:
            kv.append(dict(run=folder.name,rows=record['rows'],reference_rows=hashes[0]['rows'],
                 valid_layer_depth_pairs=len(record['entries']),KV_exact_to_reference=record['entries']==hashes[0]['entries'],
                 final_hidden_exact_to_reference=record['final_hidden_sha256']==hashes[0]['final_hidden_sha256'],
                 evidence='MEASURED_INSTRUMENTED_VALID_PREFIX_HASHES',scope=value['scope']))
    csv_write('quiet_latency_raw.csv',clean);csv_write('quiet_host_phase_raw.csv',phases)
    csv_write('e2e_component_shares.csv',shares);csv_write('quiet_paired_comparisons.csv',contrast)
    csv_write('transfer_host_probe_summary.csv',host);csv_write('quiet_isolation_audit.csv',isolation)
    csv_write('order_seed_parity.csv',order);csv_write('teacher_prefill_kv_identity.csv',kv)
    write(ROOT/'configs/teacher_results_summary.json',teacher)
    write(ROOT/'configs/GPU_timeline_summaries.json',gpu)
    if clean:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for axis,model in zip(axes,('ouro','huginn')):
            methods=['PD_G1','B','B_queued','PD_G1_queued']
            vals=[[r['cohort_makespan_ms'] for r in clean if r['model']==model and r['method']==method] for method in methods]
            for index,values in enumerate(vals):axis.scatter([index]*len(values),values)
            axis.set_xticks(range(len(methods)),methods,rotation=15);axis.set_ylabel('Finite cohort makespan (ms)')
            axis.set_title(model+' — N8, output safety cap8, independent starts')
        fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True)
        fig.savefig(ROOT/'figures/quiet_finite_cohort_makespan.svg')
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for axis,model in zip(axes,('ouro','huginn')):
            value=json.loads((ROOT/f'raw/{model}_norm_all_row_positions_v3/result.json').read_text())
            axis.plot([r['rows'] for r in value['rows']],
                      [r['stages']['output']['max_absolute'] for r in value['rows']],marker='o',label='Native versus M1')
            axis.plot([r['rows'] for r in value['rows']],
                      [r['fixed_all_rows_to_M16']['max_absolute'] for r in value['rows']],marker='x',label='Fixed versus M16')
            axis.set_xscale('log',base=2);axis.set_xlabel('Actual GEMM / normalization rows')
            axis.set_ylabel('Maximum absolute norm output difference')
            axis.set_title(model+' — one captured failure operand, 24 sizes');axis.legend()
        fig.tight_layout();fig.savefig(ROOT/'figures/captured_norm_geometry_error.svg')
    print(dict(clean_starts=len(clean),comparisons=len(contrast),order=order,teacher_runs=len(teacher),
               outside_pair_observations=sum(r['outside_pair_process_observations'] for r in isolation)))


if __name__=='__main__':main()
