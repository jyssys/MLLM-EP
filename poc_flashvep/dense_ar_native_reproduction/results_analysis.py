"""Aggregate qualified Main starts without converting finite cohorts to capacity."""
import collections
import csv
import itertools
import json
import numpy as np
from .common import ROOT,write
from .figures import start_interval
from .metrics import summarize
from .qualification import records
from .serving_analysis import analyze


FIELDS=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','mean_TBT_ms','TPOT_p95_ms',
    'generated_output_tokens_per_second','completed_requests_per_second',
    'Joint_SLO_attainment','SLO_qualified_output_tokens_per_second','max_ITL_ms',
    'actual_arrival_QPS','measurement_seconds','TTFT_SLO_attainment','every_ITL_SLO_attainment',
    'TPOT_SLO_attainment','decode_SLO_attainment')


def backlog_trace(name):
    folder=ROOT/'raw'/name;window=json.loads((folder/'window.json').read_text())
    values=records(folder/'coordinator_queues.jsonl')
    left=window['start']+.2*(window['last_arrival']-window['start'])
    right=window['start']+.8*(window['last_arrival']-window['start'])
    part=[r for r in values if left<=r['time']<=right]
    if len(part)<2:return dict(status='NOT_MEASURED')
    times=np.array([r['time']-left for r in part]);counts=np.array([r['arrived']-r['completed'] for r in part])
    last=min(values,key=lambda r:abs(r['time']-window['last_arrival']))
    slope=float(np.polyfit(times,counts,1)[0])
    return dict(fitted_outstanding_slope_requests_per_second=slope,
        outstanding_mean=float(np.mean(counts)),outstanding_last_arrival=last['arrived']-last['completed'],
        arrival_interval_relative_seconds=[left-window['start'],right-window['start']],
        scope='Linear descriptive trend over the central 20%-80% arrival interval. Outstanding includes active work. This finite cohort does not establish long-run queue stability.')


def main():
    per_start=[];runtime=[]
    for folder in sorted((ROOT/'raw').glob('Main_*_LongBench_N200_*')):
        receipt=folder/'dispatcher_result.json';gate=ROOT/'analysis'/(folder.name+'_output_gate.json')
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
        if not gate.exists() or not json.loads(gate.read_text())['passed']:continue
        cfg=json.loads((folder/'configuration.json').read_text())
        metrics=summarize(folder.name);trace=analyze(folder.name)
        command=json.loads((folder/'command.json').read_text())
        row=dict(name=folder.name,method=cfg['method'],offered_QPS=cfg['offered_QPS'],
            arrival_seed=cfg['arrival_seed'],metrics=metrics,backlog=backlog_trace(folder.name),
            timeline=trace,source_archive_SHA256=command['source_archive_SHA256'],
            dataset_SHA256=cfg['dataset_SHA256'])
        per_start.append(row)
        for file in folder.glob('actual_runtime_*.json'):
            actual=json.loads(file.read_text())
            runtime.append(dict(name=folder.name,role=actual['role'],B=actual['scheduler_B'],S=actual['scheduler_S'],
                chunked_prefill=actual['chunked_prefill'],KV_bytes=actual['KV_bytes'],KV_blocks=actual['KV_blocks'],
                UUID=actual['uuid'],model_layers=actual['model_layers'],parameter_dtypes=actual['parameter_dtypes'],
                graph_mode=actual['cudagraph_mode'],numerical_family=actual['batch_invariant']))
    cells=[];pairs=[]
    for rate in sorted({r['offered_QPS'] for r in per_start}):
        for method in ('Native','Agg'):
            part=[r for r in per_start if r['offered_QPS']==rate and r['method']==method]
            if not part:continue
            cells.append(dict(QPS=rate,method=method,independent_starts=len(part),
                arrival_seeds=[r['arrival_seed'] for r in part],
                metrics={key:dict(mean=float(np.mean([r['metrics'][key] for r in part])),
                    CI95=start_interval([r['metrics'][key] for r in part])) for key in FIELDS}))
        by_method={method:{r['arrival_seed']:r for r in per_start if r['offered_QPS']==rate and r['method']==method} for method in ('Native','Agg')}
        seeds=sorted(set(by_method['Native'])&set(by_method['Agg']))
        if not seeds:continue
        checks=[]
        for seed in seeds:
            a,b=[ROOT/'raw'/by_method[m][seed]['name'] for m in ('Native','Agg')]
            configs=[json.loads((p/'configuration.json').read_text()) for p in (a,b)]
            requests=[{r['prompt_id']:r for r in records(p/'requests.jsonl')} for p in (a,b)]
            sources=[json.loads((p/'source_hashes.json').read_text()) for p in (a,b)]
            frozen=json.loads((ROOT/'state/runtime_source_freeze.json').read_text())['files']
            different=[file for file in frozen if sources[0].get('poc_flashvep/dense_ar_native_reproduction/'+file)!=sources[1].get('poc_flashvep/dense_ar_native_reproduction/'+file)]
            checks.append(dict(seed=seed,arrival_offsets_equal=configs[0]['arrival_offsets']==configs[1]['arrival_offsets'],
                dataset_equal=configs[0]['dataset_SHA256']==configs[1]['dataset_SHA256'],
                runtime_files_equal=not different,different_runtime_source_files=different,
                outputs_equal=all(requests[0][pid]['output_ids']==requests[1][pid]['output_ids'] for pid in requests[0]),
                output_tokens_equal=sum(r['output_tokens'] for r in requests[0].values())==sum(r['output_tokens'] for r in requests[1].values())))
        ratios={}
        for field in FIELDS[:9]:
            a=[by_method['Native'][s]['metrics'][field] for s in seeds]
            b=[by_method['Agg'][s]['metrics'][field] for s in seeds]
            mean_ratio=float(np.mean(a)/np.mean(b)) if np.mean(b)>0 else None
            draws=[]
            if len(seeds)>1:
                for draw in itertools.product(range(len(seeds)),repeat=len(seeds)):
                    denom=np.mean([b[i] for i in draw])
                    if denom>0:draws.append(float(np.mean([a[i] for i in draw])/denom))
            ratios[field]=dict(Native_over_Agg=mean_ratio,
                paired_CI95=[float(np.percentile(draws,p)) for p in (2.5,97.5)] if len(draws)==len(seeds)**len(seeds) and len(seeds)>1 else None)
        pairs.append(dict(QPS=rate,paired_server_starts=len(seeds),checks=checks,ratios=ratios))
    write(ROOT/'analysis/Main_runtime_audit.json',dict(rows=runtime,
        all_B8192_S1024=bool(runtime) and all((r['B'],r['S'])==(8192,1024) for r in runtime),
        scope='Actual worker configurations. Scheduler maxima are not actual GPU batches.'))
    write(ROOT/'analysis/Main_full_analysis.json',dict(per_start=per_start,cells=cells,paired_comparison=pairs,
        evidence='MEASURED_ONLINE_AND_TRACE_ANALYSIS',complete=(ROOT/'state/main_common_sweep_complete.json').exists(),
        scope='Start-level statistics. The paired bootstrap resamples matched arrival-seed server-start pairs. n=1 has no CI. No sustainable-capacity or physical-idle inference.'))
    with (ROOT/'analysis/Main_table.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['method','offered_QPS','independent_starts']+
            [column for field in FIELDS for column in (field,field+'_CI95_low',field+'_CI95_high')])
        for cell in cells:
            row=[cell['method'],cell['QPS'],cell['independent_starts']]
            for field in FIELDS:
                value=cell['metrics'][field];row.extend([value['mean'],*(value['CI95'] or [None,None])])
            writer.writerow(row)
    task_fields=('TTFT_p95_ms','TTFT_mean_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms',
        'output_tokens','Joint_SLO_attainment','output_tokens_per_global_window_second')
    with (ROOT/'analysis/Main_task_per_start_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['name','method','offered_QPS','arrival_seed','task','requests',*task_fields,
            'TTFT_pass_fraction','every_ITL_pass_fraction','TPOT_pass_fraction','decode_pass_fraction'])
        for start in per_start:
            for task,values in start['metrics']['by_task'].items():
                writer.writerow([start['name'],start['method'],start['offered_QPS'],start['arrival_seed'],task,values['n'],
                    *(values[k] for k in task_fields),*(values['SLO_components'][k] for k in ('TTFT_pass','every_ITL_pass','TPOT_pass','decode_pass'))])
    print(json.dumps(dict(completed_starts=len(per_start),cells=len(cells),paired_cells=len(pairs))))


if __name__=='__main__':main()
