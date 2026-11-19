"""Export measured boundaries and finite-window queue diagnostics."""
import csv
import hashlib
import json
import statistics
from .common import ROOT,MODEL_REVISION,write,digest
from .historical_audit import records


def quantile(values,p):
    ordered=sorted(values)
    if not ordered:
        return None
    at=(len(ordered)-1)*p
    lo=int(at);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)


def queue_diagnostic(rows):
    """Count all arrived requests, including requests completed during Drain."""
    arrivals=sorted(r['arrival'] for r in rows)
    finishes=sorted(r['finish'] for r in rows)
    start,end=arrivals[0],arrivals[-1]
    span=end-start
    if span<=0:
        return dict(status='NOT_APPLICABLE_SERIAL_OR_ZERO_SPAN')
    samples=[]
    for i in range(21):
        now=start+span*i/20
        arrived=sum(t<=now for t in arrivals)
        completed=sum(t<=now for t in finishes)
        samples.append(dict(seconds=now-start,arrived=arrived,completed=completed,
            unfinished=arrived-completed))
    # Central arrival-window halves avoid silently discarding Drain requests.
    middle=[r for r in samples if .2*span<=r['seconds']<=.8*span]
    x=[r['seconds'] for r in middle];y=[r['unfinished'] for r in middle]
    xm,ym=statistics.mean(x),statistics.mean(y)
    denom=sum((v-xm)**2 for v in x)
    slope=sum((a-xm)*(b-ym) for a,b in zip(x,y))/denom if denom else None
    completed_in_arrival=sum(start<=t<=end for t in finishes)
    status=('OVERLOAD_DIAGNOSTIC' if slope is not None and slope>.1 and
        samples[-1]['unfinished']>max(4,.2*len(rows)) else 'FINITE_WINDOW_NO_STRONG_GROWTH')
    return dict(status=status,arrival_window_seconds=span,
        completed_during_arrivals=completed_in_arrival,
        arrival_window_completed_QPS=completed_in_arrival/span,
        arrival_window_arrival_QPS=(len(rows)-1)/span,
        unfinished_at_last_arrival=samples[-1]['unfinished'],
        unfinished_middle_window_slope_requests_s=slope,
        final_Drain_seconds=max(finishes)-end,samples=samples,
        scope='A finite-window diagnostic. No status establishes sustainable stability. The 0.1 request/s and 20% backlog flags are descriptive, not a predeclared capacity test. Every request remains in the final latency and throughput denominator.')


def export(name):
    cfgpath=ROOT/'configs'/(name+'.json')
    cfg=json.loads(cfgpath.read_text())
    metricpath=ROOT/'analysis'/(name+'_metrics.json')
    metric=json.loads(metricpath.read_text())
    folder=ROOT/'raw'/name
    rows=records(folder/'requests.jsonl')
    host=json.loads((ROOT/'analysis'/(name+'_host_paths.json')).read_text())
    boundaries={r['request_id']:r['boundaries_monotonic'] for r in host['request_paths']}
    limits=metric['SLO_regimes']['Common_Balanced_OutputSweep']['thresholds']
    request_export=[]
    for row in rows:
        events=row['token_events']
        intervals=[1000*(b['time']-a['time']) for a,b in zip(events,events[1:])]
        passes=dict(TTFT=row['ttft_ms']<=limits['TTFT_ms'],
            ITL=bool(intervals) and max(intervals)<=limits['ITL_ms'],
            TPOT=row['tpot_ms']<=limits['TPOT_ms'])
        rid='q'+str(row['sample_id'])
        token_bytes=json.dumps(row['input_ids'],separators=(',',':')).encode()
        request_export.append(dict(request_id=rid,prompt_id=row['prompt_id'],
            input_IDs_SHA256=hashlib.sha256(token_bytes).hexdigest(),input_tokens=row['input_tokens'],
            output_tokens=row['output_tokens'],TTFT_ms=row['ttft_ms'],TPOT_ms=row['tpot_ms'],
            E2E_ms=row['e2e_ms'],first_ITL_ms=intervals[0] if intervals else None,
            steady_ITL_p99_ms=quantile(intervals[1:],.99),maximum_ITL_ms=max(intervals) if intervals else None,
            SLO_passes=passes,joint_pass=all(passes.values()),actual_layer_cut=row.get('actual_layer_cut'),
            host_boundaries_monotonic=boundaries[rid],
            first_worker_output=events[0]['worker_time'],last_worker_output=events[-1]['worker_time'],
            first_client_output=events[0]['time'],last_client_output=events[-1]['time']))
    observed_span=metric['observation_seconds']
    assert sum(r['output_tokens'] for r in rows)/observed_span==metric['output_tokens_s']
    qualified=sum(r['output_tokens'] for r in request_export if r['joint_pass'])/observed_span
    assert abs(qualified-metric['SLO_regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput'])<1e-8
    worker_options={}
    for role in ('prefill','decode'):
        ready=json.loads((folder/(role+'_ready.json')).read_text())
        opts=ready['options']
        worker_options[role]=dict(UUID=ready['uuid'],KV_blocks=ready['KV_blocks'],
            KV_tensor_bytes=ready['KV_tensor_bytes'],weight_bytes=ready['weight_bytes'],
            actual_options={k:opts[k] for k in ('max_model_len','max_num_seqs','max_num_batched_tokens',
                'enable_prefix_caching','enable_chunked_prefill','async_scheduling','compilation_config')},
            actual_measured_geometry=json.loads((folder/('measured_geometry_'+role+'.json')).read_text()),
            effective_runtime_snapshot_SHA256=digest(folder/('effective_runtime_'+role+'.json')))
    keys=('offered_QPS','scheduled_trace_QPS','realized_arrival_QPS','completed_QPS','observation_seconds',
        'mean_TTFT_ms','TTFT_p50_ms','TTFT_p95_ms','TTFT_p99_ms','mean_TBT_ms','ITL_p50_ms',
        'ITL_p95_ms','ITL_p99_ms','maximum_ITL_ms','TPOT_p50_ms','TPOT_p95_ms','TPOT_p99_ms',
        'first_ITL_p95_ms','steady_ITL_p99_ms','E2E_p95_ms','output_tokens_s',
        'mean_worker_emission_TBT_ms','worker_emission_ITL_p99_ms',
        'coordinator_delivery_lag_p95_ms','parent_tail_interval_identity','coordinator_queue_observations',
        'native_observations','transport_audit','work_accounting','actual_cut_histogram')
    value=dict(name=name,method=cfg['evaluation_arm'],layers=cfg['static_layers'],
        output=cfg['cap'],count=cfg['count'],batch=cfg['D_admission_batch'],
        stage=cfg['matrix_stage'],seed=cfg['arrival_seed'],source_window=cfg['source_outstanding_limit'],
        metrics={k:metric.get(k,'NOT_INSTRUMENTED') for k in keys},
        common_Balanced=metric['SLO_regimes']['Common_Balanced_OutputSweep'],
        workers=worker_options,
        gates={k:metric.get(k,'NOT_APPLICABLE') for k in ('output_gate','exact_GEMM_row_gate',
            'unified_worker_gate','common_reservation_gate','actual_query_coverage_gate')},
        queue_diagnostic=queue_diagnostic(rows),requests=request_export,
        provenance=dict(model_revision=MODEL_REVISION,configuration_SHA256=digest(cfgpath),
            source_archive_SHA256=digest(folder/'source.zip'),request_log_SHA256=digest(folder/'requests.jsonl'),
            arrival_offsets_SHA256=hashlib.sha256(json.dumps(cfg['arrival_offsets'],separators=(',',':')).encode()).hexdigest()),
        evidence='MEASURED_ONLINE',
        scope='Client-visible finite-cohort metrics. First ITL is included. Worker clocks are separate diagnostic boundaries. Source ZIP and dispatcher logs retain the exact command, runtime and GPU mapping. Full state witnesses are separate correctness runs. No GPU busy, exclusive kernel or sustainable capacity claim follows.')
    write(ROOT/'analysis'/(name+'_measurement_export.json'),value)
    return value


def main():
    data=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    values=[export(r['name']) for r in data['starts']]
    write(ROOT/'analysis/measurement_export_index.json',dict(names=[r['name'] for r in values],
        queue_status_counts={key:sum(r['queue_diagnostic']['status']==key for r in values)
            for key in sorted({r['queue_diagnostic']['status'] for r in values})},
        evidence='MEASURED_ONLINE',scope='New clean starts only. Observer, failed and correctness-only runs are inventoried separately.'))
    path=ROOT/'analysis/per_start_client_metrics.csv'
    fieldnames=['name','method','layers','output','count','batch','stage','seed','source_window',
        'offered_QPS','scheduled_trace_QPS','realized_arrival_QPS','completed_QPS','observation_seconds',
        'mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms',
        'output_tokens_s','Joint_attainment','Output_token_SLO_goodput','queue_status']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader()
        for row in values:
            writer.writerow({**{k:row[k] for k in fieldnames[:9]},
                **{k:row['metrics'][k] for k in fieldnames[9:-3]},
                **{k:row['common_Balanced'][k] for k in ('Joint_attainment','Output_token_SLO_goodput')},
                'queue_status':row['queue_diagnostic']['status']})
    print('MEASUREMENT_EXPORTS',len(values),flush=True)


if __name__=='__main__':
    main()
