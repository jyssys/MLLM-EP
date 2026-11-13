"""Audit bytes, normal retirement, and observed Host intervals."""
import collections,re,statistics
from .common import *
from .analyze import records,percentile


def distribution(values):
    return dict(samples=len(values),mean=statistics.mean(values) if values else None,
        p50=percentile(values,.5),p95=percentile(values,.95),p99=percentile(values,.99),maximum=max(values,default=None))


def analyze(row):
    name=row['name'];folder=ROOT/'raw'/name
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    requests=records(folder/'requests.jsonl')
    migrated=sum(r.get('completed_D_prefill_layers',0)>0 for r in requests)
    layers=sum(r.get('completed_D_prefill_layers',0) for r in requests)
    events=[]
    for p in folder.glob('extension_*.jsonl'):events+=records(p)
    actual=lambda v:bool(re.match(r'^q\d+(?:-|$)',v.get('request_id','')))
    imported=[e for e in events if e['event']=='HIDDEN_GPU_IMPORTED' and actual(e)]
    exported=[e for e in events if e['event']=='HIDDEN_GPU_EXPORTED' and actual(e)]
    released=[e for e in events if e['event']=='HIDDEN_SOURCE_RELEASED' and actual(e)]
    life={p.name:json.loads(p.read_text()) for p in folder.glob('KV_lifetime_*.json')}
    poolclose=[e for e in events if e['event']=='HIDDEN_GPU_POOL_CLOSE']
    kv_bytes=[];kv_ms=[];kv_post=[];failures=[]
    for path in folder.glob('native_decode_*.json'):
        native=json.loads(path.read_text())
        for sample in native.get('NIXL_stats_intervals',[]):
            data=sample['data'];kv_bytes+=data.get('bytes_transferred',[])
            # Installed vLLM stats.py already converts NIXL microseconds to seconds.
            kv_ms += [1000*x for x in data.get('transfer_duration',[])]
            kv_post += [1000*x for x in data.get('post_duration',[])]
            for key in ('num_failed_transfers','num_failed_notifications','num_kv_expired_reqs'):
                failures += [dict(kind=key,value=x) for x in data.get(key,[]) if x]
    expected_kv=(1125*len(requests)-31.25*layers)*1024**2 if row['arm']!='Colocated' else 0
    handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    first={f'q{r["sample_id"]}':r['first_token'] for r in requests}
    commits={q:c['time'] for c in records(folder/'actual_cut_commits.jsonl') for q in c['requests']}
    positive={f'q{r["sample_id"]}' for r in requests if r.get('completed_D_prefill_layers',0)>0}
    window=json.loads((folder/'window.json').read_text());telemetry=[]
    for path in folder.glob('NVML-*.jsonl'):
        telemetry += [r for r in records(path) if window['start']<=r['time']<=window['end']]
    nvml={str(gpu):dict(samples=sum(r['physical_GPU']==gpu for r in telemetry),
        utilization_percent=distribution([r['sampled_utilization_percent'] for r in telemetry if r['physical_GPU']==gpu]),
        sampled_HBM_peak_bytes=max((r['used_HBM_bytes'] for r in telemetry if r['physical_GPU']==gpu),default=None))
        for gpu in cfg['gpus']}
    mixed={p.stem:json.loads(p.read_text())['counters']['prepared_mixed_measurement_batches']
        for p in folder.glob('phase_stable_norm_*.json')}
    steady_batches=collections.Counter()
    geometry=folder/'measured_geometry_decode.json'
    if geometry.exists():
        for shape,count in json.loads(geometry.read_text())['counts'].items():
            item=json.loads(shape)
            if not item['prefilling']:steady_batches[item['requests']]+=count
    worst=None
    for request in requests:
        for earlier,later in zip(request['token_events'],request['token_events'][1:]):
            if 'worker_time' not in earlier or 'worker_time' not in later:continue
            parent_ms=1000*(later['time']-earlier['time'])
            if worst is not None and parent_ms<=worst['parent_ms']:continue
            worker_ms=1000*(later['worker_time']-earlier['worker_time'])
            worst=dict(sample_id=request['sample_id'],from_count=earlier['count'],to_count=later['count'],
                parent_ms=parent_ms,worker_timestamp_interval_ms=worker_ms,
                delivery_lag_change_ms=parent_ms-worker_ms,
                scope='An exact timestamp identity for this interval. Worker timestamps still include scheduling. This is not exclusive GPU time or a decomposition of p99.')
    return dict(name=name,arm=row['arm'],cap=row['cap'],rate=row['rate'],seed=row['seed'],
        migrated_requests=migrated,migrated_layers=layers,
        structural_KV_bytes=expected_kv,reported_NIXL_KV_bytes=sum(kv_bytes),
        exact_reported_KV_byte_accounting=sum(kv_bytes)==expected_kv if row['arm']!='Colocated' else None,
        hidden_exported_bytes=sum(e['bytes'] for e in exported),hidden_imported_bytes=sum(e['bytes'] for e in imported),
        normal_hidden_release_count=len(released),expected_hidden_transfers=migrated,
        exact_hidden_byte_accounting=(len(imported)==len(exported)==len(released)==migrated),
        KV_telemetry_duration_ms=distribution(kv_ms),KV_telemetry_post_ms=distribution(kv_post),
        hidden_transfer_host_ms=distribution([e['transfer_request_host_ms'] for e in imported]),
        hidden_preparation_host_ms=distribution([e['preparation_host_ms'] for e in imported]),
        hidden_NIXL_telemetry_ms=distribution([e['NIXL_xfer_duration_ms'] for e in imported]),
        source_export_enqueue_host_ms=distribution([e['enqueue_host_ms'] for e in exported]),
        migrated_source_ready_to_first_output_ms=distribution([1000*(first[q]-handoffs[q]) for q in positive if q in handoffs]),
        migrated_decision_to_first_output_ms=distribution([1000*(first[q]-commits[q]) for q in positive if q in commits]),
        KV_lifetime_pass=all(v['passed'] for v in life.values()) if life else None,
        hidden_close_pass=all(e['all_ACKed'] for e in poolclose) if poolclose else None,
        NIXL_failures=failures,
        NVML_in_measurement_window=nvml,
        actual_mixed_measurement_batches=mixed,
        steady_Decode_execution_batch_counts=dict(steady_batches),
        maximum_interval_timestamp_identity=worst,
        telemetry_scope='NVML utilization is not kernel busy or compute slack. HBM peaks are sampled. Steady Decode batch counts exclude prefilling-flagged first queries and genuine suffixes.',
        scope='KV telemetry duration is not exclusive wire time. Hidden telemetry uses microseconds converted to milliseconds. Host intervals can overlap. Normal completion audit does not prove cancellation or failure recovery.')


def main():
    progress=json.loads((ROOT/'state/confirmation_progress.json').read_text())['results']
    rows=[analyze(r) for r in progress if r['arm']!='Reference']
    write(ROOT/'analysis/serving_systems_accounting.json',dict(results=rows,evidence='TRACE_ANALYSIS',
        scope='Structural byte expectation checked against actual NIXL reports. No sum of overlapping component times forms a claimed critical-path speedup.'))
    print('SYSTEMS_ACCOUNTING',len(rows),'KV_BYTE_MISMATCHES',sum(r['exact_reported_KV_byte_accounting'] is False for r in rows),
        'HIDDEN_ACCOUNT_MISMATCHES',sum(not r['exact_hidden_byte_accounting'] for r in rows))


if __name__=='__main__':main()
