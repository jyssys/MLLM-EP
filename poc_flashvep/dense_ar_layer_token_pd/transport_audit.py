"""Check submitted bytes and native telemetry without adding a profiler."""
import collections
import json
from .common import ROOT, write
from .analysis import records
from ..dense_ar_slo_layer_pd.analyze import percentile


def summary(values):
    return dict(count=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
        maximum=max(values,default=None))


def audit(name, expected):
    folder=ROOT/'raw'/name
    window=json.loads((folder/'window.json').read_text())
    data=collections.defaultdict(list)
    flushed=[]
    for path in folder.glob('native_decode_*.json'):
        value=json.loads(path.read_text())
        flushed.append(value.get('NIXL_measured_boundary_flushed'))
        for event in value['NIXL_stats_intervals']:
            for key,values in event['data'].items():
                data[key].extend(values)
    hidden=[event for p in folder.glob('extension_decode_*.jsonl') for event in records(p)
        if event.get('event')=='HIDDEN_GPU_IMPORTED' and window['start']<=event['time']<=window['end']]
    manual=json.loads((folder/'manual_KV_coalescing.json').read_text())['rows'] if (folder/'manual_KV_coalescing.json').exists() else []
    manual=[r for r in manual if window['start']<=r['time']<=window['end']]
    actual_KV=sum(data['bytes_transferred']) if data['bytes_transferred'] else None
    actual_hidden=sum(event['NIXL_total_bytes'] for event in hidden)
    result=dict(name=name,evidence='TRACE_ANALYSIS',
        expected_valid_KV_bytes=expected['valid_KV_payload_bytes'],
        NIXL_reported_KV_bytes=actual_KV,
        KV_byte_gate=actual_KV==expected['valid_KV_payload_bytes'] if actual_KV is not None else None,
        submitted_KV_bytes=sum(r['bytes'] for r in manual) if manual else None,
        measured_boundary_flushed=flushed,
        expected_hidden_residual_bytes=expected['hidden_residual_payload_bytes'],
        NIXL_reported_hidden_residual_bytes=actual_hidden,
        hidden_byte_gate=actual_hidden==expected['hidden_residual_payload_bytes'],
        KV_transfer_telemetry_ms=summary([v*1000 for v in data['transfer_duration']]),
        KV_post_telemetry_ms=summary([v*1000 for v in data['post_duration']]),
        KV_coalescing_host_ms=summary([r['coalescing_host_ms'] for r in manual]),
        hidden_transfer_telemetry_ms=summary([r['NIXL_xfer_duration_ms'] for r in hidden]),
        hidden_submit_to_done_host_ms=summary([r['transfer_request_host_ms'] for r in hidden]),
        failed_transfers=data['num_failed_transfers'],failed_notifications=data['num_failed_notifications'],
        scope='Native transfer telemetry and actual submitted ranges. Warm-up native counters are flushed at measurement begin. Manual descriptors are filtered by the measured host window. NIXL KV stats use seconds; hidden API fields use microseconds converted by the original wrapper. Concurrent transfer durations are not additive request critical paths or exclusive wire/kernel times. Hidden0 is supported only when the method sends no hidden state.')
    write(ROOT/'analysis'/(name+'_transport_audit.json'),result)
    return result
