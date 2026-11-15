"""Keep Source events, handoff enclosures and NIXL telemetry separate."""
import collections
from .common import *
from .analyze import records,percentile
from .coalesced_costs import parse


def main():
    names=json.loads((ROOT/'configs/coalesced_cost_model_v2.json').read_text())['primitive_calibration_names']
    output=[]
    for name in names:
        folder=ROOT/'raw'/name;groups=parse(name)
        events=json.loads((folder/'CUDA_event_enclosures_prefill.json').read_text())
        telemetry=collections.defaultdict(list)
        for p in folder.glob('native_decode_*.json'):
            for sample in json.loads(p.read_text())['NIXL_stats_intervals']:
                d=sample['data']
                assert len(d['transfer_duration'])==len(d['bytes_transferred'])
                for size,duration in zip(d['bytes_transferred'],d['transfer_duration']):
                    prefix=size//(8000*4096)
                    assert size==prefix*8000*4096 and prefix in (24,28,32,34,36)
                    telemetry[36-prefix].append(1000*duration)
        for group in groups:
            ids=set(group['requests'])
            selected=[e for e in events['rows'] if ids.intersection(e['requests'])]
            assert set(r for e in selected for r in e['requests'])==ids
            group.update(P_stream_event_frame_count=len(selected),
                P_stream_event_elapsed_sum_ms=sum(e['elapsed_ms'] for e in selected),
                P_stream_event_span_ms=1000*(max(e['end_monotonic'] for e in selected)-
                    min(e['start_monotonic'] for e in selected)),
                P_event_anchor_uncertainty_ms=events['anchor_uncertainty_ms'])
        for layers in (0,2,4,8,12):
            selected=[g for g in groups if g['layers']==layers]
            output.append(dict(batch=groups[0]['batch'],layers=layers,groups=len(selected),
                P_prefix_host_p95_ms=percentile([g['P_prefix_host_ms'] for g in selected],.95),
                P_stream_enclosure_p95_ms=percentile([g['P_stream_event_span_ms'] for g in selected],.95),
                P_stream_event_sum_p95_ms=percentile([g['P_stream_event_elapsed_sum_ms'] for g in selected],.95),
                handoff_to_D_stream_start_p95_ms=percentile([g['handoff_to_D_stream_start_ms'] for g in selected],.95),
                D_suffix_stream_enclosure_p95_ms=percentile([g['D_suffix_event_ms'] for g in selected],.95),
                NIXL_transactions=len(telemetry[layers]),
                NIXL_operation_duration_p50_ms=percentile(telemetry[layers],.5),
                NIXL_operation_duration_p95_ms=percentile(telemetry[layers],.95),
                source=name))
    write(ROOT/'analysis/primitive_source_handoff_destination_costs.json',dict(rows=output,
        evidence='MEASURED',
        scope='Instrumented isolated calibration, three groups per cut and shape. Source and suffix CUDA events are stream enclosures, not compute-kernel unions. Native NIXL reports operation durations in seconds, converted to milliseconds here. Native operations can overlap. Do not add or subtract separate p95 values as a request critical-path decomposition. Source event span can include inter-frame launch gaps. Profiler-free serving is evaluated separately.'))
    print('PRIMITIVE_COST_ROWS',len(output),flush=True)


if __name__=='__main__':main()
