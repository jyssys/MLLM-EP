"""Locate transfer and delivery overlaps without assigning causal kernel cost."""
import json
from .common import ROOT,write
from .qualification import records
from .serving_analysis import stats
from .profile_results import union


def analyze(name):
    folder=ROOT/'raw'/name;window=json.loads((folder/'window.json').read_text())
    calls=json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
    calls=[r for r in calls if window['start']<=r.get('start',r.get('time',0))<=window['end']]
    submits=[r for r in calls if r['kind']=='submit_transfer_host']
    jit=json.loads((folder/'JIT_events_decode.json').read_text())['rows']
    intervals=[]
    for row in records(folder/'requests.jsonl'):
        for previous,current in zip(row['token_events'],row['token_events'][1:]):
            left=previous['time'];right=current['time'];worker_left=previous['worker_time'];worker_right=current['worker_time']
            duration=1000*(right-left)
            if duration<=34:continue
            overlap=union([(max(worker_left,r['start']),min(worker_right,r['end'])) for r in submits
                if r['start']<worker_right and r['end']>worker_left])
            intervals.append(dict(prompt_id=row['prompt_id'],task=row['task'],output_position=current['count'],
                relative_start_seconds=left-window['start'],client_ITL_ms=duration,
                worker_emission_interval_ms=1000*(worker_right-worker_left),
                delivery_lag_change_ms=1000*((right-worker_right)-(left-worker_left)),
                overlapping_transfer_host_ms=1000*overlap,
                JIT_notifications_inside_worker_interval=[e for e in jit if worker_left<=e['time']<=worker_right]))
    result=dict(name=name,evidence='TRACE_ANALYSIS',threshold_ms=34,
        transfer_host_ms=stats(1000*(r['end']-r['start']) for r in submits),
        transfer_calling_thread_CPU_ms=stats(r['calling_thread_CPU_ms'] for r in submits),
        SDK_postDuration_us=stats(r['data']['postDuration'] for r in calls if r['kind']=='NIXL_native_telemetry' and r['data']['postDuration'] is not None),
        SDK_total_bytes=stats(r['data']['totalBytes'] for r in calls if r['kind']=='NIXL_native_telemetry' and r['data']['totalBytes'] is not None),
        tail_interval_count=len(intervals),tails_with_transfer_overlap=sum(r['overlapping_transfer_host_ms']>0 for r in intervals),
        tails_with_JIT_notification=sum(bool(r['JIT_notifications_inside_worker_interval']) for r in intervals),
        largest_intervals=sorted(intervals,key=lambda r:r['client_ITL_ms'],reverse=True)[:30],
        scope='Same-host timestamps. Transfer overlap is an observed co-occurrence, not exclusive GPU time or proof of causality. JIT notifications identify completion points, not compilation durations. Warmup calls are excluded by the measured window.')
    write(ROOT/'analysis'/(name+'_tail_overlap_analysis.json'),result)
    return result


if __name__=='__main__':
    import sys
    value=analyze(sys.argv[1]);print(json.dumps({k:v for k,v in value.items() if k!='largest_intervals'},indent=2))
