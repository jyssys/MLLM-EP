"""Exclude warmup before reporting native SDK telemetry distributions."""
import json
import numpy as np
from .common import ROOT,write
from .serving_analysis import stats
from .profile_results import union


def telemetry(name):
    folder=ROOT/'raw'/name;window=json.loads((folder/'window.json').read_text())
    source=json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
    data=[r for r in source if r['kind']=='NIXL_native_telemetry' and
          window['start']<=r['time']<=window['end'] and not r.get('agent_name','').startswith('BI0-hidden-')]
    calls=[r for r in source if r['kind']=='submit_transfer_host' and
           window['start']<=r['start']<=window['end'] and not r.get('agent_name','').startswith('BI0-hidden-')]
    intervals=[]
    for line in (folder/'requests.jsonl').read_text().splitlines():
        request=json.loads(line)
        for previous,current in zip(request['token_events'],request['token_events'][1:]):
            left=previous['time'];right=current['time']
            overlap=union([(max(left,c['start']),min(right,c['end'])) for c in calls if c['start']<right and c['end']>left])
            intervals.append(dict(ITL_ms=1000*(right-left),API_overlap_ms=1000*overlap))
    tail=[r for r in intervals if r['ITL_ms']>34]
    return dict(name=name,completed_requests=len((folder/'requests.jsonl').read_text().splitlines()),
        telemetry_calls=len(data),submit_calls=len(calls),
        SDK={key:stats(r['data'][key] for r in data if r['data'].get(key) is not None)
             for key in ('postDuration','xferDuration','totalBytes','descCount')},
        host_submit_ms=stats(1000*(r['end']-r['start']) for r in calls),
        calling_thread_CPU_ms=stats(r['calling_thread_CPU_ms'] for r in calls if 'calling_thread_CPU_ms' in r),
        client_intervals=dict(events=len(intervals),above34ms=len(tail),
            tail_events_with_host_API_overlap=sum(r['API_overlap_ms']>0 for r in tail),
            tail_host_API_overlap_ms=sum(r['API_overlap_ms'] for r in tail),
            tail_interval_ms=sum(r['ITL_ms'] for r in tail),
            scope='Temporal overlap, not independent causal proof. Counts are delivered event intervals, not independent server starts.'),
        agents=[r for r in source if r['kind']=='NIXL_agent_configuration'],
        evidence='TRACE_ANALYSIS',
        scope='Measurement-window SDK calls only. Warmup is excluded. SDK post/xfer durations are microseconds. '
        'Neither SDK enclosure nor host submit time is pure DMA or exclusive GPU kernel time.')


def main():
    names=['diag_Native_T4_ShortAnswer_N32_q2.35_BI0_v1',
        'diag_Native_KV128_ShortAnswer_N32_q2.35_BI0_v1','diag_Native_KV256_ShortAnswer_N32_q2.35_BI0_v1',
        'InitAB_ObserverImport_Native_N32_q2.35_BI0_v2','InitAB_OfficialInit_Native_N32_q2.35_BI0_v2']
    results=[]
    for name in names:
        if (ROOT/'raw'/name/'window.json').exists():results.append(telemetry(name))
    write(ROOT/'analysis/BI0_filtered_NIXL_telemetry_v1.json',dict(results=results))
    return results


if __name__=='__main__':
    for r in main():print(r['name'],r['telemetry_calls'],r['SDK'])
