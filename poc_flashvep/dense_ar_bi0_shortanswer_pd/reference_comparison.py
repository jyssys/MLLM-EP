"""Compare BI0 client boundaries without calling them kernel durations."""
import json
import numpy as np
from .common import ROOT,PREVIOUS,write
from .serving_analysis import stats


def main():
    name='BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1'
    source=json.loads((ROOT/'raw'/name/'natural_reference.json').read_text())['rows']
    intervals=[];tpots=[]
    for row in source:
        events=row['token_events'];first=events[0]['time'];last=events[-1]['time']
        if row['output_tokens']>1:tpots.append(1000*(last-first)/(row['output_tokens']-1))
        intervals.extend(1000*(b['time']-a['time']) for a,b in zip(events,events[1:]))
    reference=dict(name=name,n=200,independent_starts=1,mean_TBT_ms=float(np.mean(intervals)),
        ITL_ms=stats(intervals),request_TPOT_ms=stats(tpots),
        scope='Sequential stock-vLLM engine outputs, concurrency1. No network/IPC. Standard sampling and Natural EOS. Not open-loop throughput or GPU-exclusive time.')
    native=[]
    for pattern in ('Main_Native_*_BI0_v3_metrics.json','POnly_ContinuousNative_*clean_before*_metrics.json',
                    'Confirmation_ControlledNative*_metrics.json'):
        for p in sorted((ROOT/'analysis').glob(pattern)):
            data=json.loads(p.read_text());native.append({key:data[key] for key in
                ('name','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens','offered_QPS')})
    historical=PREVIOUS/'analysis/decode_kernel_host_breakdown_v2.json'
    write(ROOT/'analysis/BI0_official_reference_comparison_v1.json',dict(reference=reference,Native=native,
        previous_kernel_decomposition=str(historical) if historical.exists() else None,
        evidence='MEASURED_REFERENCE_AND_ONLINE_WITH_DIFFERENT_BOUNDARIES',
        scope='Do not compare an engine-output interval to exclusive kernel time. Native coordinator delivery includes transport/IPC and actual batching. '
        'Old BI1/BI0 kernel controls remain historical. New sparse tensor probes are correctness-only and cannot provide latency.'))


if __name__=='__main__':main()
