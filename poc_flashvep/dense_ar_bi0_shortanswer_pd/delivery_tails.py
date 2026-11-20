"""Separate client delivery increments from worker emission intervals."""
import json
import numpy as np
from .common import ROOT,write


def intervals(row):
    events=row['token_events'];paired=[];unpaired=max(0,events[0]['count']-1)
    for previous,current in zip(events,events[1:]):
        count=current['count']-previous['count'];assert count>0
        if count!=1 or previous.get('worker_time') is None or current.get('worker_time') is None:
            unpaired+=count;continue
        client=1000*(current['time']-previous['time'])
        worker=1000*(current['worker_time']-previous['worker_time'])
        delivery_delta=1000*((current['time']-current['worker_time'])-(previous['time']-previous['worker_time']))
        assert abs(client-worker-delivery_delta)<1e-3
        paired.append(dict(prompt_id=row['prompt_id'],position_after_first=current['count']-1,
            client_ITL_ms=client,worker_emission_interval_ms=worker,delivery_lag_increment_ms=delivery_delta))
    assert len(paired)+unpaired==row['output_tokens']-1
    return paired,unpaired


def main():
    patterns=('Main_*_BI0_v3_metrics.json','ControlledDiscovery_*_BI0_v3_metrics.json',
        'Confirmation_*_BI0_v2_metrics.json','FixedQPS_*_BI0_v1_metrics.json','RootCut12_*_BI0_v1_metrics.json',
        'LowQPSControl_*_BI0_v1_metrics.json','EqualOutputControl_*_BI0_v1_metrics.json')
    results=[]
    for file in sorted({p for pattern in patterns for p in (ROOT/'analysis').glob(pattern)}):
        metrics=json.loads(file.read_text());name=metrics['name'];folder=ROOT/'raw'/name
        cfg=json.loads((folder/'configuration.json').read_text())
        rows=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
        all_pairs=[];unpaired=0;threshold=metrics['SLO']['every_ITL_ms']
        for row in rows:
            values,missing=intervals(row);all_pairs.extend(values);unpaired+=missing
        violations=[value for value in all_pairs if value['client_ITL_ms']>threshold]
        results.append(dict(name=name,requests=len(rows),paired_intervals=len(all_pairs),unpaired_intervals=unpaired,
            total_client_intervals=sum(row['output_tokens']-1 for row in rows),threshold_ms=threshold,
            paired_client_violations=len(violations),
            worker_already_above_threshold=sum(value['worker_emission_interval_ms']>threshold for value in violations),
            client_crosses_threshold_with_worker_below=sum(value['worker_emission_interval_ms']<=threshold for value in violations),
            maximum_client_ITL_ms=max((value['client_ITL_ms'] for value in all_pairs),default=None),
            worker_emission_interval_p99_ms=float(np.percentile([value['worker_emission_interval_ms'] for value in all_pairs],99)) if all_pairs else None,
            top_client_intervals=sorted(all_pairs,key=lambda value:value['client_ITL_ms'],reverse=True)[:10],
            output_contract=rows[0].get('output_contract',
                'NATURAL_EOS' if not cfg.get('ignore_eos',False) else 'CONTROLLED_OUTPUT_LENGTH')))
    value=dict(results=results,evidence='TRACE_ANALYSIS',scope='Client ITL equals worker emission interval plus delivery-lag increment for each paired event. '
        'Worker timestamps are host output-emission clocks. They are not GPU kernel completion times. '
        'A long worker interval can include host scheduling, NIXL and GPU work. Multi-token events without individual worker clocks remain unpaired. '
        'Primary client-visible latency is not replaced by the worker interval.')
    write(ROOT/'analysis/BI0_client_worker_delivery_tail_decomposition_v1.json',value);return value


if __name__=='__main__':main()
