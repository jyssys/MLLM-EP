"""Separate a serial-rate estimate from observed pipeline queue behavior."""
import json
import time
import numpy as np
from .common import ROOT,write
from .metrics import summarize
from .qualification import audit
from .service_accounting import analyze


def main():
    protocol=json.loads((ROOT/'configs/proper_pipeline_load_discovery_frozen_v1.json').read_text())
    cells=[]
    for cell in protocol['cells']:
        row={k:v for k,v in cell.items() if k!='jobs'}
        row.update(results=[],status='NOT_RUN',comparisons=[])
        for name in cell['jobs']:
            folder=ROOT/'raw'/name
            receipt=folder/'dispatcher_result.json'
            if not receipt.exists():
                continue
            status=json.loads(receipt.read_text())
            if not status['complete']:
                row['results'].append(dict(name=name,status='FAILED'))
                continue
            cfg=json.loads((folder/'configuration.json').read_text())
            assert status['source_SHA256']==protocol['source_capsule_SHA256']
            m=summarize(name)
            assert m['n']==cell['request_count'] and m['output_tokens']==m['n']*cell['output']
            assert all(g['maximum_requests']==1 for g in m['actual_geometry'].values())
            coverage=audit(name) if cfg.get('axis_kind') else None
            if coverage:
                assert coverage['passed'] and coverage['decode_full36_observation_complete']
            stages=analyze(name)
            queue=[json.loads(s) for s in (folder/'coordinator_queues.jsonl').read_text().splitlines()]
            requests=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
            last_arrival=max(r['arrival'] for r in requests)
            arrival_queue=[r for r in queue if r['time']<=last_arrival]
            middle=arrival_queue[len(arrival_queue)//2:]
            growth={}
            for field in ('P_pipeline_waiting','D_pipeline_waiting'):
                values=[r[field] for r in arrival_queue]
                slope=float(np.polyfit([r['time']-middle[0]['time'] for r in middle],
                    [r[field] for r in middle],1)[0]) if len(middle)>1 else None
                growth[field]=dict(mean=float(np.mean(values)),max=max(values),
                    at_last_arrival=values[-1],second_half_OLS_slope_requests_per_second=slope)
            metrics={k:m[k] for k in ('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms',
                'SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second',
                'completed_requests_per_second','actual_arrival_QPS','actual_dispatch_QPS',
                'Joint_SLO_attainment','decode_SLO_attainment','measurement_seconds')}
            row['results'].append(dict(name=name,status='MEASURED_ONLINE',axis=cfg.get('axis_kind','Native'),
                cut=cfg.get('axis_cut',0),metrics=metrics,host_stage_means_ms=stages['means'],
                coordinator_queue_observations=growth,actual_geometry=m['actual_geometry'],
                overload_scope='A finite30/32-request trace does not establish asymptotic queue stability. '
                    'Report observed queue growth,drain time and realized arrival/completion rates. '
                    'Do not equate max(Pmean,Dmean) from the older serial pilot to measured capacity.'))
        good=[r for r in row['results'] if r['status']=='MEASURED_ONLINE']
        native=next((r for r in good if r['axis']=='Native'),None)
        if native:
            for cut in (2,4,8,12):
                a=next((r for r in good if r['axis']=='Layer' and r['cut']==cut),None)
                b=next((r for r in good if r['axis']=='Token' and r['cut']==cut),None)
                if not a or not b:
                    continue
                row['comparisons'].append(dict(cut=cut,independent_paired_starts=1,
                    Layer_over_Token={k:a['metrics'][k]/b['metrics'][k] for k in
                        ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')},
                    native_relative={v['axis']:{k:v['metrics'][k]/native['metrics'][k] for k in
                        ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')}
                        for v in (a,b)},
                    host_stage_mean_Layer_minus_Token_ms={k:a['host_stage_means_ms'][k]-b['host_stage_means_ms'][k]
                        for k in a['host_stage_means_ms']}))
        if len(good)==len(cell['jobs']):
            row['status']='DISCOVERY_COMPLETE'
        elif row['results']:
            row['status']='PARTIAL_DISCOVERY'
        cells.append(row)
    write(ROOT/'analysis/proper_pipeline_load_discovery_v1.json',dict(time=time.time(),cells=cells,
        evidence='MEASURED_ONLINE',scope='New discriminating load-regime supplement. No replacement of the original64-cell matrix. '
            'All measured unfavorable cells and failures remain. N1 discovery is not a winner or safety CI. '
            'Host service enclosures and queue observations are not GPU-exclusive time or utilization.'))
    print([(r['input_bucket'],r['output'],round(r['QPS'],3),len(r['results']),r['status']) for r in cells],flush=True)


if __name__=='__main__':
    main()
