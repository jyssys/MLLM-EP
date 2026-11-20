"""Validate fixed-service FIFO counterfactuals on matched diagnostic traces."""
import itertools
import json
import numpy as np
from .common import ROOT,write
from .queue_accounting import lines


def simulate(arrival,p_service,d_first,d_decode,credit,p_overhead=None,d_overhead=None,serial=False):
    n=len(arrival);rows=[]
    p_overhead=[0.]*n if p_overhead is None else p_overhead
    d_overhead=[0.]*n if d_overhead is None else d_overhead
    for i,a in enumerate(arrival):
        source_previous=rows[-1]['P_ready'] if rows else a
        credit_return=rows[i-credit]['first'] if i>=credit else a
        serial_return=rows[-1]['finish'] if rows and serial else a
        pstart=max(a,source_previous,credit_return,serial_return)+p_overhead[i]
        ready=pstart+p_service[i]
        dstart=max(ready,rows[-1]['finish'] if rows else ready)+d_overhead[i]
        first=dstart+d_first[i];finish=first+d_decode[i]
        rows.append(dict(P_start=pstart,P_ready=ready,D_start=dstart,first=first,finish=finish,TTFT_ms=1000*(first-a)))
    values=[r['TTFT_ms'] for r in rows]
    return dict(rows=rows,TTFT_mean_ms=float(np.mean(values)),TTFT_p95_ms=float(np.percentile(values,95)))


def extract(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    requests=sorted(lines(folder/'requests.jsonl'),key=lambda r:r['sample_id'])
    offered={r['request_id']:r for r in lines(folder/'offered_requests.jsonl')}
    handoffs={r['request_id']:r for r in lines(folder/'handoffs.jsonl')}
    decisions={r['request_id']:r for r in lines(folder/'axis_decisions.jsonl')}
    submits={(r['role'],r['request_id']):r for r in lines(folder/'request_submissions.jsonl') if r['request_id'].startswith('q')}
    queries=json.loads((folder/'actual_queries_prefill.json').read_text())['rows']
    origin=json.loads((folder/'window.json').read_text())['start']
    rows=[];credit=cfg['source_credit_limit']
    for i,r in enumerate(requests):
        rid=f'q{r["sample_id"]}';pstart=decisions[rid]['time'];ready=handoffs[rid]['received']
        pprepare=min(q['time'] for q in queries if q['request_id']==rid)
        dstart=submits[('decode',rid)]['time'];first=r['first_token'];finish=r['finish']
        previous_ready=handoffs[f'q{requests[i-1]["sample_id"]}']['received'] if i else r['arrival']
        previous_finish=requests[i-1]['finish'] if i else r['arrival']
        credit_return=requests[i-credit]['first_token'] if i>=credit else r['arrival']
        eligible=max(r['arrival'],previous_ready,credit_return,previous_finish if cfg.get('strict_serial_pipeline') else r['arrival'])
        hp=pstart-eligible;hd=dstart-max(ready,previous_finish)
        assert hp>=-1e-6 and hd>=-1e-6,(name,rid,hp,hd)
        rows.append(dict(prompt_id=r['prompt_id'],request_id=rid,arrival=r['arrival']-origin,
            P_service=ready-pstart,D_first_service=first-dstart,D_decode=finish-first,
            P_overhead=max(0.,hp),D_overhead=max(0.,hd),
            P_prepare_to_ready_ms=1000*(handoffs[rid]['source_ready']-pprepare),
            P_dispatch_to_prepare_ms=1000*(pprepare-pstart),
            old_serial_boundary_P_plus_D_ms=1000*((handoffs[rid]['source_ready']-pprepare)+(r['token_events'][0]['worker_time']-handoffs[rid]['source_ready'])),
            output_tokens=r['output_tokens'],output_ids=r['output_ids'],actual_TTFT_ms=r['TTFT_ms'],
            arrival_dispatch_lag_ms=1000*(offered[rid]['enqueue']-r['arrival'])))
    return dict(name=name,axis=cfg['method'],serial=bool(cfg.get('strict_serial_pipeline')),credit=credit,
        rows=rows,arrival_offsets=cfg['arrival_offsets'],dataset_SHA256=cfg['dataset_SHA256'],
        measured_mean_TTFT_ms=float(np.mean([r['actual_TTFT_ms'] for r in rows])),
        measured_p95_TTFT_ms=float(np.percentile([r['actual_TTFT_ms'] for r in rows],95)))


def predict(cost,target,credit=None):
    assert [r['prompt_id'] for r in cost['rows']]==[r['prompt_id'] for r in target['rows']]
    fields=('P_service','D_first_service','D_decode','P_overhead','D_overhead')
    v={f:[r[f] for r in cost['rows']] for f in fields}
    return simulate([r['arrival'] for r in target['rows']],v['P_service'],v['D_first_service'],v['D_decode'],
        target['credit'] if credit is None else credit,v['P_overhead'],v['D_overhead'],target['serial'])


def error(predicted,target):
    actual=np.asarray([r['actual_TTFT_ms'] for r in target['rows']]);p=np.asarray([r['TTFT_ms'] for r in predicted['rows']])
    mae=float(np.mean(abs(p-actual)));p95_ratio=predicted['TTFT_p95_ms']/target['measured_p95_TTFT_ms']
    return dict(MAE_ms=mae,p95_ratio=p95_ratio,underprediction_frequency=float(np.mean(p<actual)),
        predeclared_validation_pass=abs(p95_ratio-1)<=.1 and mae<=max(100.,.1*float(np.mean(actual))))


def main(cut=4):
    source='serial_online_causal_control_v1.json' if cut==4 else 'Last8_serial_online_causal_control_v1.json'
    raw=json.loads((ROOT/'analysis'/source).read_text())['results']
    cells={(r['axis'],r['causal_mode'],r['credit']):extract(r['name']) for r in raw if r['status']!='FAILED'}
    assert all((axis,mode,4) in cells for axis in ('Layer','Token') for mode in ('SERIAL_ADMISSION','ONLINE'))
    summaries={};validation=[]
    for key,value in cells.items():
        axis,mode,credit=key
        summaries[f'{axis}_{mode}_c{credit}']=dict(name=value['name'],
            P_service_mean_ms=1000*float(np.mean([r['P_service'] for r in value['rows']])),
            D_first_service_mean_ms=1000*float(np.mean([r['D_first_service'] for r in value['rows']])),
            D_decode_mean_ms=1000*float(np.mean([r['D_decode'] for r in value['rows']])),
            P_plus_D_first_mean_ms=1000*float(np.mean([r['P_service']+r['D_first_service'] for r in value['rows']])),
            old_serial_boundary_P_plus_D_mean_ms=float(np.mean([r['old_serial_boundary_P_plus_D_ms'] for r in value['rows']])),
            TTFT_mean_ms=value['measured_mean_TTFT_ms'],TTFT_p95_ms=value['measured_p95_TTFT_ms'])
        for label,cost in [('SERIAL_DEMAND',cells[(axis,'SERIAL_ADMISSION',4)]),('LOADED_DEMAND',cells[(axis,'ONLINE',4)])]:
            model=predict(cost,value)
            validation.append(dict(target=value['name'],model=label,predicted_TTFT_p95_ms=model['TTFT_p95_ms'],**error(model,value)))
    interventions=[]
    target=cells[('Layer','ONLINE',4)]
    for source_axis,dest_axis,decode_axis in itertools.product(('Layer','Token'),repeat=3):
        p=cells[(source_axis,'ONLINE',4)]['rows'];d=cells[(dest_axis,'ONLINE',4)]['rows'];dec=cells[(decode_axis,'ONLINE',4)]['rows']
        model=simulate([r['arrival'] for r in target['rows']],[r['P_service'] for r in p],
            [r['D_first_service'] for r in d],[r['D_decode'] for r in dec],4,
            [r['P_overhead'] for r in p],[r['D_overhead'] for r in d])
        interventions.append(dict(P_axis=source_axis,D_suffix_axis=dest_axis,decode_axis=decode_axis,
            TTFT_mean_ms=model['TTFT_mean_ms'],TTFT_p95_ms=model['TTFT_p95_ms']))
    destination='serial_online_service_vector_replay_v1.json' if cut==4 else 'Last8_serial_online_service_vector_replay_v1.json'
    write(ROOT/'analysis'/destination,dict(cut=cut,summaries=summaries,validation=validation,
        factorial_interventions=interventions,evidence='OFFLINE_REPLAY',
        scope='Fixed-output matched-request diagnostic. Calibration service vectors are observed host demands, not GPU-exclusive costs. '
        'Loaded-demand self-reconstruction is an accounting check, not held-out validation. Service swaps are conditional counterfactuals, not measured interventions or online speedups. '
        'A mixed source/destination service vector is not a legal Transformer partition or an achievable method. It isolates model sensitivity only. '
        'No Last8 serial number is combined with a Last4 online number.'))
    print(json.dumps(dict(summaries=summaries,validation=validation),indent=2))


if __name__=='__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv)>1 else 4)
