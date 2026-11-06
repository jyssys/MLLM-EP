"""Derive metrics from request timestamps; preserve all failed run receipts."""
import csv
import json
import statistics
from .common import ROOT, write


def pct(values, q):
    if not values:
        return None
    import numpy as np
    return float(np.percentile(values,q))


def table(path, rows):
    path.parent.mkdir(exist_ok=True,parents=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def requests(folder):
    path=folder/'requests.jsonl'
    return [] if not path.exists() else [row for line in path.read_text().splitlines()
        if not (row:=json.loads(line)).get('warmup',True)]


def stream_starts(folder,window_start):
    starts={}
    for path in folder.glob('service_device_times_*.json'):
        for item in json.loads(path.read_text())['requests']:
            stamp=item['first_model_stream_entry']['time']
            if stamp<window_start:continue
            starts[item['seed']]=min(starts.get(item['seed'],stamp),stamp)
    return starts


def main():
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())
    # Frozen file has per-model dictionaries; preserve its actual field names.
    runs=[];inventory=[];micro=[];qual=[]
    for path in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        folder=path.parent;cfg=json.loads((ROOT/'configs'/(folder.name+'.json')).read_text())
        receipt=json.loads(path.read_text());rows=requests(folder)
        row=dict(name=folder.name,model=cfg['model'],method=cfg.get('dlocal_method'),
            worker=cfg['worker'],status=receipt.get('status'),complete=receipt.get('complete'),
            observed_requests=len(rows),requested_requests=cfg.get('count'),
            heavy_correctness=bool(cfg.get('state_probe') or cfg.get('verify_transfer')),
            elapsed_seconds=receipt.get('elapsed_seconds'),configuration=str(ROOT/'configs'/(folder.name+'.json')))
        row['dispatcher_timing_qualification']=receipt.get('timing_qualification','NORMAL_GUARDED_EXECUTION')
        inventory.append(row)
        qp=folder/'local_qualification.json'
        if qp.exists():
            q=json.loads(qp.read_text());qual.append(dict(name=folder.name,model=cfg['model'],count=q['count'],passed=q['passed']))
        for item in json.loads((folder/'full_core_micro.json').read_text()) if (folder/'full_core_micro.json').exists() else []:
            valid_geometry=not folder.name.endswith('fair_grid_s1') or item['chunk'] in (16,32,64) or (item['chunk']==128 and item['decode_batch']==1)
            base=dict(name=folder.name,model=cfg['model'],chunk=item['chunk'],decode_batch=item['decode_batch'],
                depth_class=item['depth_class'],hidden_exact=item['hidden_exact'],KV_exact=item['KV_exact'],
                control_version=item.get('control_version',1),
                    primary_eligible=item.get('control_version',1)>=5 and valid_geometry and not (folder/'superseded_control.json').exists())
            for index, repeat in enumerate(item.get('paired_repetitions',[])):
                micro.append(dict(base,repetition=index,batched_ms=pct(repeat['batched_ms'],50),
                    separate_ms=pct(repeat['separate_ms'],50),
                    batching_ratio=pct(repeat['batched_ms'],50)/pct(repeat['separate_ms'],50)))
        if not rows or cfg['worker']!='rlt_worker' or row['heavy_correctness']:
            continue
        window_path=folder/'window.json'
        if not window_path.exists():
            continue
        window=json.loads(window_path.read_text());duration=window['end']-window['start']
        offered_path=folder/'offered_requests.jsonl'
        offered=[json.loads(line) for line in offered_path.read_text().splitlines()] if offered_path.exists() else []
        sends=[r['actual_send_start'] for r in offered]
        limit=slos[f'{cfg["model"]}_L{cfg["length"]}']
        ttft,tpot,itl_limit=limit['TTFT_ms'],limit['TPOT_ms'],limit['ITL_ms']
        itls=[];unresolved=0;per_request=[]
        for request in rows:
            local=[]
            for event in request['token_events']:
                if event['first']:
                    continue
                if event['new_tokens']!=1:
                    unresolved+=1
                    continue
                value=1000*(event['time']-event['previous_time']);itls.append(value);local.append(value)
            per_request.append(dict(sample_id=request['sample_id'],deadline_misses=sum(x>itl_limit for x in local),
                max_ITL_ms=max(local,default=None)))
        starts=stream_starts(folder,window['start'])
        service=[1000*(r['first_token']-starts[r['seed']]) for r in rows if r['seed'] in starts]
        queues=[1000*(starts[r['seed']]-r['arrival']) for r in rows if r['seed'] in starts]
        decomposed=[dict(sample_id=r['sample_id'],seed=r['seed'],arrival=r['arrival'],first_token=r['first_token'],
            first_model_stream_entry=starts[r['seed']],TTFT_ms=r['ttft_ms'],
            queue_ms=1000*(starts[r['seed']]-r['arrival']),service_TTFT_ms=1000*(r['first_token']-starts[r['seed']]))
            for r in rows if r['seed'] in starts]
        if decomposed:
            assert all(abs(r['TTFT_ms']-r['queue_ms']-r['service_TTFT_ms'])<.01 for r in decomposed)
            table(folder/'service_decomposition.csv',decomposed)
        result=dict(row,offered_QPS=cfg['arrival_rate'],input_tokens=cfg['length'],output_tokens=cfg['cap'],
            arrival_seed=cfg['arrival_seed'],marker_on=cfg.get('observe_service_timing',False),
            activity_profiler=bool(cfg.get('kernel_activity_seconds')),duration_seconds=duration,
            offered_requests=len(offered),failed_or_unfinished_requests=max(0,cfg['count']-len(rows)),
            scheduled_cohort_QPS=(cfg['count']-1)/(window['last_arrival']-window['start']),
            realized_QPS=(len(sends)-1)/(max(sends)-min(sends)) if len(sends)>1 and max(sends)>min(sends) else None,
            send_delay_p99_ms=pct([1000*(r['actual_send_start']-r['arrival']) for r in offered],99),
            completed_QPS=len(rows)/duration,input_tokens_per_s=sum(r['input_tokens'] for r in rows)/duration,
            output_tokens_per_s=sum(r['output_tokens'] for r in rows)/duration,
            total_tokens_per_s=sum(r['input_tokens']+r['output_tokens'] for r in rows)/duration,
            joint_SLO_goodput=sum(r['ttft_ms']<=ttft and r['tpot_ms']<=tpot for r in rows)/duration,
            joint_SLO_fraction=sum(r['ttft_ms']<=ttft and r['tpot_ms']<=tpot for r in rows)/cfg['count'],
            ITL_unresolved_delivery_events=unresolved,
            max_ITL_ms=max(itls,default=None),ITL_deadline_misses=sum(x>itl_limit for x in itls),
            ITL_samples=len(itls),requests_with_ITL_miss=sum(r['deadline_misses']>0 for r in per_request),
            service_timestamp_count=len(service),service_TTFT_scope='CUDA-event mapped first model-stream entry; includes all subsequent work and handoff' if service else 'NOT_MEASURED',
            decode_screen_TPOT_ms=.9*tpot if cfg.get('dlocal_method')!='native' else tpot,
            decode_screen_ITL_ms=itl_limit)
        for field,values in [('TTFT',[r['ttft_ms'] for r in rows]),('TPOT',[r['tpot_ms'] for r in rows]),
                ('E2E',[r['e2e_ms'] for r in rows]),('ITL',itls),('service_TTFT',service),('queue',queues)]:
            for q in (50,95,99):
                result[f'{field}_p{q}_ms']=pct(values,q)
        result['decode_screen_pass']=bool(unresolved==0 and result['TPOT_p95_ms']<=result['decode_screen_TPOT_ms']
            and result['ITL_p99_ms']<=itl_limit and len(rows)==cfg['count'])
        from .qualification import compare_outputs
        parity=compare_outputs(f'local_{cfg["model"]}_native_n128_s2026110701_off',folder.name,cfg['count'])
        result['serving_output_exact']=parity['passed']
        write(folder/'derived_request_misses.json',per_request)
        runs.append(result)
    table(ROOT/'run_inventory.csv',inventory);table(ROOT/'serving_runs.csv',runs)
    table(ROOT/'full_core_micro.csv',micro);table(ROOT/'qualification.csv',qual)
    write(ROOT/'analysis_snapshot.json',dict(runs=len(inventory),serving=len(runs),micro_rows=len(micro),qualifications=qual))
    print(json.dumps(dict(runs=len(inventory),serving=len(runs),micro_rows=len(micro))))


if __name__=='__main__':
    main()
