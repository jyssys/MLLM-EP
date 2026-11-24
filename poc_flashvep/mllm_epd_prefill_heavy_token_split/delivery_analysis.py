"""Compare exact token streams at EngineCore, API yields and the client."""
from collections import defaultdict
import argparse
import json
import time

import numpy as np

from .common import ROOT,write,preserve


def key(request_id):
    if request_id.startswith('chatcmpl-'):request_id=request_id[len('chatcmpl-'):]
    return request_id.split(':D')[0]


def distribution(x):
    return dict(count=len(x),mean=float(np.mean(x)),p50=float(np.percentile(x,50)),
                p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),max=float(max(x))) if x else None


def main():
    p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args();preserve()
    folder=ROOT/'raw/host_timeline'/args.name
    engine=json.loads((folder/'decode/engine.json').read_text())['rows']
    api=json.loads((folder/'decode_api_gpu2/api.json').read_text())['rows']
    clients=[json.loads(s) for s in (ROOT/'raw'/f'{args.name}_profile_requests.jsonl').read_text().splitlines()]
    client_host_path=ROOT/'raw/client_host'/f'{args.name}_profile.json'
    client_host=json.loads(client_host_path.read_text())['rows'] if client_host_path.exists() else []
    emitted=defaultdict(lambda:dict(tokens=[],times=[]));yielded=defaultdict(lambda:dict(tokens=[],times=[]))
    for event in engine:
        if event['kind']!='engine_output':continue
        for item in event['emitted']:
            row=emitted[key(item['request_id'])];row['tokens']+=item['token_ids']
            row['times'] += [event['end_ns']/1e9]*len(item['token_ids'])
    for event in api:
        if event['kind']!='api_generate_yield':continue
        row=yielded[key(event['request_id'])]
        for item in event['outputs']:
            row['tokens']+=item['token_ids'];row['times'] += [event['time_ns']/1e9]*len(item['token_ids'])
    valid=[];unmatched=[];gaps=[];e_to_a=[];a_to_c=[]
    for row in clients:
        if row['status']!='MEASURED_ONLINE':continue
        e=emitted[row['runtime_request_id']];a=yielded[row['runtime_request_id']]
        if e['tokens']!=a['tokens'] or a['tokens']!=row['token_ids']:
            unmatched.append(dict(request_id=row['request_id'],runtime_request_id=row['runtime_request_id'],
                engine_tokens=len(e['tokens']),API_tokens=len(a['tokens']),client_tokens=len(row['token_ids'])))
            continue
        valid.append(row['runtime_request_id'])
        e_to_a.extend((np.array(a['times'])-np.array(e['times']))*1000)
        a_to_c.extend((np.array(row['token_times'])-np.array(a['times']))*1000)
        for i,ms in enumerate(row['ITL_ms']):
            if ms<=34:continue
            left,right=row['token_times'][i:i+2]
            heartbeat=[x['delay_ns']/1e6 for x in api if x['kind']=='event_loop_heartbeat' and left<=x['time_ns']/1e9<=right+.01]
            processing=[dict(kind=x['kind'],duration_ms=(x['end_ns']-x['begin_ns'])/1e6)
                        for x in api if x['kind'] in ['input_processor','online_renderer'] and x['begin_ns']/1e9<right and x['end_ns']/1e9>left]
            detailed_processing=[dict(kind=x['kind'],duration_ms=(x['end_ns']-x['begin_ns'])/1e6,
                thread_CPU_ms=x.get('thread_cpu_ns',0)/1e6,request_id=x.get('request_id'),
                generation=x.get('generation'),thread_id=x.get('thread_id')) for x in api
                if x['kind'] in ['api_gc','multimodal_preprocess_host','ipc_encode_host','ipc_encode_into_host']
                and x['begin_ns']/1e9<right and x['end_ns']/1e9>left]
            client_work=[dict(kind=x['kind'],duration_ms=(x['end_ns']-x['begin_ns'])/1e6,
                generation=x.get('generation'),request_id=x.get('request_id')) for x in client_host
                if x['kind'] in ['gc','raw_append'] and x['begin_ns']/1e9<right and x['end_ns']/1e9>left]
            client_beats=[x['delay_ns']/1e6 for x in client_host if x['kind']=='heartbeat'
                and x['begin_ns']/1e9<right and x['end_ns']/1e9>left]
            gaps.append(dict(request_id=row['request_id'],runtime_request_id=row['runtime_request_id'],index=i+1,
                client_ITL_ms=ms,engine_ITL_ms=(e['times'][i+1]-e['times'][i])*1000,
                API_ITL_ms=(a['times'][i+1]-a['times'][i])*1000,
                next_token_engine_to_API_ms=(a['times'][i+1]-e['times'][i+1])*1000,
                next_token_API_to_client_ms=(right-a['times'][i+1])*1000,
                max_API_heartbeat_delay_ms=max(heartbeat,default=None),overlapping_processing=processing,
                detailed_API_host_work=detailed_processing,
                overlapping_client_work=client_work,max_client_heartbeat_delay_ms=max(client_beats,default=None)))
    result=dict(time=time.time(),name=args.name,evidence='DIAGNOSTIC TOKEN_TIMELINE_ANALYSIS',
        completed_client_requests=sum(r['status']=='MEASURED_ONLINE' for r in clients),
        failed_client_requests=sum(r['status']!='MEASURED_ONLINE' for r in clients),
        exact_engine_API_client_streams=len(valid),unmatched=unmatched,
        engine_to_API_ms=distribution(e_to_a),API_to_client_ms=distribution(a_to_c),
        violating_client_gaps=gaps,
        API_heartbeat_delay_ms=distribution([r['delay_ns']/1e6 for r in api if r['kind']=='event_loop_heartbeat']),
        client_host_observation=str(client_host_path) if client_host else 'NOT_MEASURED',
        client_heartbeat_delay_ms=distribution([r['delay_ns']/1e6 for r in client_host if r['kind']=='heartbeat']),
        client_GC_ms=distribution([(r['end_ns']-r['begin_ns'])/1e6 for r in client_host if r['kind']=='gc']),
        client_raw_append_ms=distribution([(r['end_ns']-r['begin_ns'])/1e6 for r in client_host if r['kind']=='raw_append']),
        scope='Analyze only exactly aligned token streams. Failed requests remain failures. '
            'This analysis does not qualify an incomplete online run or establish a controlled causal effect.')
    write(ROOT/'analysis'/f'{args.name}_delivery_analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='violating_client_gaps'},indent=2))
    for row in sorted(gaps,key=lambda r:r['client_ITL_ms'],reverse=True)[:12]:print(json.dumps(row))
    preserve()


if __name__=='__main__':main()
