"""Open-loop HTTP measurements with explicit token IDs and planned arrivals."""
import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

from .common import ROOT,MODEL_ID,write,append,digest,preserve

LOADED_SOURCE_SHA256=digest(Path(__file__))


def duration_anchor(serial,planned_start,results):
    return min(r['scheduled_arrival'] for r in results) if serial else planned_start


def load_replay_counts(path):
    counts={}
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['status']!='MEASURED_ONLINE':continue
            key=row['request_id'];value=row['output_tokens']
            assert key not in counts,'Duplicate reference request IDs'
            assert isinstance(value,int) and value>0
            counts[key]=value
    return counts


def messages(row,nonce):
    content=[dict(type='image_url',image_url=dict(url=Path(image['path']).as_uri()),
        uuid=hashlib.sha256((nonce+row['request_id']+str(i)).encode()).hexdigest())
        for i,image in enumerate(row['images'])]
    content.append(dict(type='text',text=row['question']))
    return [dict(role='user',content=content)]


def payload(row,nonce,output_count=None,stream=True):
    request=dict(model=MODEL_ID,messages=messages(row,nonce),temperature=0,top_p=1,
        max_tokens=2048 if output_count is None else output_count,ignore_eos=output_count is not None,
        stream=stream,return_token_ids=True,seed=2026112401)
    if stream:request['stream_options']=dict(include_usage=True)
    return request


def resolve_manifest(explicit, active_path=None):
    if explicit is not None:return Path(explicit)
    active_path=Path(active_path) if active_path is not None else ROOT/'configs/active_workload.json'
    active=json.loads(active_path.read_text());path=Path(active['main_manifest'])
    assert digest(path)==active['main_SHA256'],'The active workload manifest changed after freeze'
    return path


def resolve_output_count(output_count,row):
    value=output_count[row['request_id']] if isinstance(output_count,dict) else output_count
    assert value is None or isinstance(value,int) and value>0
    return value


async def one(session,url,row,body,arrival,run,index,host_observer=None,evidence_sink=None):
    await asyncio.sleep(max(0,arrival-time.monotonic()))
    dispatch=time.monotonic();request_id=f'{run}_{index:04d}'
    result=dict(request_id=row['request_id'],runtime_request_id=request_id,index=index,
        scheduled_arrival=arrival,dispatch=dispatch,dispatch_lateness=dispatch-arrival,
        payload_SHA256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
        prompt_SHA256=row['prompt_SHA256'],input_tokens_manifest=row['total_input_tokens'],
        source=row['source'],stratum=row['stratum'],image_count=row['image_count'],
        status='FAILED_EXECUTION',token_ids=[],token_times=[],text='',chunks=[],usage=None)
    try:
        async with session.post(url+'/v1/chat/completions',json=body,headers={'x-request-id':request_id}) as response:
            result['HTTP_status']=response.status
            if response.status!=200:raise RuntimeError((await response.text())[:4000])
            pending=b''
            async for block in response.content.iter_any():
                pending+=block
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip();now=time.monotonic()
                    if data==b'[DONE]':continue
                    chunk=json.loads(data);result['chunks'].append(dict(time=now,data=chunk))
                    if chunk.get('error'):raise RuntimeError(str(chunk['error']))
                    if chunk.get('usage'):result['usage']=chunk['usage']
                    if chunk.get('prompt_token_ids') is not None:
                        result['prompt_token_ids']=chunk['prompt_token_ids']
                    for choice in chunk.get('choices',[]):
                        delta=choice.get('delta') or {}
                        ids=choice.get('token_ids') or delta.get('token_ids') or []
                        result['token_ids']+=ids;result['token_times'] += [now]*len(ids)
                        result['text']+=delta.get('content') or ''
                        if choice.get('finish_reason'):result['finish_reason']=choice['finish_reason']
            result['finish']=time.monotonic()
            assert result['token_ids'],'No explicit output token IDs. Text chunks are not tokens.'
            assert result['usage'] and result['usage']['completion_tokens']==len(result['token_ids']),(
                result['usage'],len(result['token_ids']))
            assert result['usage']['prompt_tokens']==row['total_input_tokens'],(
                'Processor/server input mismatch',result['usage']['prompt_tokens'],row['total_input_tokens'])
            times=result['token_times'];n=len(times)
            result.update(status='MEASURED_ONLINE',output_tokens=n,TTFT_ms=(times[0]-arrival)*1000,
                ITL_ms=[(b-a)*1000 for a,b in zip(times,times[1:])],
                TPOT_ms=(times[-1]-times[0])*1000/(n-1) if n>1 else 0,
                E2E_ms=(result['finish']-arrival)*1000,
                output_cap_hit=result.get('finish_reason')=='length')
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error);result['finish']=time.monotonic()
    raw_path=ROOT/'raw'/(run+'_requests.jsonl')
    if evidence_sink is not None:evidence_sink.record(result)
    elif host_observer is None:append(raw_path,result)
    else:host_observer.record_append(raw_path,result)
    print('REQUEST_FINISHED',run,index,result['status'],result.get('output_tokens'),result.get('error',''),flush=True)
    return result


async def execute(rows,urls,run,qps,seed,output_count,serial=False,metadata=None,observe_host=False,
                  freeze_static_heap=False,defer_raw=False,unique_media_occurrence=False,timing_hook=None):
    import aiohttp
    rng=random.Random(seed);offset=0;trace=[]
    for i,row in enumerate(rows):
        if i and not serial:offset+=rng.expovariate(qps)
        trace.append(dict(index=i,request_id=row['request_id'],offset_seconds=offset,replica=i%len(urls)))
    trace_file=ROOT/'configs'/(run+'_arrival.json')
    assert not (ROOT/'raw'/(run+'_requests.jsonl')).exists(),'Do not overwrite a measured run'
    assert not (ROOT/'raw'/(run+'_completion_journal.jsonl')).exists(),'Do not overwrite a completion journal'
    write(trace_file,dict(seed=seed,QPS=qps,serial=serial,trace=trace,
        definition='Open-loop Poisson. Arrival is the planned timestamp. Dispatch lateness remains in TTFT.'))
    write(ROOT/'configs'/(run+'.json'),dict(urls=urls,QPS=qps,seed=seed,count=len(rows),
        source_SHA256=LOADED_SOURCE_SHA256,source_hash_scope='Captured when this client module was imported',arrival_SHA256=digest(trace_file),
        output_mode='Natural EOS' if output_count is None else
            'Per-request equal-output replay' if isinstance(output_count,dict) else 'Fixed-output diagnostic',
        output_count=output_count,mm_processor_cache_gb=1,unique_media_UUID_per_request=True,
        unique_media_UUID_per_occurrence=unique_media_occurrence,
        prefix_cache=False,transport='HTTP SSE',evidence='MEASURED_ONLINE' if not serial else 'CORRECTNESS_REFERENCE',
        workload=metadata or {},client_host_observer=observe_host,
        freeze_client_static_heap=freeze_static_heap,defer_client_full_raw=defer_raw))
    timeout=aiohttp.ClientTimeout(total=7200,sock_read=3600)
    connector=aiohttp.TCPConnector(limit=0,limit_per_host=0,enable_cleanup_closed=True,keepalive_timeout=1)
    heap=None;heap_state=None
    if freeze_static_heap:
        from .client_gc import StaticHeapFreeze
        heap=StaticHeapFreeze();heap_state=heap.begin()
    wall=time.time();start=time.monotonic()+1
    host_observer=None
    if observe_host:
        from .client_host_observer import ClientHostObserver
        host_observer=ClientHostObserver(run)
        await host_observer.begin()
    evidence_sink=None;evidence_state=None
    if defer_raw:
        from .client_evidence import DeferredEvidence
        evidence_sink=DeferredEvidence(ROOT/'raw'/(run+'_requests.jsonl'),
            ROOT/'raw'/(run+'_completion_journal.jsonl'),host_observer)
    measurement_done=asyncio.Event() if timing_hook is not None else None
    timing_task=asyncio.create_task(timing_hook(start,trace,measurement_done)) if timing_hook is not None else None
    try:
        async with aiohttp.ClientSession(timeout=timeout,connector=connector,trust_env=False) as session:
            if serial:
                results=[]
                for i,row in enumerate(rows):
                    results.append(await one(session,urls[i%len(urls)],row,payload(row,run,resolve_output_count(output_count,row)),
                        time.monotonic(),run,i,host_observer,evidence_sink))
                    if results[-1]['status']!='MEASURED_ONLINE':break
            else:
                results=await asyncio.gather(*(one(session,urls[t['replica']],row,payload(
                    row,f"{run}_{t['index']:04d}" if unique_media_occurrence else run,resolve_output_count(output_count,row)),
                    start+t['offset_seconds'],run,t['index'],host_observer,evidence_sink) for row,t in zip(rows,trace)))
    finally:
        try:
            if timing_task is not None:
                measurement_done.set()
                await timing_task
        finally:
            if host_observer is not None:await host_observer.close()
            if evidence_sink is not None:evidence_state=evidence_sink.flush()
            if heap is not None:heap.close()
    measured_start=duration_anchor(serial,start,results)
    record=dict(run=run,wall_start=wall,monotonic_start=measured_start,wall_end=time.time(),
        planned_count=len(rows),actual_count=len(results),not_run_count=len(rows)-len(results),
        completed=sum(r['status']=='MEASURED_ONLINE' for r in results),
        duration_seconds=max(r['finish'] for r in results)-measured_start,serial=serial,
        duration_anchor='Actual first serial arrival' if serial else 'Planned first open-loop arrival',
        client_heap_policy=heap_state,client_evidence_policy=evidence_state,
        scope='Serial reference, not sustainable capacity' if serial else 'Finite-cohort open-loop arrival and drain',
        raw_SHA256=digest(ROOT/'raw'/(run+'_requests.jsonl')),passed=len(results)==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in results))
    write(ROOT/'state'/(run+'_complete.json'),record);preserve();return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,
        help='Default: the frozen manifest in configs/active_workload.json')
    p.add_argument('--section',default='requests');p.add_argument('--urls',default='http://127.0.0.1:18303')
    p.add_argument('--run',required=True);p.add_argument('--qps',type=float,default=.25)
    p.add_argument('--seed',type=int,default=2026112402);p.add_argument('--count',type=int)
    p.add_argument('--subset',choices=['all','discovery','heldout'],default='all')
    p.add_argument('--order-seed',type=int,default=2026112404)
    output=p.add_mutually_exclusive_group()
    output.add_argument('--output-count',type=int)
    output.add_argument('--replay-from',type=Path,help='Preserve each request output count from a completed reference JSONL')
    p.add_argument('--serial',action='store_true')
    p.add_argument('--observe-client-host',action='store_true',help='Diagnostic GC, heartbeat and append timing')
    p.add_argument('--freeze-client-static-heap',action='store_true',help='Controlled startup-only client GC intervention')
    p.add_argument('--defer-client-raw',action='store_true',help='Write a compact completion journal, then full unchanged rows after drain')
    args=p.parse_args();args.manifest=resolve_manifest(args.manifest)
    manifest=json.loads(args.manifest.read_text());rows=manifest[args.section]
    if args.subset!='all':
        selected=set(manifest[args.subset+'_ids']);rows=[r for r in rows if r['request_id'] in selected]
    if not args.serial:random.Random(args.order_seed).shuffle(rows)
    if args.count:rows=rows[:args.count]
    replay=None
    if args.replay_from:
        replay=load_replay_counts(args.replay_from)
        assert all(row['request_id'] in replay for row in rows),'Replay lacks a selected request'
    metadata=dict(manifest=str(args.manifest),manifest_SHA256=digest(args.manifest),subset=args.subset,
        order_seed=args.order_seed,shuffled=not args.serial)
    if replay is not None:
        metadata.update(replay_reference=str(args.replay_from),replay_reference_SHA256=digest(args.replay_from),
            caveat='Equal output length is not equal answer text. This is not Natural EOS.')
    asyncio.run(execute(rows,args.urls.split(','),args.run,args.qps,args.seed,
        replay if replay is not None else args.output_count,args.serial,metadata=metadata,
        observe_host=args.observe_client_host,freeze_static_heap=args.freeze_client_static_heap,
        defer_raw=args.defer_client_raw))


if __name__=='__main__':main()
