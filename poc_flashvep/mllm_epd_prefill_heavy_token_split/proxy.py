"""Minimal HTTP stage coordinator for installed official EC and NIXL connectors."""
import argparse
import asyncio
import copy
import hashlib
from contextlib import asynccontextmanager
import json
import time

from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,StreamingResponse
import aiohttp

from .common import ROOT,append,write,digest
from .token_split import select_prefix as token_prefix,POLICY_KEY


def application(mode,encoder,prefill,decoders,name,encoder_policy='whole_request',processor_handoff='none',release_encoder_ipc=False,token_suffix_fraction=0.,token_prefix_tokens=0):
    assert processor_handoff in ('none','full','grid_only')
    assert processor_handoff=='none' or mode in ('epd','e_pd') and encoder_policy=='whole_request'
    assert not release_encoder_ipc or processor_handoff!='none' and mode in ('epd','e_pd')
    assert not (token_suffix_fraction or token_prefix_tokens) or mode=='epd' and processor_handoff!='none'
    assert 0<=token_suffix_fraction<1 and token_prefix_tokens>=0
    assert not (token_suffix_fraction and token_prefix_tokens)
    cursor=0;active={url:0 for url in decoders};session=None
    state=dict(received=0,encoder_pending=0,prefill_pending=0,decode_pending=0,completed=0,failed=0)
    @asynccontextmanager
    async def lifespan(app):
        nonlocal session
        session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200,sock_read=3600),
            connector=aiohttp.TCPConnector(limit=0,limit_per_host=0,keepalive_timeout=1),trust_env=False)
        yield
        await session.close()
    app=FastAPI(lifespan=lifespan)

    def headers(base,stage):
        value={'x-request-id':stage}
        if processor_handoff!='none':
            value.update({'x-campaign-metadata-key':hashlib.sha256(base.encode()).hexdigest(),
                          'x-campaign-metadata-mode':processor_handoff})
        return value

    @app.get('/health')
    async def health():return dict(status='ready',mode=mode,name=name,**state)

    @app.get('/campaign_state')
    async def campaign_state():return dict(time=time.time(),**state,active_http_streams=active,
        scope='Coordinator HTTP state. Not engine decode-ready or GPU activity.')

    async def encode(body,request_id,trace):
        parts=[part for message in body['messages'] for part in message.get('content',[])
            if isinstance(part,dict) and part.get('type')=='image_url']
        assert parts,'Image-only workload requires original media'
        async def child(index,part):
            payload=dict(model=body['model'],messages=[dict(role='user',content=[part])],
                temperature=0,max_tokens=1,stream=False)
            begin=time.monotonic()
            async with session.post(encoder+'/v1/chat/completions',json=payload,
                headers=headers(request_id,request_id+':E:'+str(index))) as reply:
                content=await reply.json()
                if reply.status!=200:raise RuntimeError(('encoder',reply.status,str(content)[:1500]))
            return dict(index=index,begin=begin,end=time.monotonic(),usage=content.get('usage'),
                response=content)
        state['encoder_pending']+=1;trace['E_submit']=time.monotonic()
        try:
            if encoder_policy=='whole_request':
                payload=copy.deepcopy(body);payload.update(max_tokens=1,stream=False)
                if token_suffix_fraction or token_prefix_tokens:
                    payload['kv_transfer_params']={POLICY_KEY:dict(
                        suffix_fraction=token_suffix_fraction,fixed_prefix=token_prefix_tokens)}
                payload.pop('stream_options',None)
                if 'max_completion_tokens' in payload:payload['max_completion_tokens']=1
                begin=time.monotonic()
                async with session.post(encoder+'/v1/chat/completions',json=payload,
                    headers=headers(request_id,request_id+':E:full')) as reply:
                    output=await reply.json()
                    if reply.status!=200:raise RuntimeError(('encoder',reply.status,str(output)[:1500]))
                trace['E_items']=[dict(begin=begin,end=time.monotonic(),image_count=len(parts),
                    policy=encoder_policy,usage=output.get('usage'),response=output)]
                return
            results=await asyncio.gather(*(child(i,p) for i,p in enumerate(parts)),return_exceptions=True)
            trace['E_items']=[r if isinstance(r,dict) else dict(index=i,error=type(r).__name__+': '+str(r))
                for i,r in enumerate(results)]
            errors=[r for r in results if isinstance(r,BaseException)]
            if errors:raise errors[0]
        finally:
            state['encoder_pending']-=1
            trace['E_complete']=time.monotonic()

    async def prefill_request(body,request_id,trace):
        request=copy.deepcopy(body);request['stream']=False;request['max_tokens']=1
        request.pop('stream_options',None)
        if 'max_completion_tokens' in request:request['max_completion_tokens']=1
        request['kv_transfer_params']=dict(do_remote_decode=True,do_remote_prefill=False,
            remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None)
        plan=None
        if token_suffix_fraction or token_prefix_tokens:
            length=trace['E_items'][0]['usage']['prompt_tokens']
            prefix=token_prefix(length,token_suffix_fraction,token_prefix_tokens)
            plan=dict(prefix_tokens=prefix,full_prompt_tokens=length)
            request['kv_transfer_params']['campaign_token_split_v1']=plan
            trace['token_partition']=dict(plan,suffix_tokens=length-prefix,
                P_sample='Discarded. All original prompt tokens are processed before D sampling.')
        state['prefill_pending']+=1;trace['P_submit']=time.monotonic()
        try:
            async with session.post(prefill+'/v1/chat/completions',json=request,
                headers=headers(request_id,request_id+':P')) as reply:
                result=await reply.json()
                if reply.status!=200:raise RuntimeError(('prefill',reply.status,str(result)[:2000]))
        finally:state['prefill_pending']-=1
        trace['P_ready']=time.monotonic();trace['P_response']=result
        params=result.get('kv_transfer_params')
        assert params and params.get('remote_block_ids') is not None,('Missing actual NIXL handoff',result)
        if plan is not None:
            assert params.get('remote_num_tokens')==plan['prefix_tokens'],('Invalid P prefix export',params,plan)
            params['campaign_token_split_v1']=plan
        output=copy.deepcopy(body);output['kv_transfer_params']=params
        return output

    async def release_encoder(request_id,trace):
        if not release_encoder_ipc:return
        begin=time.monotonic()
        async with session.post(encoder+'/campaign/ec_ipc/release',
            json=dict(key=hashlib.sha256(request_id.encode()).hexdigest())) as reply:
            value=await reply.json()
            assert reply.status==200,('EC ownership release failed',reply.status,str(value)[:1000])
        assert len(value)==1
        trace['E_embedding_release']=dict(begin=begin,end=time.monotonic(),state=value[0])

    @app.post('/v1/chat/completions')
    async def completion(request:Request):
        nonlocal cursor
        body=await request.json();request_id=request.headers.get('x-request-id')
        assert request_id,'Campaign requests must have an explicit request ID'
        trace=dict(runtime_request_id=request_id,mode=mode,name=name,received=time.monotonic(),
            wall_received=time.time(),state_at_arrival=dict(state),status='FAILED_EXECUTION',processor_handoff=processor_handoff)
        state['received']+=1
        target=decoders[cursor%len(decoders)];cursor+=1
        try:
            if mode in ('e_pd','epd'):
                await encode(body,request_id,trace)
            if mode in ('epd','pd'):body=await prefill_request(body,request_id,trace)
            trace['D_submit']=time.monotonic();trace['D_target']=target
            state['decode_pending']+=1;active[target]+=1
            reply=await session.post(target+'/v1/chat/completions',json=body,
                headers=headers(request_id,request_id+':D'))
            trace['D_headers']=time.monotonic()
            if reply.status!=200:
                error=await reply.text();reply.release()
                raise RuntimeError(('decode',reply.status,error[:3000]))
            if not body.get('stream'):
                output=await reply.json();reply.release()
                await release_encoder(request_id,trace)
                state['decode_pending']-=1;active[target]-=1;state['completed']+=1
                trace.update(status='MEASURED_ONLINE',finished=time.monotonic(),D_response=output)
                append(ROOT/'raw'/('proxy_'+name+'.jsonl'),trace)
                return JSONResponse(output)
            async def stream():
                try:
                    first=True
                    async for block in reply.content.iter_any():
                        if first:trace['D_first_SSE_bytes']=time.monotonic();first=False
                        yield block
                    trace['status']='MEASURED_ONLINE';state['completed']+=1
                except BaseException as error:
                    trace['error']=type(error).__name__+': '+str(error);state['failed']+=1
                    raise
                finally:
                    reply.release();state['decode_pending']-=1;active[target]-=1
                    try:
                        if trace['status']=='MEASURED_ONLINE':await release_encoder(request_id,trace)
                    except BaseException as error:
                        trace.update(status='FAILED_EXECUTION',error='EC_RELEASE: '+type(error).__name__+': '+str(error))
                        state['completed']-=1;state['failed']+=1
                        raise
                    finally:
                        trace['finished']=time.monotonic()
                        append(ROOT/'raw'/('proxy_'+name+'.jsonl'),trace)
            return StreamingResponse(stream(),media_type='text/event-stream')
        except Exception as error:
            if trace.get('D_submit') and active[target]>0:state['decode_pending']-=1;active[target]-=1
            trace['error']=type(error).__name__+': '+str(error);trace['finished']=time.monotonic()
            append(ROOT/'raw'/('proxy_'+name+'.jsonl'),trace);state['failed']+=1
            return JSONResponse(dict(error=trace['error']),status_code=502)
    return app


def main():
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['agg','e_pd','epd','pd'],required=True)
    p.add_argument('--encoder',default='http://127.0.0.1:18400');p.add_argument('--prefill',default='http://127.0.0.1:18401')
    p.add_argument('--decoders',default='http://127.0.0.1:18402');p.add_argument('--port',type=int,default=18403)
    p.add_argument('--name',required=True)
    p.add_argument('--processor-handoff',choices=['none','full','grid_only'],default='none')
    p.add_argument('--release-encoder-ipc',action='store_true',help='Release only after successful D completion and clone ACKs')
    p.add_argument('--token-suffix-fraction',type=float,default=0.)
    p.add_argument('--token-prefix-tokens',type=int,default=0,help='Fixed boundary for correctness controls only')
    p.add_argument('--encoder-policy',choices=['whole_request','image_fanout'],default='whole_request');args=p.parse_args()
    write(ROOT/'configs'/('proxy_'+args.name+'.json'),dict(args=vars(args),source_SHA256=digest(__file__),
        policy='Default engine continuous scheduler. No source credit or artificial D admission cap.',
        EC_transport='Campaign CUDA IPC with original E tensors' if args.release_encoder_ipc else 'Installed ECExampleConnector CPU staging through per-run tmpfs',
        payload='Original media and full prompt forwarded. No new-version metadata rewrite.',
        cache='mm_processor_cache_gb=1 preserves per-request UUIDs. Unique media UUIDs prevent cross-request reuse in every baseline.'))
    uvicorn.run(application(args.mode,args.encoder,args.prefill,args.decoders.split(','),args.name,args.encoder_policy,args.processor_handoff,args.release_encoder_ipc,args.token_suffix_fraction,args.token_prefix_tokens),
        host='127.0.0.1',port=args.port,log_level='warning')


if __name__=='__main__':main()
