"""Whole-request baseline. Decide after real encode, never split or approximate a prompt."""
import argparse
import copy
import hashlib
from contextlib import asynccontextmanager
import time

import aiohttp
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,StreamingResponse

from .common import ROOT,append,digest,write
from .idle_admission import IdleAdmission,FirstToken


def application(name,encoder,prefill,decoder,policy='idle_after_encode',correctness_only=False,capture=False,defer_release=False):
    assert policy in ['never','idle_after_encode','always_correctness']
    assert policy!='always_correctness' or correctness_only
    assert not (capture or defer_release) or correctness_only and policy=='always_correctness'
    gate=IdleAdmission();session=None
    state=dict(received=0,encoder_pending=0,prefill_pending=0,decode_pending=0,completed=0,failed=0)
    @asynccontextmanager
    async def lifespan(app):
        nonlocal session
        session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200,sock_read=3600),
            connector=aiohttp.TCPConnector(limit=0,limit_per_host=0,keepalive_timeout=1),trust_env=False)
        yield
        await session.close()
    app=FastAPI(lifespan=lifespan)

    def headers(base,stage):return {'x-request-id':base+':'+stage,
        'x-campaign-metadata-key':hashlib.sha256(base.encode()).hexdigest(),'x-campaign-metadata-mode':'grid_only'}

    async def post(url,body,request_headers=None):
        async with session.post(url,json=body,headers=request_headers) as reply:
            value=await reply.json()
            if reply.status!=200:raise RuntimeError((url,reply.status,str(value)[:1800]))
            return value

    @app.get('/health')
    async def health():return dict(status='ready',name=name,policy=policy,**state)

    @app.get('/campaign_state')
    async def status():return dict(time=time.time(),**state,continuation_admission=gate.state(),
        scope='Coordinator HTTP state. Not physical idle, exact P GPU queue, or a predicted safe window.')

    @app.post('/v1/chat/completions')
    async def completion(request:Request):
        body=await request.json();base=request.headers.get('x-request-id');assert base
        assert body.get('return_token_ids') is True
        assert body.get('stream') or body.get('max_tokens')==1
        assert not body.get('kv_transfer_params'),'Whole baseline accepts original full requests, not a hidden split plan'
        trace=dict(runtime_request_id=base,mode='whole_deflection',name=name,received=time.monotonic(),
            wall_received=time.time(),state_at_arrival=dict(state),status='FAILED_EXECUTION',
            evidence='CORRECTNESS_ONLY' if correctness_only else 'MEASURED_ONLINE runtime; task quality pending')
        state['received']+=1;entered=False;counted=False;probe_open=False;reply=None
        async def close(failed):
            nonlocal entered,counted,probe_open
            if probe_open:
                trace['D_correctness_capture_end']=await post(decoder+'/campaign/llm_probe/end',{})
                probe_open=False
            if counted:state['decode_pending']-=1;counted=False
            if entered:await gate.leave(base,failed=failed);entered=False
        async def release():
            if defer_release:
                trace['E_release']='DEFERRED_FOR_SAME_E_CORRECTNESS_REFERENCE'
                return
            trace['E_embedding_release']=await post(encoder+'/campaign/ec_ipc/release',
                dict(key=hashlib.sha256(base.encode()).hexdigest()))
        try:
            assert gate.failure is None,'A previous D failure disabled admission'
            encoded=copy.deepcopy(body);encoded.update(max_tokens=1,stream=False);encoded.pop('stream_options',None)
            if 'max_completion_tokens' in encoded:encoded['max_completion_tokens']=1
            trace['E_submit']=time.monotonic();state['encoder_pending']+=1
            try:
                value=await post(encoder+'/v1/chat/completions',encoded,headers(base,'E:full'))
                trace['E_response']=value
            finally:state['encoder_pending']-=1;trace['E_complete']=time.monotonic()
            selected=policy=='always_correctness' or policy=='idle_after_encode' and \
                state['prefill_pending']>0 and not gate.active and not gate.pending and gate.prefill_owner is None
            trace['deflected']=selected
            trace['decision']=dict(policy=policy,after_encode=True,P_HTTP_outstanding=state['prefill_pending'],
                D_HTTP_outstanding=state['decode_pending'],admission=gate.state(),selected=selected,
                scope='Current state after actual online encode. No future EOS or idle duration.')
            output=copy.deepcopy(body)
            if selected:
                trace['P_stage']='NOT_APPLICABLE: full prompt goes directly from E to D'
                trace['KV_handoff']='NONE: no P prefix is computed or imported'
            else:
                source=copy.deepcopy(body);source.update(max_tokens=1,stream=False);source.pop('stream_options',None)
                if 'max_completion_tokens' in source:source['max_completion_tokens']=1
                source['kv_transfer_params']=dict(do_remote_decode=True,do_remote_prefill=False,
                    remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None)
                trace['P_submit']=time.monotonic();state['prefill_pending']+=1
                try:value=await post(prefill+'/v1/chat/completions',source,headers(base,'P'))
                finally:state['prefill_pending']-=1
                trace['P_ready']=time.monotonic();trace['P_response']=value
                params=value.get('kv_transfer_params');assert params and params.get('remote_block_ids') is not None
                output['kv_transfer_params']=params
            trace['D_admission']=await gate.enter(base,selected);entered=True
            if capture:
                trace['D_correctness_capture_begin']=await post(decoder+'/campaign/llm_probe/begin',
                    dict(name=base+'_D',max_logits=1,include_decode_inputs=False));probe_open=True
            trace['D_submit']=time.monotonic();state['decode_pending']+=1;counted=True
            reply=await session.post(decoder+'/v1/chat/completions',json=output,headers=headers(base,'D'))
            if reply.status!=200:raise RuntimeError(('D',reply.status,(await reply.text())[:1800]))
            if not output.get('stream'):
                value=await reply.json();reply.release()
                assert any(c.get('token_ids') for c in value.get('choices',[]))
                await gate.first_token(base);await close(False);await release()
                state['completed']+=1;trace.update(status='MEASURED_ONLINE',finished=time.monotonic())
                append(ROOT/'raw'/f'proxy_{name}.jsonl',trace)
                return JSONResponse(value)
            async def stream():
                detector=FirstToken()
                try:
                    async for block in reply.content.iter_any():
                        if detector.feed(block):
                            trace['D_first_generated_token']=time.monotonic();await gate.first_token(base)
                        yield block
                    assert detector.seen
                    trace['status']='MEASURED_ONLINE';state['completed']+=1
                except BaseException as error:
                    trace['error']=type(error).__name__+': '+str(error);state['failed']+=1
                    raise
                finally:
                    reply.release()
                    try:
                        await close(trace['status']!='MEASURED_ONLINE')
                        if trace['status']=='MEASURED_ONLINE':await release()
                    except BaseException as error:
                        trace.update(status='FAILED_EXECUTION',error='CLOSE: '+type(error).__name__+': '+str(error))
                        raise
                    finally:
                        trace['finished']=time.monotonic();append(ROOT/'raw'/f'proxy_{name}.jsonl',trace)
            return StreamingResponse(stream(),media_type='text/event-stream')
        except BaseException as error:
            if reply is not None:reply.release()
            await close(True);state['failed']+=1
            trace.update(error=type(error).__name__+': '+str(error),finished=time.monotonic())
            append(ROOT/'raw'/f'proxy_{name}.jsonl',trace)
            if not isinstance(error,Exception):raise
            return JSONResponse(dict(error=trace['error']),status_code=502)
    return app


def main():
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--port',type=int,default=18849)
    p.add_argument('--policy',choices=['never','idle_after_encode','always_correctness'],default='idle_after_encode')
    p.add_argument('--correctness-only',action='store_true');p.add_argument('--capture',action='store_true');p.add_argument('--defer-release',action='store_true')
    args=p.parse_args()
    write(ROOT/'configs'/f'whole_proxy_{args.name}.json',dict(time=time.time(),args=vars(args),source_SHA256=digest(__file__),
        EC_transport='Original E CUDA IPC. On-demand descriptors, no unneeded inline consumer.',
        scope='Whole-request full28 baseline, not token split, modality-aware method or Kairos reproduction.',
        safety='Full D prefill begins only with zero prior D streams. New Native D can resume after actual first token.',
        original_requests_unchanged=True))
    uvicorn.run(application(args.name,'http://127.0.0.1:18840','http://127.0.0.1:18841','http://127.0.0.1:18842',
        args.policy,args.correctness_only,args.capture,args.defer_release),host='127.0.0.1',port=args.port,log_level='warning')


if __name__=='__main__':main()
