"""Minimal HTTP stage coordinator for installed official EC and NIXL connectors."""
import argparse
import asyncio
import copy
import hashlib
from contextlib import asynccontextmanager
import json
import os
import secrets
import time

from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,StreamingResponse
import aiohttp

from . import idle_admission
from .common import ROOT,append,write,digest
from .idle_admission import DECODE_CAPS,FirstToken
from .token_split import select_prefix as token_prefix, PLAN_KEY, POLICY_KEY


def application(mode,encoder,prefill,decoders,name,encoder_policy='whole_request',processor_handoff='none',
        release_encoder_ipc=False,token_suffix_fraction=0.,token_prefix_tokens=0,token_admission='none',
        token_policy='static',decode_batch_one=False,decode_cap=None,late_policy=None):
    assert processor_handoff in ('none','full','grid_only')
    assert processor_handoff=='none' or mode in ('epd','e_pd') and encoder_policy=='whole_request'
    assert not release_encoder_ipc or processor_handoff!='none' and mode in ('epd','e_pd')
    assert not (token_suffix_fraction or token_prefix_tokens) or mode=='epd' and processor_handoff!='none'
    assert 0<=token_suffix_fraction<1 and token_prefix_tokens>=0
    assert not (token_suffix_fraction and token_prefix_tokens)
    assert token_admission in ('none','decode_idle') and token_policy in ('static','idle_fallback')
    assert token_admission=='none' or mode=='epd' and len(decoders)==1
    assert token_policy=='static' or token_admission=='decode_idle' and (token_suffix_fraction or token_prefix_tokens)
    if decode_batch_one:
        assert decode_cap in (None,1)
        decode_cap=1
    assert decode_cap is None or mode=='epd' and len(decoders)==1
    admission=None
    late=None
    if late_policy is not None:
        from ..mllm_epd_h200_modality_aware_dynamic_split.admission import LateAdmission
        from ..mllm_epd_h200_modality_aware_dynamic_split.coordinator import LateCoordinator

        assert mode=='epd' and len(decoders)==1 and release_encoder_ipc and processor_handoff!='none'
        assert not token_suffix_fraction and not token_prefix_tokens and token_admission=='none'
        admission=LateAdmission(decode_cap)
        late=LateCoordinator(admission,late_policy,decoders[0],os.environ['MLLM_EPD_METADATA_PATH'])
    elif decode_cap is not None:
        admission=idle_admission.IdleAdmission(decode_cap=decode_cap)
    elif token_admission=='decode_idle':
        admission=idle_admission.IdleAdmission()
    cursor=0;active={url:0 for url in decoders};session=None
    state=dict(received=0,encoder_pending=0,prefill_pending=0,decode_pending=0,completed=0,failed=0)
    serial_encoder = asyncio.Lock() if os.environ.get('MLLM_EPD_LATE_VERIFY_E_SERIAL')=='1' else None
    correctness_barriers = {}
    correctness_leads = {}
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
        continuation_admission=admission.state() if admission is not None else None,
        scope='Coordinator HTTP state. Not engine decode-ready or GPU activity.')

    if late is not None:
        @app.post('/campaign/late-decision')
        async def late_decision(request:Request):
            nonce=request.headers.get('x-campaign-late-nonce','')
            if not secrets.compare_digest(nonce,os.environ['MLLM_EPD_RUN_NONCE']):
                return JSONResponse({'accepted':False,'reason':'WRONG_RUN'},status_code=403)
            return await late.decide(await request.json(),session)

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
                if trace['token_enabled']:
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
        if trace['token_enabled']:
            length=trace['E_items'][0]['usage']['prompt_tokens']
            prefix=token_prefix(length,token_suffix_fraction,token_prefix_tokens)
            plan=dict(prefix_tokens=prefix,full_prompt_tokens=length)
            request['kv_transfer_params']['campaign_token_split_v1']=plan
            trace['token_partition']=dict(plan,suffix_tokens=length-prefix,
                P_sample='Discarded. All original prompt tokens are processed before D sampling.')
        elif late is not None and not trace.get('late_correctness_native'):
            from ..mllm_epd_h200_modality_aware_dynamic_split.policy import LATE_KEY

            length=trace['E_items'][0]['usage']['prompt_tokens']
            max_output=body.get('max_completion_tokens') or body.get('max_tokens') or 2048
            request['kv_transfer_params'][LATE_KEY]=late.register(request_id,length,max_output,trace)
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
        if late is not None and not trace.get('late_correctness_native'):
            plan=params.get(PLAN_KEY)
            if plan is None:
                trace['late_P_continued']=True
                await late.expire(request_id)
            else:
                prefix=plan['prefix_tokens']
                assert late.prefills[request_id].supports(prefix) and plan['full_prompt_tokens']==length
                assert params.get('remote_num_tokens')==prefix
                trace['token_enabled']=True
                trace['late_P_continued']=False
                trace['token_partition']=dict(plan,suffix_tokens=length-prefix,
                    P_sample='No P token emitted from the completed prefix cut.')
                late.prefills.pop(request_id)
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
        if late is not None and 'x-campaign-late-cut' in request.headers:
            nonce=request.headers.get('x-campaign-late-nonce','')
            if os.environ.get('MLLM_EPD_LATE_CORRECTNESS')!='1' or not secrets.compare_digest(
                    nonce,os.environ['MLLM_EPD_RUN_NONCE']):
                return JSONResponse({'error':'Correctness cut unavailable'},status_code=403)
            trace['late_correctness_cut']=int(request.headers['x-campaign-late-cut'])
            trace['late_correctness_continue']=request.headers.get('x-campaign-late-choice')=='continue'
            trace['late_correctness_native']=request.headers.get('x-campaign-late-choice')=='native'
            boundary=request.headers.get('x-campaign-late-boundary')
            trace['late_correctness_natural']=boundary in ('natural','natural_hold')
            trace['late_correctness_natural_hold']=boundary=='natural_hold'
            cohort=request.headers.get('x-campaign-late-e-cohort')
            if cohort and serial_encoder is not None:
                trace['late_correctness_cohort']=cohort
                lead=request.headers.get('x-campaign-late-e-lead')
                if lead in ('0','1'):
                    trace['late_correctness_lead']=lead=='1'
        requested=bool(token_suffix_fraction or token_prefix_tokens)
        trace['token_enabled']=requested and (token_policy=='static' or
            state['prefill_pending']>0 and not any(active.values()) and admission.prefill_owner is None)
        trace['token_decision']=dict(policy=token_policy,requested=requested,selected=trace['token_enabled'],
            P_HTTP_outstanding=state['prefill_pending'],D_HTTP_outstanding=sum(active.values()),
            admission=admission.state() if admission is not None else None,
            scope='Current state before encode. HTTP P pressure is not an exact GPU queue or a future window.')
        state['received']+=1
        target=decoders[cursor%len(decoders)];cursor+=1
        gate_entered=False;http_counted=False
        async def close_admission(failed=False):
            nonlocal gate_entered,http_counted
            if http_counted:
                state['decode_pending']-=1;active[target]-=1;http_counted=False
            if gate_entered:
                await admission.leave(request_id,failed=failed);gate_entered=False
        try:
            if admission is not None:
                assert admission.failure is None,'A previous D failure disabled new admissions for this experiment'
                assert body.get('return_token_ids') is True,'The admission boundary requires actual generated token IDs'
                assert body.get('stream') or body.get('max_tokens')==1,'Use streaming to release prefill at the first token, not after full decode'
            if mode in ('e_pd','epd'):
                if serial_encoder is None:
                    await encode(body,request_id,trace)
                else:
                    async with serial_encoder:
                        await encode(body,request_id,trace)
                    cohort=trace.get('late_correctness_cohort')
                    if cohort is not None:
                        barrier=correctness_barriers.setdefault(cohort,asyncio.Barrier(8))
                        try:
                            released=await asyncio.wait_for(barrier.wait(),120)
                        except BaseException:
                            await barrier.abort()
                            correctness_barriers.pop(cohort,None)
                            raise
                        if released==0:
                            correctness_barriers.pop(cohort,None)
                        if 'late_correctness_lead' in trace:
                            lead_event=correctness_leads.setdefault(cohort,asyncio.Event())
                            if not trace['late_correctness_lead']:
                                await asyncio.wait_for(lead_event.wait(),120)
            if mode in ('epd','pd'):
                try:
                    body=await prefill_request(body,request_id,trace)
                finally:
                    if trace.get('late_correctness_lead'):
                        correctness_leads[trace['late_correctness_cohort']].set()
                trace['D_at_P_ready']=dict(time=trace['P_ready'],active_http_requests=sum(active.values()),
                    pending_admissions=admission.waiting if admission is not None else 0,
                    suffix_owner=admission.prefill_owner if admission is not None else None,
                    max_active_http_requests=decode_cap,
                    cap_binding=admission.cap_binding if admission is not None else False)
            if admission is not None:
                if late is not None and trace['token_enabled']:
                    trace['D_admission']=await admission.claim(request_id,trace['token_partition']['prefix_tokens'])
                else:
                    trace['D_admission']=await admission.enter(request_id,trace['token_enabled'])
                gate_entered=True
            trace['D_submit']=time.monotonic();trace['D_target']=target
            state['decode_pending']+=1;active[target]+=1;http_counted=True
            reply=await session.post(target+'/v1/chat/completions',json=body,
                headers=headers(request_id,request_id+':D'))
            trace['D_headers']=time.monotonic()
            if reply.status!=200:
                error=await reply.text();reply.release()
                raise RuntimeError(('decode',reply.status,error[:3000]))
            if not body.get('stream'):
                output=await reply.json();reply.release()
                trace['D_http_complete']=time.monotonic()
                if admission is not None:
                    assert any(c.get('token_ids') for c in output.get('choices',[]))
                    await admission.first_token(request_id)
                await release_encoder(request_id,trace)
                await close_admission();state['completed']+=1
                trace.update(status='MEASURED_ONLINE',finished=time.monotonic(),D_response=output)
                append(ROOT/'raw'/('proxy_'+name+'.jsonl'),trace)
                return JSONResponse(output)
            async def stream():
                try:
                    first=True
                    detector=FirstToken() if admission is not None else None
                    async for block in reply.content.iter_any():
                        if first:trace['D_first_SSE_bytes']=time.monotonic();first=False
                        if detector is not None and detector.feed(block):
                            trace['D_first_generated_token']=time.monotonic()
                            await admission.first_token(request_id)
                        yield block
                    assert detector is None or detector.seen,'No generated token reached the admission boundary'
                    trace['D_http_complete']=time.monotonic()
                    trace['status']='MEASURED_ONLINE';state['completed']+=1
                except BaseException as error:
                    trace['error']=type(error).__name__+': '+str(error);state['failed']+=1
                    raise
                finally:
                    reply.release();await close_admission(failed=trace['status']!='MEASURED_ONLINE')
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
        except BaseException as error:
            if late is not None:
                await late.expire(request_id)
            await close_admission(failed=True)
            trace['error']=type(error).__name__+': '+str(error);trace['finished']=time.monotonic()
            append(ROOT/'raw'/('proxy_'+name+'.jsonl'),trace);state['failed']+=1
            if not isinstance(error,Exception):raise
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
    p.add_argument('--token-admission',choices=['none','decode_idle'],default='none')
    p.add_argument('--token-policy',choices=['static','idle_fallback'],default='static')
    decode=p.add_mutually_exclusive_group()
    decode.add_argument('--decode-batch-one',action='store_true',
        help='Admit one D request at a time in P-ready FIFO order through completion')
    decode.add_argument('--decode-cap',choices=('free',*map(str,DECODE_CAPS)),
        help='Limit active D HTTP requests through completion using a P-ready FIFO')
    p.add_argument('--late-policy',choices=('late10','generic','generic_pure','hard','soft','fake_boundary'))
    p.add_argument('--late-cost-model')
    p.add_argument('--late-ratios',type=float,nargs='+',default=(.30,.20,.10,.05))
    p.add_argument('--late-idle-budget-ms',type=float,default=160.)
    p.add_argument('--late-safety-ms',type=float,default=10.)
    p.add_argument('--late-natural-boundaries',action='store_true')
    p.add_argument('--late-natural-hold',action='store_true')
    p.add_argument('--late-shadow',action='store_true')
    p.add_argument('--encoder-policy',choices=['whole_request','image_fanout'],default='whole_request');args=p.parse_args()
    decode_cap=1 if args.decode_batch_one else None if args.decode_cap in (None,'free') else int(args.decode_cap)
    late_policy=None
    if args.late_policy is not None:
        from ..mllm_epd_h200_modality_aware_dynamic_split.policy import load_policy

        late_policy=load_policy(args.late_policy,args.late_cost_model,ratios=tuple(args.late_ratios),
            max_idle_budget_ms=args.late_idle_budget_ms,safety_ms=args.late_safety_ms,
            natural=args.late_natural_boundaries,natural_hold=args.late_natural_hold,shadow=args.late_shadow)
    if args.decode_cap=='free' and (args.token_suffix_fraction or args.token_prefix_tokens) and args.token_admission!='decode_idle':
        p.error('Unrestricted Token Split requires --token-admission decode_idle to isolate the suffix')
    if late_policy is not None:
        policy='Native P-ready admission with a nonwaiting D reservation at a completed P boundary.'
    elif decode_cap is not None:
        policy=f'P-ready FIFO capped at {decode_cap} active D HTTP requests through completion. Suffix starts with an empty D and owns it until the first generated token.'
    elif args.token_admission=='decode_idle':
        policy='Extra suffix starts only with zero D HTTP streams. New Native D waits until the first actual suffix output token.'
    else:
        policy='Default engine continuous scheduler. No source credit or artificial D admission cap.'
    write(ROOT/'configs'/('proxy_'+args.name+'.json'),dict(args=vars(args),source_SHA256=digest(__file__),
        policy=policy,max_active_http_requests=decode_cap,
        EC_transport='Campaign CUDA IPC with original E tensors' if args.release_encoder_ipc else 'Installed ECExampleConnector CPU staging through per-run tmpfs',
        payload='Original media and full prompt forwarded. No new-version metadata rewrite.',
        cache='mm_processor_cache_gb=1 preserves per-request UUIDs. Unique media UUIDs prevent cross-request reuse in every baseline.'))
    uvicorn.run(application(args.mode,args.encoder,args.prefill,args.decoders.split(','),args.name,args.encoder_policy,
        args.processor_handoff,args.release_encoder_ipc,args.token_suffix_fraction,args.token_prefix_tokens,
        args.token_admission,args.token_policy,decode_cap=decode_cap,late_policy=late_policy),
        host='127.0.0.1',port=args.port,log_level='warning')


if __name__=='__main__':main()
