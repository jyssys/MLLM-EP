"""Read-only token-emission and API-delivery timestamps. No GPU copies."""
import asyncio
import os
import re
import time

from .common import ROOT,write

API_RECORDER=None


class EngineTimeline:
    def __init__(self,runner,name):
        assert re.fullmatch('[A-Za-z0-9_.-]+',name)
        self.runner=runner;self.rows=[];self.originals=[]
        role=os.environ['MLLM_EPD_ROLE']
        suffix='_gpu'+os.environ['CUDA_VISIBLE_DEVICES'] if role in ('agg','pd') else ''
        self.folder=ROOT/'raw/host_timeline'/name/(role+suffix)
        assert not self.folder.exists();self.folder.mkdir(parents=True)

    def patch(self,obj,key,value):
        self.originals.append((obj,key,key in vars(obj),vars(obj).get(key)))
        setattr(obj,key,value)

    def begin(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        schedule=Scheduler.schedule;update=Scheduler.update_from_output
        execute=self.runner.execute_model;me=self
        def scheduled(scheduler,*args,**kwargs):
            start=time.monotonic_ns();result=schedule(scheduler,*args,**kwargs)
            me.rows.append(dict(kind='schedule',begin_ns=start,end_ns=time.monotonic_ns(),
                requests=dict(result.num_scheduled_tokens),waiting=len(scheduler.waiting),running=len(scheduler.running)))
            return result
        def updated(scheduler,*args,**kwargs):
            start=time.monotonic_ns();result=update(scheduler,*args,**kwargs);end=time.monotonic_ns()
            emitted=[dict(request_id=item.request_id,token_ids=list(item.new_token_ids),
                finished=item.finish_reason is not None) for group in result.values() for item in group.outputs]
            me.rows.append(dict(kind='engine_output',begin_ns=start,end_ns=end,emitted=emitted,
                waiting=len(scheduler.waiting),running=len(scheduler.running)))
            return result
        def executed(output,*args,**kwargs):
            start=time.monotonic_ns();result=execute(output,*args,**kwargs)
            me.rows.append(dict(kind='execute_host',begin_ns=start,end_ns=time.monotonic_ns(),
                requests=dict(output.num_scheduled_tokens)))
            return result
        self.patch(Scheduler,'schedule',scheduled);self.patch(Scheduler,'update_from_output',updated)
        self.patch(self.runner,'execute_model',executed)
        return dict(folder=str(self.folder),scope='Read-only CPU timestamps. Host execute is not GPU-exclusive time.')

    def close(self):
        for obj,key,existed,value in reversed(self.originals):
            if existed:setattr(obj,key,value)
            else:delattr(obj,key)
        write(self.folder/'engine.json',dict(pid=os.getpid(),rows=self.rows,evidence='DIAGNOSTIC',
            scope='CPU scheduler and EngineCore token emission. No model input,output,KV or sampling changes. No GPU synchronization.'))
        return dict(folder=str(self.folder),events=len(self.rows),token_ids=sum(len(x['token_ids']) for r in self.rows for x in r.get('emitted',[])))


class APITimeline:
    def __init__(self):self.active=False

    def install(self):
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.engine.input_processor import InputProcessor
        from vllm.renderers.online_renderer import OnlineRenderer
        from .api_host_scopes import install
        self.host_scope_originals=install(self)
        generate=AsyncLLM.generate;process=InputProcessor.process_inputs;render=OnlineRenderer.render_chat;me=self
        async def generated(engine,prompt,sampling_params,request_id,*args,**kwargs):
            async for out in generate(engine,prompt,sampling_params,request_id,*args,**kwargs):
                if me.active:
                    me.rows.append(dict(kind='api_generate_yield',time_ns=time.monotonic_ns(),
                        request_id=out.request_id,outputs=[dict(index=x.index,token_ids=list(x.token_ids)) for x in out.outputs],finished=out.finished))
                yield out
        def processed(processor,*args,**kwargs):
            if not me.active:return process(processor,*args,**kwargs)
            start=time.monotonic_ns();result=process(processor,*args,**kwargs)
            me.rows.append(dict(kind='input_processor',begin_ns=start,end_ns=time.monotonic_ns(),request_id=result.request_id))
            return result
        async def rendered(renderer,request,*args,**kwargs):
            if not me.active:return await render(renderer,request,*args,**kwargs)
            start=time.monotonic_ns();result=await render(renderer,request,*args,**kwargs)
            me.rows.append(dict(kind='online_renderer',begin_ns=start,end_ns=time.monotonic_ns()))
            return result
        AsyncLLM.generate=generated;InputProcessor.process_inputs=processed;OnlineRenderer.render_chat=rendered

    async def begin(self,name):
        import gc
        import threading
        assert not self.active and re.fullmatch('[A-Za-z0-9_.-]+',name)
        role=os.environ['MLLM_EPD_ROLE'];gpu=os.environ['CUDA_VISIBLE_DEVICES']
        self.folder=ROOT/'raw/host_timeline'/name/f'{role}_api_gpu{gpu}'
        assert not self.folder.exists();self.folder.mkdir(parents=True)
        self.rows=[];self.active=True
        self.gc_inflight={}
        def gc_event(phase,info):
            now=time.monotonic_ns();key=(threading.get_ident(),info['generation'])
            if phase=='start':self.gc_inflight[key]=now
            else:
                begin=self.gc_inflight.pop(key,None)
                if begin is not None and self.active:
                    self.rows.append(dict(kind='api_gc',begin_ns=begin,end_ns=now,
                        generation=key[1],thread_id=key[0],collected=info.get('collected')))
        self.gc_callback=gc_event;gc.callbacks.append(self.gc_callback)
        async def heartbeat():
            while self.active:
                before=time.monotonic_ns();await asyncio.sleep(.005);now=time.monotonic_ns()
                if self.active:self.rows.append(dict(kind='event_loop_heartbeat',time_ns=now,delay_ns=max(0,now-before-5000000)))
        self.task=asyncio.create_task(heartbeat())
        return dict(folder=str(self.folder),scope='API timing only. Five-ms event-loop heartbeat. Observer qualification remains required.')

    async def close(self):
        import gc
        self.active=False;self.task.cancel()
        gc.callbacks.remove(self.gc_callback)
        try:await self.task
        except asyncio.CancelledError:pass
        write(self.folder/'api.json',dict(pid=os.getpid(),rows=self.rows,evidence='DIAGNOSTIC',
            scope='API preprocessing, EngineClient yields, ASGI sends and event-loop lags. No token corrections.'))
        return dict(folder=str(self.folder),events=len(self.rows))


class APITimelineMiddleware:
    def __init__(self,app,recorder=None):
        self.app=app;self.recorder=recorder if recorder is not None else API_RECORDER
        assert self.recorder is not None,'Configure the recorder before the official API app constructs middleware'

    async def __call__(self,scope,receive,send):
        from .processor_handoff import HANDOFF_CONTEXT
        context=None
        if scope['type']=='http' and scope.get('path')=='/v1/chat/completions':
            headers=dict(scope.get('headers',[]))
            key=headers.get(b'x-campaign-metadata-key')
            mode=headers.get(b'x-campaign-metadata-mode')
            if key is not None or mode is not None:
                assert key is not None and mode in (b'full',b'grid_only'), 'Incomplete processed-prompt handoff headers'
                context=HANDOFF_CONTEXT.set((key.decode(),mode.decode()))
        try:return await self.dispatch(scope,receive,send)
        finally:
            if context is not None:HANDOFF_CONTEXT.reset(context)

    async def dispatch(self,scope,receive,send):
        if not self.recorder.active or scope['type']!='http' or scope.get('path')!='/v1/chat/completions':
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]));request_id=headers.get(b'x-request-id',b'').decode()
        from .api_host_scopes import HTTP_REQUEST_ID
        context_token=HTTP_REQUEST_ID.set(request_id)
        self.recorder.rows.append(dict(kind='http_received',time_ns=time.monotonic_ns(),request_id=request_id))
        async def observed_send(message):
            start=time.monotonic_ns();result=await send(message)
            if self.recorder.active and message['type']=='http.response.body':
                self.recorder.rows.append(dict(kind='asgi_body_sent',begin_ns=start,end_ns=time.monotonic_ns(),
                    request_id=request_id,bytes=len(message.get('body',b'')),more_body=message.get('more_body',False)))
            return result
        try:return await self.app(scope,receive,observed_send)
        finally:HTTP_REQUEST_ID.reset(context_token)
