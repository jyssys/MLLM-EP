"""Retain qualified numerical and transport code. Add bounded span evidence."""
import collections
from dataclasses import replace
from .common import *

INSTALLED=False


def install():
    global INSTALLED
    if INSTALLED:return
    INSTALLED=True
    for source,expected in json.loads(os.environ.get('ATOMIC_SOURCE_HASHES','{}')).items():
        assert digest(source)==expected,'Runtime source changed before worker startup: '+source
    cfg=config();redirect()
    from ..looped_pd_bottleneck_crossover import serving_worker as previous
    previous.ROOT=ROOT;previous.config=config;previous.install()
    from ..looped_pd_abc import worker as inherited
    configure=inherited.configuration
    def configuration(model,width,chunk,graphs=False,asynchronous=False):
        result=configure(model,width,chunk,graphs,asynchronous)
        result['scheduler_config']=replace(result['scheduler_config'],max_num_batched_tokens=cfg['token_budget'],
            prefill_chunk_size=cfg['chunk'],wavefront_chunked_prefill=cfg.get('wavefront',False),
            mode=cfg.get('scheduler_mode','refill'))
        result['cache_config']=replace(result['cache_config'],num_blocks=cfg['kv_blocks'],enable_prefix_caching=False)
        result['execution_config']=replace(result['execution_config'],cuda_graph_max_batch_size=cfg.get('graph_rows',cfg['capacity']))
        return result
    inherited.configuration=configuration
    from vllm_rlt.pd.engine import PDEngine
    initialize=PDEngine.__init__
    def initialize_pd(self,*args,**kwargs):
        if 'prefill_scheduler_config' in kwargs:
            kwargs['prefill_scheduler_config']=replace(kwargs['prefill_scheduler_config'],max_num_seqs=cfg['P_slots'])
            kwargs['decode_scheduler_config']=replace(kwargs['decode_scheduler_config'],max_num_seqs=cfg['D_slots'])
            if cfg.get('D_token_budget'):
                kwargs['decode_scheduler_config']=replace(kwargs['decode_scheduler_config'],
                    max_num_batched_tokens=cfg['D_token_budget'],prefill_chunk_size=cfg['D_token_budget'])
        if cfg.get('D_kv_blocks') and 'decode_cache_config' in kwargs:
            kwargs['decode_cache_config']=replace(kwargs['decode_cache_config'],num_blocks=cfg['D_kv_blocks'])
        if cfg.get('native_transport_limits'):
            # Existing supported PDConfig fields only. No new transport,
            # ownership rule, tensor approximation, or scheduler is added.
            from vllm_rlt.pd.config import PDConfig
            kwargs['pd_config']=replace(kwargs.get('pd_config') or PDConfig(),**cfg['native_transport_limits'])
        return initialize(self,*args,**kwargs)
    PDEngine.__init__=initialize_pd
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase
    submit=ModelRunner.submit;close=StageEngine.close
    prompt_spans=[];histogram=collections.Counter();time_histogram=collections.Counter();roles={}
    transport_CPU=collections.Counter();transport_waits=collections.Counter();transport_samples=[]
    if cfg.get('observe_transport'):
        from vllm_rlt.pd.transport import NixlConnector
        from vllm_rlt.pd.connector import PDStageConnector
        for method in ('submit','poll'):
            original=getattr(NixlConnector,method)
            def observed_transport(self,*args,_original=original,_method=method,**kwargs):
                start=time.perf_counter()
                try:return _original(self,*args,**kwargs)
                finally:
                    transport_CPU[_method+'_host_s']+=time.perf_counter()-start
                    transport_CPU[_method+'_calls']+=1
            setattr(NixlConnector,method,observed_transport)
        available=PDStageConnector.can_schedule;clock=[0.]
        def observed_available(self,request):
            result=available(self,request)
            if self.exports(request) and not result:
                transfer=self.transfers.get(request.request_id)
                if transfer is not None:
                    if transfer.chunks:transport_waits['queued_wire_fragments']+=1
                    if transfer.last_event is not None and not transfer.last_event.query():transport_waits['producer_event_not_ready']+=1
                    if self.transport.inflight_bytes>=self.config.max_inflight_bytes:transport_waits['global_wire_credit_full']+=1
            now=time.monotonic()
            if now-clock[0]>=.25:
                clock[0]=now;transport_samples.append(dict(time=now,inflight_bytes=self.transport.inflight_bytes,
                    max_inflight_bytes=self.config.max_inflight_bytes,pending_handles=len(self.transport.pending),
                    queued_wire_fragments=sum(len(t.chunks) for t in self.transfers.values()),
                    blocked_schedule_calls=dict(transport_waits),host_transport=dict(transport_CPU)))
            return result
        PDStageConnector.can_schedule=observed_available
    def observed_submit(self,scheduled_batch=None,*,batch):
        now=time.monotonic()
        histogram[(batch.operation.value,batch.num_rows,batch.loop_count,len(batch.spans))]+=1
        time_histogram[(int(now*2)/2,batch.operation.value,batch.num_rows,batch.loop_count,len(batch.spans))]+=1
        for index,span in enumerate(batch.spans):
            request=span.request
            phase=scheduled_batch.phase_for(scheduled_batch.requests[index]) if scheduled_batch is not None else request.phase
            if phase not in (RequestPhase.PREFILL,RequestPhase.PREFILLING_PRELUDE,RequestPhase.PREFILLING_RECURRENT):continue
            n=len(request.prompt_token_ids)
            if cfg.get('atomic_prompt'):
                assert span.token_start==0 and span.token_count==n,('ATOMIC_PROMPT_WAS_SPLIT',span.token_start,span.token_count,n)
                if phase==RequestPhase.PREFILL:assert batch.loop_count==cfg['depth']
            prompt_spans.append(dict(time=now,request_id=request.request_id,seed=request.sampling_params.seed,
                phase=phase.value,start=span.token_start,count=span.token_count,prompt_tokens=n,
                loop_count=batch.loop_count,depth=span.depth,batch_rows=batch.num_rows,batch_requests=len(batch.spans)))
        if cfg.get('nsys'):
            import torch
            torch.cuda.nvtx.range_push('APD_FORWARD|'+batch.operation.value+'|rows='+str(batch.num_rows)+'|loops='+str(batch.loop_count))
        try:return submit(self,scheduled_batch,batch=batch)
        finally:
            if cfg.get('nsys'):
                import torch
                torch.cuda.nvtx.range_pop()
    ModelRunner.submit=observed_submit
    def close_evidence(self):
        self.model_runner.synchronize()
        if cfg.get('observe_transport'):
            write(ROOT/'raw'/cfg['name']/f'transport_observation_{os.getpid()}.json',dict(role=self.stage_config.role,
                host_CPU_observation=dict(transport_CPU),blocked_schedule_calls=dict(transport_waits),samples=transport_samples,
                scope='Diagnostic host wall spans and scheduling reasons. Includes warm-up. Counters can overlap and are not additive E2E or GPU DMA time.'))
        write(ROOT/'raw'/cfg['name']/f'atomic_spans_{os.getpid()}.json',dict(role=self.stage_config.role,
            physical_visible_device=str(self.model_runner.device),atomic_requested=cfg.get('atomic_prompt'),
            prompt_spans=prompt_spans,
            execution_histogram=[dict(operation=k[0],logical_rows=k[1],loop_count=k[2],requests=k[3],submissions=v) for k,v in histogram.items()],
            time_histogram=[dict(time=k[0],operation=k[1],logical_rows=k[2],loop_count=k[3],requests=k[4],submissions=v) for k,v in time_histogram.items()],
            scope='Actual ExecutionBatch before ModelRunner submission. Logical rows are not padded GEMM rows. Includes warm-up.'))
        return close(self)
    StageEngine.close=close_evidence
