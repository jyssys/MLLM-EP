"""Native vLLM/NIXL Full-Prefill PD and independent colocated replicas."""
import argparse
import collections
import socket
import subprocess
import tempfile
from multiprocessing.connection import Listener,Client
from .common import *


def prompt_rows(result,requests,before):
    rows=[]
    for rid,count in result.num_scheduled_tokens.items():
        request=requests.get(rid)
        if request is None:continue
        length=len(request.prompt_token_ids);start=before.get(rid,0)
        if start<length:rows.append((rid,start,min(count,length-start),length))
    return rows


def llm_options(cfg,role):
    slots=cfg['P_slots'] if role=='prefill' else cfg['D_slots']
    options=dict(model=str(MODEL_PATHS['qwen8']),dtype='bfloat16',max_model_len=cfg['length']+cfg['cap'],
        max_num_seqs=slots,max_num_batched_tokens=cfg['token_budget'],enable_prefix_caching=False,
        enable_chunked_prefill=True,async_scheduling=cfg.get('ar_async_scheduling',False),gpu_memory_utilization=cfg.get('gpu_memory_utilization',.80),
        seed=cfg['seed'],compilation_config={'cudagraph_capture_sizes':[n for n in (1,2,4,8,16,32) if n<=slots]})
    if role!='decode' and cfg['chunk']<cfg['length']:
        # vLLM 0.26 rejects nondefault Concurrent Partial Prefill options.
        # Keep its supported scheduler and use only the per-request threshold.
        options.update(long_prefill_token_threshold=cfg['chunk'])
    if role in ('prefill','decode'):
        options['kv_transfer_config']=dict(kv_connector='NixlConnector',kv_role='kv_producer' if role=='prefill' else 'kv_consumer')
    if cfg.get('ar_enforce_eager'):options['enforce_eager']=True
    if cfg.get('ar_cudagraph_mode'):options['compilation_config']['cudagraph_mode']=cfg['ar_cudagraph_mode']
    if cfg.get('ar_attention_config'):options['attention_config']=cfg['ar_attention_config']
    return options


def stage(cfg,role,address):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_NIXL_SIDE_CHANNEL_HOST']='127.0.0.1'
    os.environ['VLLM_NIXL_SIDE_CHANNEL_PORT']=os.environ['APD_SIDE_PORT']
    import torch
    from vllm import LLM,SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    torch.set_num_threads(1);channel=Client(address,family='AF_UNIX')
    spans=[];samples=[];step_spans=[];native_stats=[];warm_stats=[];done_events=[];allocation_refusals=[];clock=[0.]
    allocate=KVCacheManager.allocate_slots
    def observed_allocate(self,request,*a,**kw):
        value=allocate(self,request,*a,**kw)
        if value is None:allocation_refusals.append(dict(time=time.monotonic(),request_id=request.request_id,
            free_blocks=self.block_pool.get_num_free_blocks(),reason='KV_ALLOCATION_RETURNED_NONE'))
        return value
    from .credits import LEDGER
    def credit_allocate(self,request,*a,**kw):
        external=LEDGER.internal.get(request.request_id)
        other=LEDGER.outstanding(exclude=external)
        if other:
            scheduled=int(a[0]) if a else int(kw.get('num_tokens',1))
            need=(request.num_prompt_tokens+15)//16 if request.num_computed_tokens==0 else (scheduled+15)//16+1
            if self.block_pool.get_num_free_blocks()<other+need:
                allocation_refusals.append(dict(time=time.monotonic(),request_id=request.request_id,
                    free_blocks=self.block_pool.get_num_free_blocks(),reason='RESERVED_FUTURE_KV_CREDIT'))
                return None
        value=observed_allocate(self,request,*a,**kw)
        if value is not None:LEDGER.allocated(request.request_id)
        return value
    KVCacheManager.allocate_slots=credit_allocate
    schedule=Scheduler.schedule
    def observed_schedule(self,*a,**kw):
        observes_prompts=role=='prefill' or role.startswith('replica')
        before={rid:r.num_computed_tokens for rid,r in self.requests.items()} if observes_prompts else {}
        result=schedule(self,*a,**kw)
        if observes_prompts:
            for rid,start,actual,length in prompt_rows(result,self.requests,before):
                if cfg['atomic_prompt'] and role=='prefill':assert start==0 and actual==length,('ATOMIC_AR_PROMPT_SPLIT',rid,start,actual,length)
                spans.append(dict(time=time.monotonic(),request_id=rid,start=start,count=actual,prompt_tokens=length,
                    batch_tokens=sum(result.num_scheduled_tokens.values()),scheduled_requests=len(result.num_scheduled_tokens)))
        now=time.monotonic()
        if now-clock[0]>=.25:
            clock[0]=now;samples.append(dict(time=now,waiting=len(self.waiting),running=len(self.running),KV_usage=self.kv_cache_manager.usage,
                KV_free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),max_running_requests=self.max_num_running_reqs,
                waiting_statuses=dict(collections.Counter(str(r.status) for r in self.waiting))))
        return result
    Scheduler.schedule=observed_schedule
    get_stats=NixlBaseConnectorWorker.get_kv_connector_stats
    def collect(self):
        value=get_stats(self)
        if value is not None:native_stats.append(dict(time=time.monotonic(),data={k:list(v) for k,v in value.data.items()}))
        return value
    NixlBaseConnectorWorker.get_kv_connector_stats=collect
    pop_done=NixlBaseConnectorWorker._pop_done_transfers
    def observe_done(self,transfers):
        result=pop_done(self,transfers)
        if result:done_events.append(dict(time=time.monotonic(),request_ids=sorted(result)))
        return result
    NixlBaseConnectorWorker._pop_done_transfers=observe_done
    llm=LLM(**llm_options(cfg,role));engine=llm.llm_engine
    runner=engine.model_executor.driver_worker.worker.model_runner
    from vllm import envs
    channel.send(dict(kind='ready',role=role,pid=os.getpid(),uuid=str(torch.cuda.get_device_properties(0).uuid),
        unique_model_parameters=sum(p.numel() for p in runner.model.parameters()),weight_bytes=sum(p.numel()*p.element_size() for p in runner.model.parameters()),
        KV_blocks=runner.kv_cache_config.num_blocks,KV_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),
        KV_dtype=str(runner.kv_cache_dtype),engine='Native vLLM 0.26/NIXL',model_revision=MODEL_REVISION,options=llm_options(cfg,role),
        VLLM_BATCH_INVARIANT=cfg.get('ar_batch_invariant',False),effective_VLLM_BATCH_INVARIANT=envs.VLLM_BATCH_INVARIANT,
        effective_BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        preferred_blas_library=str(torch.backends.cuda.preferred_blas_library())))
    from .credits import LEDGER
    running=True;profile_marker_open=False;measuring=False;feedback_clock=0.
    request_cuts={}
    scheduler=engine.engine_core.engine_core.scheduler
    try:
        while running:
            while channel.poll():
                command=channel.recv()
                if command['kind']=='stop':running=False;break
                if command['kind']=='measurement_begin':
                    measuring=True
                    from vllm.distributed.kv_transfer import get_kv_transfer_group,has_kv_transfer_group
                    if has_kv_transfer_group():get_kv_transfer_group().get_kv_connector_stats()
                    warm_stats.extend(native_stats);native_stats.clear();done_events.clear();allocation_refusals.clear()
                    if cfg.get('nsys_P_only') and role=='prefill':
                        torch.cuda.nvtx.range_push('APD_MEASURED_WINDOW');profile_marker_open=True
                        write(ROOT/'raw'/cfg['name']/'P_profile_anchor.json',dict(time=time.monotonic(),pid=os.getpid(),
                            scope='Monotonic clock directly after the child NVTX marker. Align this P-only trace with the parent Serving window.'))
                    channel.send(dict(kind='measurement_ready',time=time.monotonic()));continue
                if command['kind']=='reserve':
                    assert role=='decode'
                    free=scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                    active_requests=list(scheduler.requests.values())
                    future_decode_blocks=sum((max(0,(r.sampling_params.max_tokens if r.sampling_params else 0)-r.num_output_tokens)+15)//16 for r in active_requests)
                    accepted=LEDGER.reserve(command['generation'],command['cut'],command['requests'],
                        command['blocks_each'],free,decode_reserve=future_decode_blocks,
                        max_pending=cfg.get('max_reserved_requests',16))
                    channel.send(dict(kind='reservation_result',request_id=command['generation'],accepted=accepted,
                        cut=command['cut'],KV_free_blocks=free,reserved_blocks=LEDGER.outstanding(),
                        time=time.monotonic(),generation=command['generation']))
                    continue
                if command['kind']=='commit':
                    LEDGER.commit(command['generation'])
                    channel.send(dict(kind='commit_result',request_id=command['generation'],generation=command['generation'],
                        accepted=True,time=time.monotonic()))
                    continue
                if command['kind']=='cancel_reservation':
                    if command['generation'] in LEDGER.reservations:LEDGER.cancel(command['generation'])
                    channel.send(dict(kind='reservation_cancelled',request_id=command['generation'],generation=command['generation'],time=time.monotonic()))
                    continue
                extra={}
                request_cuts[command['request_id']]=int(command.get('layer_cut',36)) if role=='prefill' else int((command.get('kv_params') or {}).get('dense_layer_cut',36))
                if role in ('prefill','decode'):
                    extra['extra_args']=dict(kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,dense_layer_cut=request_cuts[command['request_id']]) if role=='prefill' else command['kv_params'])
                params=SamplingParams(max_tokens=1 if role=='prefill' else command['cap'],temperature=0,ignore_eos=True,seed=command['seed'],**extra)
                engine.add_request(command['request_id'],{'prompt_token_ids':command['input_ids']},params)
                channel.send(dict(kind='accepted',request_id=command['request_id'],time=time.monotonic()))
            if not running:break
            start=time.monotonic();outputs=engine.step();end=time.monotonic()
            if outputs:step_spans.append(dict(start=start,end=end,outputs=len(outputs)))
            for result in outputs:
                value=result.outputs[0]
                channel.send(dict(kind='output',role=role,request_id=result.request_id,time=time.monotonic(),
                    count=len(value.token_ids),finished=result.finished,output_ids=list(value.token_ids) if result.finished else None,
                    finish_reason=value.finish_reason,kv_params=(dict(result.kv_transfer_params,dense_layer_cut=request_cuts[result.request_id])
                        if role=='prefill' and result.finished and result.kv_transfer_params else result.kv_transfer_params if result.finished else None),
                    cached_prompt_tokens=getattr(result,'num_cached_tokens',None) if result.finished else None))
                if result.finished:LEDGER.finish(result.request_id)
            if measuring and cfg.get('worker_state_feedback') and time.monotonic()-feedback_clock>=.05:
                feedback_clock=time.monotonic()
                ready=[r for r in scheduler.running if r.num_computed_tokens>=r.num_prompt_tokens]
                channel.send(dict(kind='runtime_state',role=role,time=feedback_clock,
                    running=len(scheduler.running),waiting=len(scheduler.waiting),
                    decode_ready=len(ready),KV_free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                    reserved_future_blocks=LEDGER.outstanding(),
                    remaining_Decode_tokens=[max(0,r.sampling_params.max_tokens-r.num_output_tokens) for r in ready],
                    scope='Host scheduler state after engine.step. Not physical GPU idle or future-arrival knowledge.'))
            if not engine.has_unfinished_requests():time.sleep(.0005)
    finally:
        if profile_marker_open:torch.cuda.nvtx.range_pop()
        from vllm.distributed.kv_transfer import get_kv_transfer_group,has_kv_transfer_group
        if has_kv_transfer_group():get_kv_transfer_group().get_kv_connector_stats()
        write(ROOT/'raw'/cfg['name']/f'native_{role}_{os.getpid()}.json',dict(role=role,prompt_spans=spans,
            queue_samples=samples,host_step_spans=step_spans,NIXL_stats_intervals=native_stats,NIXL_done_events=done_events,
            KV_allocation_refusals=allocation_refusals,warmup_NIXL_stats_intervals=warm_stats,NIXL_measured_boundary_flushed=True,
            scope='Process lifetime includes warm-up. Engine host steps are not GPU kernel durations.'))
        write(ROOT/'raw'/cfg['name']/f'reservation_lifetime_{role}.json',dict(
            outstanding_generations=list(LEDGER.reservations),history=LEDGER.history,
            passed=not LEDGER.reservations,
            scope='Allocator-enforced prospective credits. Physical native blocks are allocated on admission. Cancellation is pre-commit only.'))
        assert not LEDGER.reservations,'A committed credit generation remains live'
        shutdown=getattr(engine.engine_core,'shutdown',None)
        if shutdown:shutdown()
        channel.close()


