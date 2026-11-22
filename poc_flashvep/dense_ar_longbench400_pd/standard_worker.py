"""Observe the standard scheduler and install the campaign adapter only for partition arms."""
from .runtime_binding import activate
activate()
import argparse
import collections
import hashlib
import json
import os
import time
from multiprocessing.connection import Client
from .common import ROOT,MODEL,write
from ..dense_ar_bi0_shortanswer_pd.engine_options import options


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--role',required=True);parser.add_argument('--address',required=True)
    args=parser.parse_args();cfg=json.load(open(args.config));role=args.role
    assert os.environ['VLLM_BATCH_INVARIANT']==str(int(cfg['batch_invariant']))
    folder=ROOT/'raw'/cfg['name'];activity=None;activity_left=None;activity_right=None
    if cfg.get('kernel_activity') and role in cfg.get('kernel_activity_roles',('prefill','decode','single','replica0','replica1')):
        from .profile_activity import Activity
        activity_folder=folder/f'activity_{role}';activity_folder.mkdir()
        activity=Activity(activity_folder,include_copies=cfg.get('activity_copies',False),graph_enclosures=cfg.get('graph_enclosures',False),runtime_api=cfg.get('runtime_api_activity',False))
    import torch
    torch.set_num_threads(1)
    if cfg.get('gpu_digest_selftest') and role in ('single','prefill'):
        from .gpu_digest import selftest
        selftest()
    from vllm import LLM,SamplingParams
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from ..dense_ar_bi0_shortanswer_pd.remote_token_bound import install as install_remote_bound
    install_remote_bound(cfg,role)
    from ..dense_ar_bi0_shortanswer_pd.nixl_transport import install as observe_NIXL
    nixl_pending=observe_NIXL(cfg,role)
    from vllm.utils import jit_monitor
    jit_events=[];original_jit=jit_monitor._handle_jit_event
    def observed_jit(*a,**kw):
        jit_events.append(dict(time=time.monotonic(),**kw))
        return original_jit(*a,**kw)
    jit_monitor._handle_jit_event=observed_jit
    info={};query_rows=[];state_rows={};pending_hidden={};replays=[0];active={};batch_serial=[0]
    native_import_states={};native_export_prefix_states={}
    if activity:activity.attach_hooks(cfg,role)
    from .hybrid_teacher import install as install_hybrid_teacher
    install_hybrid_teacher(cfg,role,info)
    last_emitted={};quiescent=[];idle_event=None;idle_left=None
    from .numerical_probe import install as install_numerical_probe
    install_numerical_probe(cfg,role,info)
    from .teacher_stage_path import install as install_teacher_stage_path
    install_teacher_stage_path(cfg,role,info)
    adapter=None
    if cfg.get('axis_kind'):
        if cfg.get('token_stock_model'):
            from .stock_token_adapter import StockTokenAdapter as Adapter
        elif cfg.get('preserve_compiled_full_model'):
            from .compiled_full_adapter import CompiledFullAdapter as Adapter
        else:
            from .adapter import Adapter
        adapter=Adapter(cfg,role,info)
    from .decode_prefill_throttle import install as install_decode_prefill_throttle
    install_decode_prefill_throttle(cfg,role)
    from .remote_setup_deferral import install as install_remote_setup_deferral
    install_remote_setup_deferral(cfg,role)
    from .prefill_completion_guard import install as install_prefill_completion_guard
    install_prefill_completion_guard(cfg,role,info,adapter)
    original_prepare=GPUModelRunner.prepare_inputs
    def prompt_KV(self,batch,index,length):
        positions=torch.arange(length,device='cuda');table=self.block_tables.input_block_tables[0][index]
        planes=[]
        for layer in range(36):
            attn=self.vllm_config.compilation_config.static_forward_context[f'model.layers.{layer}.self_attn.attn']
            block=attn.kv_cache.shape[2]
            assert block==cfg['KV_block_tokens'],('Unexpected KV token axis',attn.kv_cache.shape)
            value=attn.kv_cache[table[positions//block].long(),:,positions%block,:].detach().contiguous()
            assert tuple(value.shape)==(length,8,256) and value.dtype==torch.bfloat16 and value.numel()*value.element_size()==length*4096,(
                'Unexpected packed HND KV value',tuple(value.shape),str(value.dtype))
            schema=folder/f'KV_cache_layout_{role}.json'
            if layer==0 and not schema.exists():
                write(schema,dict(shape=list(attn.kv_cache.shape),strides=list(attn.kv_cache.stride()),dtype=str(value.dtype),
                    token_axis=2,block_tokens=block,valid_per_token_shape=[8,256],valid_bytes_per_token=4096,
                    scope='Actual packed HND layout. All K and V channels are included in each valid-token digest.'))
            if cfg.get('gpu_state_digest'):
                from .gpu_digest import gpu
                planes.append(dict(layer=layer,**gpu(value,verify=cfg.get('verify_GPU_digest_with_CPU',False))))
            else:
                value=value.view(torch.uint8).cpu()
                planes.append(dict(layer=layer,SHA256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),bytes=value.numel()))
        return planes
    def prepare(self,*a,**kw):
        batch=original_prepare(self,*a,**kw);active.update(batch=batch,runner=self)
        batch_serial[0]+=1;prepared_time=time.monotonic()
        for i,rid in enumerate(batch.req_ids):
            external=rid.split('-')[0];metadata=info.get(external)
            if metadata is None:continue
            start=int(batch.num_computed_tokens_np[i]);count=int(batch.num_scheduled_tokens[i])
            query_rows.append(dict(time=prepared_time,batch_id=batch_serial[0],request_id=external,start=start,count=count,
                original_input_tokens=metadata['original_length'],
                logical_phase='PREFILL' if start<metadata['original_length'] else 'DECODE',
                first_D_logit_recompute=role=='decode' and start==metadata['original_length']-1 and count==1,
                actual_batch_queries=int(batch.num_tokens),
                API_prefilling=bool(batch.is_prefilling_np[i]),requests=int(batch.num_reqs),
                padded_queries=int(batch.num_tokens_after_padding)))
        return batch
    GPUModelRunner.prepare_inputs=prepare
    original_attn=GPUModelRunner.prepare_attn
    def prepared_attn(self,batch,*a,**kw):
        result=original_attn(self,batch,*a,**kw)
        if cfg.get('state_capture') and role=='decode':
            for index,rid in enumerate(batch.req_ids):
                external=rid.split('-')[0];metadata=info.get(external)
                if metadata is None or external in state_rows or external in native_import_states:continue
                if metadata.get('D_local_full_prefill'):continue
                length=metadata['original_length'];start=int(batch.num_computed_tokens_np[index])
                assert start>=length-1,'Unexpected original-prompt recomputation in Native decode'
                valid_prefix=length-1
                native_import_states[external]=dict(KV=prompt_KV(self,batch,index,valid_prefix),
                    original_input_tokens=length,valid_prefix_tokens=valid_prefix,first_observed_query=start,
                    scope='Only positions0:L-1 are claimed before the Native last-query recomputation. '
                        'PositionL-1 becomes a valid full-state observation after the D model executes it.')
        return result
    GPUModelRunner.prepare_attn=prepared_attn
    original_replay=CudaGraphManager.run_fullgraph
    def replay(self,*a,**kw):replays[0]+=1;return original_replay(self,*a,**kw)
    CudaGraphManager.run_fullgraph=replay
    if cfg.get('state_capture'):
        original_execute=GPUModelRunner.execute_model
        def execute(self,*a,**kw):
            result=original_execute(self,*a,**kw)
            if kw.get('dummy_run') or torch.cuda.is_current_stream_capturing():return result
            frame=self.execute_model_state
            if frame is None or frame.hidden_states is None:return result
            batch=frame.input_batch
            for i,rid in enumerate(batch.req_ids):
                external=rid.split('-')[0];metadata=info.get(external)
                if metadata is None or external in state_rows:continue
                length=metadata['original_length'];start=int(batch.num_computed_tokens_np[i]);count=int(batch.num_scheduled_tokens[i])
                local_full=metadata.get('D_local_full_prefill',False)
                if role!='decode' or local_full:
                    size=min(count,max(0,length-start))
                    if not size:continue
                    record=pending_hidden.setdefault(external,dict(cursor=0,parts=[]))
                    assert record['cursor']==start
                    offset=int(batch.query_start_loc_np[i])
                    record['parts'].append(frame.hidden_states[offset:offset+size].detach().clone().cpu())
                    record['cursor']+=size
                    if record['cursor']!=length:continue
                else:
                    if start+count<length:continue
                    assert start<=length-1<start+count, 'The full Native prompt state must follow its first last-prompt query'
                planes=prompt_KV(self,batch,i,length)
                row=dict(KV=planes,original_input_tokens=length,first_observed_query=start)
                if role!='decode' or local_full:
                    hidden=torch.cat(pending_hidden.pop(external)['parts'])
                    row['final_hidden_SHA256']=hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    row['query_coverage_end']=length
                    if local_full:
                        tensor_path=folder/f'native_D_local_hidden_{external}.pt'
                        torch.save(dict(hidden_last=hidden[-1:]),tensor_path)
                        row['hidden_tensor_file']=str(tensor_path)
                    if role=='prefill' and not cfg.get('axis_kind'):
                        native_export_prefix_states[external]=dict(KV=prompt_KV(self,batch,i,length-1),
                            valid_prefix_tokens=length-1,original_input_tokens=length)
                else:
                    offset=int(batch.query_start_loc_np[i])+length-1-start
                    hidden=frame.hidden_states[offset:offset+1].detach().clone().cpu()
                    row.update(final_hidden_SHA256=hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                        query_coverage_end=length,scope='Full valid prompt KV after the Native first last-prompt query. '
                            'The separately recorded pre-query import proof excludes the recomputed last position.')
                    tensor_path=folder/f'native_D_recomputed_hidden_{external}.pt'
                    torch.save(dict(hidden_last=hidden),tensor_path);row['hidden_tensor_file']=str(tensor_path)
                state_rows[external]=row
            return result
        GPUModelRunner.execute_model=execute
    channel=Client(args.address,family='AF_UNIX');engine=None;steps=[];queues=[];transfers=[];last_queue=None;sent_counts={};schedule_queues=[];allocation_refusals=[]
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    original_allocate=KVCacheManager.allocate_slots
    def allocated(self,request,*a,**kw):
        result=original_allocate(self,request,*a,**kw)
        if result is None:
            allocation_refusals.append(dict(time=time.monotonic(),request_id=request.request_id,
                free_blocks=self.block_pool.get_num_free_blocks(),
                num_new_tokens=kw.get('num_new_tokens',a[0] if a else None)))
        return result
    KVCacheManager.allocate_slots=allocated
    from vllm.v1.core.sched.scheduler import Scheduler
    original_schedule=Scheduler.schedule
    def scheduled(self,*a,**kw):
        value=original_schedule(self,*a,**kw);now=time.monotonic()
        waiting=list(self.waiting)+list(self.skipped_waiting)
        status=dict(collections.Counter(r.status.name for r in waiting))
        key=(len(self.waiting),len(self.skipped_waiting),len(self.running),tuple(sorted(status.items())))
        if not schedule_queues or key!=schedule_queues[-1]['state_key'] or now-schedule_queues[-1]['time']>=.05:
            snapshot=dict(time=now,state_key=key,waiting_primary=len(self.waiting),
                skipped_waiting=len(self.skipped_waiting),waiting_total=len(waiting),running=len(self.running),
                waiting_status=status,remote_KV_waiting=status.get('WAITING_FOR_REMOTE_KVS',0),
                active_decode=0 if role=='prefill' else sum(r.num_computed_tokens>=info.get(r.request_id.split('-')[0],{}).get('original_length',r.num_prompt_tokens) for r in self.running),
                decode_ready_waiting=0 if role=='prefill' else sum(r.status.name in ('WAITING','PREEMPTED') and r.num_computed_tokens>=info.get(r.request_id.split('-')[0],{}).get('original_length',r.num_prompt_tokens) for r in waiting),
                decode_ready_including_first_query=0 if role=='prefill' else sum(r.status.name in ('WAITING','PREEMPTED') and r.num_computed_tokens>=info.get(r.request_id.split('-')[0],{}).get('original_length',r.num_prompt_tokens)-(1 if role=='decode' else 0) for r in waiting),
                free_KV_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks())
            if cfg.get('quiescent_witness') and role=='decode':
                deadlines=[last_emitted[r.request_id.split('-')[0]]+.034
                    for r in list(self.running)+waiting if r.request_id.split('-')[0] in last_emitted]
                snapshot.update(earliest_worker_decode_deadline=min(deadlines,default=None),
                    output_started_requests=len(deadlines),deadline_scope='Worker emission proxy, not client delivery. Frozen34ms interval, no future arrival knowledge.')
            schedule_queues.append(snapshot)
        return value
    Scheduler.schedule=scheduled
    original_ack=Scheduler._update_from_kv_xfer_finished
    def ack(self,value):
        transfers.append(dict(time=time.monotonic(),finished_sending=list(value.finished_sending or []),
            finished_recving=list(value.finished_recving or [])))
        return original_ack(self,value)
    Scheduler._update_from_kv_xfer_finished=ack
    eos=json.loads((MODEL/'generation_config.json').read_text())['eos_token_id']
    try:
        llm=LLM(**options(cfg,role));engine=llm.llm_engine;runner=engine.model_executor.driver_worker.worker.model_runner
        if callable(nixl_pending):assert not nixl_pending()['startup_errors'],nixl_pending()['startup_errors']
        if adapter is not None:adapter.attach(llm)
        actual=engine.vllm_config
        scheduler=engine.engine_core.engine_core.scheduler
        if role=='decode' and cfg.get('D_running_admission_cap'):
            cap=int(cfg['D_running_admission_cap'])
            assert 1<=cap<=actual.scheduler_config.max_num_seqs
            assert cfg.get('d_compute_throttle'), 'Keep this bounded admission intervention decode-first'
            scheduler.max_num_running_reqs=cap
        from vllm.engine.arg_utils import EngineArgs
        default_B,default_S=EngineArgs.get_batch_defaults(1)
        channel.send(dict(kind='ready',role=role,pid=os.getpid(),uuid=str(torch.cuda.get_device_properties(0).uuid),
            scheduler_B=actual.scheduler_config.max_num_batched_tokens,scheduler_S=actual.scheduler_config.max_num_seqs,
            runtime_max_num_running_requests=scheduler.max_num_running_reqs,
            D_running_admission_cap=cfg.get('D_running_admission_cap') if role=='decode' else None,
            KV_block_tokens=actual.cache_config.block_size,
            UCX_RCACHE_MAX_UNRELEASED=os.environ.get('UCX_RCACHE_MAX_UNRELEASED'),
            chunked_prefill=actual.scheduler_config.enable_chunked_prefill,KV_blocks=runner.kv_cache_config.num_blocks,
            KV_bytes=sum(t.numel()*t.element_size() for t in runner.kv_caches),
            model_layers=len(runner.model.model.layers),
            weight_bytes=sum(p.numel()*p.element_size() for p in runner.model.parameters()),
            parameter_dtypes=sorted({str(p.dtype) for p in runner.model.parameters()}),
            model_forward_module=runner.model.forward.__module__,
            official_device_batch_defaults={str(k):dict(B=v,S=default_S.get(k)) for k,v in default_B.items()},
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
            cudagraph_mode=str(actual.compilation_config.cudagraph_mode),compile_mode=str(actual.compilation_config.mode),
            cudagraph_capture_sizes=actual.compilation_config.cudagraph_capture_sizes,
            options=options(cfg,role),batch_invariant=cfg['batch_invariant']))
        pool=scheduler.kv_cache_manager.block_pool;initial_free=pool.get_num_free_blocks()
        running=True;stop_at=None
        while running:
            while channel.poll():
                command=channel.recv()
                if command['kind']=='stop':stop_at=time.monotonic();break
                if command['kind']=='boundary':
                    query_rows.clear();steps.clear();queues.clear();schedule_queues.clear();transfers.clear();jit_events.clear();allocation_refusals.clear();replays[0]=0
                    if activity:activity_left=activity.stamp();activity_start_host=time.monotonic()
                    quiescent.clear();idle_event=None;idle_left=None
                    channel.send(dict(kind='boundary_ready',role=role));continue
                if command['kind']=='certify_idle':
                    assert role=='decode' and (cfg.get('controlled_pipeline') or cfg.get('d_quiescent_admission') or cfg.get('whole_prefill_deflection'))
                    state=nixl_pending() if callable(nixl_pending) else None
                    clear=not scheduler.requests and not engine.has_unfinished_requests() and state is not None and state['pending_handles']==0 and not state['active_read']
                    left=time.monotonic()
                    if clear:torch.cuda.synchronize()
                    channel.send(dict(kind='idle_certificate',passed=bool(clear),before_sync=left,time=time.monotonic(),
                        pending_NIXL=state,scheduler_requests=len(scheduler.requests),
                        scope='Device-wide completion and no known D work. Valid only while the coordinator dispatches no other D request.'))
                    continue
                if idle_left is not None:
                    quiescent.append(dict(start=idle_left,end=time.monotonic(),ended_by='CONTROL_RECEIPT'))
                    idle_left=None
                idle_event=None
                if adapter is not None:command=adapter.command(command)
                rid=command['request_id'];info[rid]=dict(original_length=command['original_length'],
                    D_local_full_prefill=role=='decode' and not bool((command.get('kv_params') or {}).get('do_remote_prefill')),
                    axis_action=command.get('axis_action',cfg.get('axis_kind')),
                    axis_cut=command.get('axis_cut',cfg.get('axis_cut',0)),
                    axis_full_length=command.get('axis_full_length',command['original_length']),
                    axis_prefix_length=command.get('axis_prefix_length',command['original_length']))
                extra={}
                if role=='prefill':extra['extra_args']=dict(kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False))
                elif role=='decode':extra['extra_args']=dict(kv_transfer_params=command['kv_params'])
                ignore_eos=True if role=='prefill' else command.get('ignore_eos',False)
                params=SamplingParams(max_tokens=1 if role=='prefill' else command['cap'],temperature=0,
                    top_p=1,top_k=-1,ignore_eos=ignore_eos,
                    stop_token_ids=[] if ignore_eos else eos,seed=cfg.get('seed',2026111901),**extra)
                prompt=dict(prompt_token_ids=command['input_ids'])
                if command['kind']=='cleanup_remote':
                    request=engine.input_processor.process_inputs(rid,prompt,params,supported_tasks=engine.get_supported_tasks())
                    request.abort_immediately=True;engine.add_request(rid,request,params)
                else:engine.add_request(rid,prompt,params)
                channel.send(dict(kind='accepted',request_id=rid,time=time.monotonic(),role=role))
            if not running:break
            busy=engine.has_unfinished_requests();begin=time.monotonic();outputs=engine.step();end=time.monotonic()
            if busy or outputs:steps.append(dict(start=begin,end=end,wall_ms=1000*(end-begin),outputs=len(outputs)))
            combined_waiting=list(scheduler.waiting)+list(scheduler.skipped_waiting)
            state=(len(scheduler.waiting),len(scheduler.running),len(scheduler.skipped_waiting))
            if state!=last_queue or not queues or end-queues[-1]['time']>.05:
                queues.append(dict(time=end,waiting=state[0],running=state[1],
                    skipped_waiting=state[2],waiting_total=len(combined_waiting),queue_schema='V2_INCLUDES_SKIPPED_WAITING',
                    waiting_status=dict(collections.Counter(str(r.status) for r in combined_waiting)),
                    remote_KV_waiting=sum(r.status.name=='WAITING_FOR_REMOTE_KVS' for r in combined_waiting),
                    decode_ready_waiting=0 if role=='prefill' else sum(r.status.name in ('WAITING','PREEMPTED') and r.num_computed_tokens>=info.get(r.request_id.split('-')[0],{}).get('original_length',r.num_prompt_tokens) for r in combined_waiting),
                    decode_ready_including_first_query=0 if role=='prefill' else sum(r.status.name in ('WAITING','PREEMPTED') and r.num_computed_tokens>=info.get(r.request_id.split('-')[0],{}).get('original_length',r.num_prompt_tokens)-(1 if role=='decode' else 0) for r in combined_waiting),
                    free_KV_blocks=pool.get_num_free_blocks(),retained_requests=len(scheduler.requests),
                    active_decode=0 if role=='prefill' else sum(r.num_computed_tokens>=info.get(r.request_id.split('-')[0],{}).get('original_length',r.num_prompt_tokens) for r in scheduler.running)))
                last_queue=state
            for result in outputs:
                value=result.outputs[0];ids=list(value.token_ids)
                prior=sent_counts.get(result.request_id,0);sent_counts[result.request_id]=len(ids)
                channel.send(dict(kind='output',role=role,request_id=result.request_id,time=time.monotonic(),
                    count=len(ids),new_ids=ids[prior:],finished=result.finished,output_ids=ids if result.finished else None,finish_reason=value.finish_reason,
                    stop_reason=value.stop_reason,kv_params=result.kv_transfer_params if result.finished else None))
                if len(ids)>prior:last_emitted[result.request_id]=time.monotonic()
            if cfg.get('quiescent_witness') and role=='decode' and not scheduler.requests and not engine.has_unfinished_requests():
                if idle_event is None:idle_event=torch.cuda.Event();idle_event.record()
                if idle_left is None and idle_event.query():idle_left=time.monotonic()
            if not engine.has_unfinished_requests():time.sleep(.0005)
            if activity and activity_left and activity_right is None and not cfg.get('activity_delay_seconds') and time.monotonic()-activity_start_host>=cfg.get('kernel_activity_seconds',10):
                activity_right=activity.stamp();activity.pause()
            if stop_at is not None:
                if not scheduler.requests and pool.get_num_free_blocks()==initial_free:break
                assert time.monotonic()-stop_at<20,'KV retirement did not restore initial credit'
    finally:
        if cfg.get('quiescent_witness') and role=='decode':
            if idle_left is not None:quiescent.append(dict(start=idle_left,end=time.monotonic(),ended_by='WORKER_SHUTDOWN'))
            write(folder/'D_quiescent_witness.json',dict(rows=quiescent,
                evidence='MEASURED_HOST_STATE_WITH_CUDA_COMPLETION',
                scope='No scheduler requests and no unfinished engine work. A CUDA event completed before each left boundary. '
                'The endpoint is observed control receipt, not a predicted future window. This is not an online fit guarantee.'))
        if activity:
            torch.cuda.synchronize()
            if activity_right is None:activity_right=activity.stamp()
            activity.close()
            write(folder/f'activity_window_{role}.json',dict(left=activity_left,right=activity_right,
                scope='Worker measurement-boundary receipt to normal retirement. Compare observer overhead before primary interpretation.'))
        write(folder/f'actual_queries_{role}.json',dict(rows=query_rows,scope='Actual query spans. Partition coverage is recorded separately by the campaign adapter.'))
        write(folder/f'engine_steps_{role}.json',dict(rows=steps,queues=queues,graph_replays=replays[0],
            scope='Host engine-step wall clocks and standard scheduler queues. Not exclusive GPU time.'))
        write(folder/f'scheduler_queues_{role}.json',dict(rows=schedule_queues,
            scope='Scheduler-exit states before the associated GPU/connector work. Includes skipped remote-KV waiting. State-change and50ms snapshots. Host state is not physical idle.'))
        write(folder/f'KV_allocation_refusals_{role}.json',dict(rows=allocation_refusals,
            scope='Read-only observation of native allocator None returns after the measurement boundary. No admission override.'))
        write(folder/f'NIXL_completion_{role}.json',dict(rows=transfers,
            scope='Standard connector completion notifications at scheduler update. Sending completion permits source block retirement. Not pure DMA time.'))
        write(folder/f'JIT_events_{role}.json',dict(rows=jit_events,
            scope='Read-only post-compilation notifications after the measurement boundary. These events do not measure compiler-exclusive time.'))
        if cfg.get('state_capture'):
            write(folder/f'prompt_state_{role}.json',dict(requests=state_rows,evidence='CORRECTNESS_ONLY'))
            if role in ('prefill','decode'):
                write(folder/f'native_valid_prefix_{role}.json',dict(
                    requests=native_export_prefix_states if role=='prefill' else native_import_states,
                    evidence='CORRECTNESS_ONLY',
                    scope='Native proxy positions0:L-1,excluding its recomputed last prompt query. '
                        'Full valid prompt state is recorded after the first D model query. No performance measurement.'))
        if adapter is not None:adapter.close()
        if engine is not None:
            pool=scheduler.kv_cache_manager.block_pool
            write(folder/f'KV_lifetime_{role}.json',dict(initial_free_blocks=initial_free,final_free_blocks=pool.get_num_free_blocks(),
                pending_requests=len(scheduler.requests),passed=pool.get_num_free_blocks()==initial_free and not scheduler.requests,
                scope='Standard connector ACK and normal request retirement. No custom reservation or cohort barrier.'))
            engine.engine_core.shutdown()
        channel.close()


if __name__=='__main__':main()

