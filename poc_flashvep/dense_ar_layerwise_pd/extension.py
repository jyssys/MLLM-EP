"""Optional native-model layer spans and diagnostic state observations.

The default Native path does not install this extension. The current path
uses persistent NIXL GPU buffers for Hidden/Residual. Early CPU staging is
retained only as a diagnostic. Native NIXL retains its own KV lifetime.
"""
import atexit,hashlib,json,os,time
from pathlib import Path
from .common import ROOT,write


def tensor_hash(tensor):
    import torch
    value=tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def install(cfg,role,notify=None):
    import torch,vllm
    from vllm.sequence import IntermediateTensors
    from vllm.model_executor.models.qwen3 import Qwen3Model,Qwen3ForCausalLM
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    folder=ROOT/'raw'/cfg['name'];mode=cfg.get('layer_mode');cut=int(cfg.get('layer_cut',32))
    assert not mode or mode in ('STATE_REFERENCE','LOCAL_SPLIT','SPAN_ZERO','MIGRATE')
    assert 0<cut<36
    storage=folder/'state_transport';storage.mkdir(exist_ok=True)
    log=(folder/f'extension_{role}_{os.getpid()}.jsonl').open('a',buffering=1)
    active={'batch':None,'runner':None,'submitted':0,'measuring':False}
    imports={};external_ids={};transport={};execution_events=[];sampler_events=[];released_tokens=set()
    def event(kind,**kw):log.write(json.dumps(dict(event=kind,time=time.monotonic(),role=role,**kw))+'\n')
    if cfg.get('prime_detokenizer'):
        from .output_priming import install as install_output_priming
        install_output_priming(event,cfg.get('verify_primed_text',False))
    atexit.register(log.close)
    if cfg.get('execution_events'):
        def export_execution_events():
            torch.cuda.synchronize()
            anchor=active['event_anchor'];lower=active['anchor_host_lower'];upper=active['anchor_host_upper']
            write(folder/f'CUDA_event_enclosures_{role}.json',dict(
                anchor_host_lower=lower,anchor_host_upper=upper,anchor_uncertainty_ms=1000*(upper-lower),
                rows=[dict(meta,elapsed_ms=a.elapsed_time(b),
                    start_monotonic=(lower+upper)/2+anchor.elapsed_time(a)/1000,
                    end_monotonic=(lower+upper)/2+anchor.elapsed_time(b)/1000) for a,b,meta in execution_events],
                scope='CUDA stream-event enclosures. They include launch gaps and waits. They are not compute-kernel union times or exclusive TTFT components.'))
            write(folder/f'CUDA_sampler_enclosures_{role}.json',dict(rows=[dict(meta,
                elapsed_ms=a.elapsed_time(b),start_monotonic=(lower+upper)/2+anchor.elapsed_time(a)/1000,
                end_monotonic=(lower+upper)/2+anchor.elapsed_time(b)/1000) for a,b,meta in sampler_events],
                scope='Sampler stream-event and Host enclosures. They can include GPU waits and CPU gaps. They are not exclusive sampling-kernel time.'))
        atexit.register(export_execution_events)
    if cfg.get('minimal_geometry_counter'):
        active['measured_geometry']={}
        atexit.register(lambda:write(folder/f'measured_geometry_{role}.json',dict(
            counts=active['measured_geometry'],scope='Actual measured input batches. Counts exclude warm-up. These counts are not GPU kernel durations.')))
    if cfg.get('hidden_transport')=='NIXL_GPU' or cfg.get('execution_events') or cfg.get('normal_lifetime_audit'):
        from .hidden_transport import GPUHiddenTransport
        original_llm=vllm.LLM
        def initialized_llm(*args,**kwargs):
            llm=original_llm(*args,**kwargs)
            scheduler=llm.llm_engine.engine_core.engine_core.scheduler
            initial_free=scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            if cfg.get('hidden_transport')=='NIXL_GPU':transport['value']=GPUHiddenTransport(cfg,role,folder,event)
            if cfg.get('execution_events'):
                torch.cuda.synchronize();anchor=torch.cuda.Event(enable_timing=True)
                before=time.monotonic();anchor.record();anchor.synchronize();after=time.monotonic()
                active.update(event_anchor=anchor,anchor_host_lower=before,anchor_host_upper=after)
            shutdown=llm.llm_engine.engine_core.shutdown
            def closed(*a,**kw):
                if 'value' in transport:transport['value'].close()
                pool=scheduler.kv_cache_manager.block_pool
                write(folder/f'KV_lifetime_{role}.json',dict(initial_free_blocks=initial_free,
                    final_free_blocks=pool.get_num_free_blocks(),total_blocks=pool.num_gpu_blocks,
                    pending_requests=len(scheduler.requests),running=len(scheduler.running),waiting=len(scheduler.waiting),
                    passed=pool.get_num_free_blocks()==initial_free and not scheduler.requests,
                    scope='Observed after all outputs and Hidden ACK cleanup, before Native worker shutdown. Prefix caching is disabled. This checks normal completion, not cancellation or GPU failure recovery.'))
                return shutdown(*a,**kw)
            llm.llm_engine.engine_core.shutdown=closed
            return llm
        vllm.LLM=initialized_llm
    from vllm.v1.engine.input_processor import InputProcessor
    assign=InputProcessor.assign_request_id
    def assigned(request):
        if cfg.get('ar_detokenize') is False and request.sampling_params is not None:
            assert request.sampling_params.ignore_eos and not request.sampling_params.stop
            request.sampling_params.detokenize=False
        result=assign(request)
        external_ids[request.request_id]=request.external_req_id
        event('REQUEST_IDENTITY',request_id=request.request_id,external_request_id=request.external_req_id)
        return result
    InputProcessor.assign_request_id=staticmethod(assigned)

    # Preserve the PP state contract. Keep raw Hidden and Residual separate.
    if mode in ('LOCAL_SPLIT','SPAN_ZERO','MIGRATE') and not (mode=='MIGRATE' and role=='decode' and cfg.get('separate_suffix_runner')):
        def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
            if intermediate_tensors is not None:
                hidden=intermediate_tensors['hidden_states'];residual=intermediate_tensors['residual'];start=cut
            else:
                hidden=inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
                residual=None;start=0
            end=cut if mode=='MIGRATE' and role=='prefill' else 36
            for index in range(start,end):
                hidden,residual=self.layers[index](positions,hidden,residual)
                if mode=='LOCAL_SPLIT' and index+1==cut:
                    hidden=hidden.clone();residual=residual.clone()
            if mode=='MIGRATE' and role=='prefill':
                if cfg.get('retain_fp32_boundary_residual'):residual=residual.float()
                return IntermediateTensors({'hidden_states':hidden,'residual':residual})
            hidden,_=self.norm(hidden,residual)
            return hidden
        # The existing decorator compiles this forward. We do not disable BI.
        Qwen3Model.forward=forward

    # A no-guards compiled module cannot switch between a None intermediate
    # input and a real PP input. Give suffix Prefill its own mandatory-input
    # compiled function. Full Decode retains the original full-model runner.
    if mode=='MIGRATE' and role=='decode' and cfg.get('separate_suffix_runner'):
        from vllm.compilation.decorators import support_torch_compile
        @support_torch_compile(dynamic_arg_dims={'positions':-1,'hidden':0,'residual':0})
        class DenseSuffix(torch.nn.Module):
            def __init__(self,*,vllm_config,full):
                super().__init__()
                self.layers=full.layers
                self.norm=full.norm
            def forward(self,positions,hidden,residual):
                if cfg.get('residual_dtype_bridge'):residual=residual.to(hidden.dtype)
                for index in range(cut,36):
                    hidden,residual=self.layers[index](positions,hidden,residual)
                hidden,_=self.norm(hidden,residual)
                return hidden

    if cfg.get('verify_layer_execution'):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        attention_forward=FlashAttentionImpl.forward
        def audited_attention(self,layer,*args,**kwargs):
            if active.get('audit_layers') is not None:
                active['audit_layers'].append(int(layer.layer_name.split('.')[2]))
            return attention_forward(self,layer,*args,**kwargs)
        FlashAttentionImpl.forward=audited_attention

    if cfg.get('graph_replay_audit'):
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        replay=CudaGraphManager.run_fullgraph;replay_counts={}
        def audited_replay(self,desc):
            batch=active['batch'];key=str(desc)
            replay_counts[key]=replay_counts.get(key,0)+1
            if batch is not None and cfg.get('verify_layer_execution'):
                event('FULL_GRAPH_REPLAY',requests=[external_ids.get(rid,rid) for rid in batch.req_ids],
                    prefilling=bool(batch.is_prefilling_np[:batch.num_reqs].any()),
                    descriptor=key,actual_model_layers=36,
                    evidence='CAPTURED_FULL_MODEL_GRAPH_DEFINITION_AND_ACTUAL_REPLAY_CALL')
            timed=cfg.get('execution_events') and batch is not None and all(
                external_ids.get(rid,'').startswith('q') and external_ids.get(rid,'')[1:].isdigit() for rid in batch.req_ids)
            if timed:
                start=time.monotonic();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            out=replay(self,desc)
            if timed:
                end.record();execution_events.append((begin,end,dict(host_submit=start,
                    requests=[external_ids[rid] for rid in batch.req_ids],query_tokens=batch.num_tokens,
                    prefilling=bool(batch.is_prefilling_np[:batch.num_reqs].any()),
                    layer_mode=mode,cut=cut,role=role,graph_replay=True)))
            if batch is not None and role=='decode' and cfg.get('reuse_prefill_token'):
                for i,rid in enumerate(batch.req_ids):
                    external=external_ids.get(rid,'')
                    if not (external.startswith('q') and external[1:].isdigit()):continue
                    if not bool(batch.is_prefilling_np[i]):continue
                    assert int(batch.num_computed_tokens_np[i])==cfg['length'] and rid not in released_tokens
                    released_tokens.add(rid)
                    notify(dict(kind='prefill_token_ready',request_id=external,time=time.monotonic(),
                        original_prompt_cached_tokens=cfg['length']))
                    event('NATIVE_REUSED_TOKEN_RELEASE',request_id=rid,external_request_id=external,
                        original_prompt_cached_tokens=cfg['length'],
                        scope='Release follows Native import and actual next full Decode Graph replay submission. GPU completion is not assumed.')
            active['batch']=None;return out
        CudaGraphManager.run_fullgraph=audited_replay
        atexit.register(lambda:write(folder/f'graph_replay_{role}.json',dict(counts=replay_counts,
            total_calls=sum(replay_counts.values()),scope='Actual full-graph replay calls. Warm-up and measured calls require separate per-request qualification records.')))

    original_prepare=GPUModelRunner.prepare_inputs
    def prepare(self,*args,**kwargs):
        batch=original_prepare(self,*args,**kwargs);active.update(batch=batch,runner=self)
        if cfg.get('execution_events'):
            active['last_prepared_meta']=dict(requests=[external_ids.get(rid,rid) for rid in batch.req_ids],
                prefilling=bool(batch.is_prefilling_np[:batch.num_reqs].any()),query_tokens=batch.num_tokens)
            active['sample_pending']=True
        if cfg.get('minimal_geometry_counter') and all(external_ids.get(rid,'').startswith('q') and external_ids.get(rid,'')[1:].isdigit() for rid in batch.req_ids):
            key=json.dumps(dict(prefilling=bool(batch.is_prefilling_np[:batch.num_reqs].any()),
                requests=batch.num_reqs,query_tokens=batch.num_tokens,
                padded_query_tokens=batch.num_tokens_after_padding),sort_keys=True)
            active['measured_geometry'][key]=active['measured_geometry'].get(key,0)+1
        if cfg.get('runtime_observation'):
            event('MODEL_INPUT_READY',request_ids=list(batch.req_ids),
                query_lengths=batch.num_scheduled_tokens.tolist(),computed_tokens=batch.num_computed_tokens_np.tolist(),
                prefilling=batch.is_prefilling_np.tolist(),tokens=batch.num_tokens)
        return batch
    GPUModelRunner.prepare_inputs=prepare
    if cfg.get('execution_events'):
        sample=GPUModelRunner.sample_tokens
        def timed_sample(self,*args,**kwargs):
            meta=active.get('last_prepared_meta',{});ids=meta.get('requests',[])
            measured=active.pop('sample_pending',False) and bool(ids) and all(r.startswith('q') and r[1:].isdigit() for r in ids)
            if measured:
                start=time.monotonic();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            out=sample(self,*args,**kwargs)
            if measured:
                end.record();sampler_events.append((begin,end,dict(meta,host_start=start,host_end=time.monotonic())))
            return out
        GPUModelRunner.sample_tokens=timed_sample

    schedule=Scheduler.schedule
    def observed_schedule(self,*args,**kwargs):
        credit_guard=cfg.get('hidden_source_backpressure') and mode=='MIGRATE' and role=='prefill' and 'value' in transport
        old_limit=self.max_num_running_reqs
        if credit_guard:
            transport['value'].reap();free=transport['value'].slots.free_count()
            self.max_num_running_reqs=min(old_limit,free)
        try:result=schedule(self,*args,**kwargs)
        finally:self.max_num_running_reqs=old_limit
        if credit_guard:
            assert len(result.num_scheduled_tokens)<=free,'Do not execute a Prefix without an export slot'
            if free<old_limit and time.monotonic()-active.get('last_credit_observation',0)>.1:
                active['last_credit_observation']=time.monotonic()
                event('HIDDEN_SOURCE_CREDIT_BACKPRESSURE',free_slots=free,normal_limit=old_limit,
                    admitted_requests=len(result.num_scheduled_tokens),
                    scope='Unstarted Source Prefill remains queued. No committed state is discarded or replayed.')
        if cfg.get('runtime_observation'):
            waiting=[r for r in self.waiting]
            event('SCHEDULER_RESULT',scheduled=dict(result.num_scheduled_tokens),
                running=len(self.running),waiting=len(waiting),
                waiting_requests=[dict(request_id=r.request_id,status=str(r.status),computed=r.num_computed_tokens,
                    prompt_tokens=r.num_prompt_tokens) for r in waiting],
                KV_free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                KV_usage=self.kv_cache_manager.usage)
        return result
    if cfg.get('runtime_observation') or (cfg.get('hidden_source_backpressure') and mode=='MIGRATE' and role=='prefill'):
        Scheduler.schedule=observed_schedule

    if cfg.get('cohort_barrier') and role=='decode':
        promote=Scheduler._try_promote_blocked_waiting_request;committed=set()
        def promote_cohort(self,request):
            params=request.kv_transfer_params or {};cohort=params.get('dense_cohort_id')
            if cohort is not None and cohort not in committed:
                expected=int(params['dense_cohort_size'])
                members=[r for r in self.requests.values() if (r.kv_transfer_params or {}).get('dense_cohort_id')==cohort]
                assert len(members)<=expected,'Cohort identity was reused'
                if len(members)<expected or any(r.request_id not in self.finished_recving_kv_req_ids for r in members):
                    return False
                assert not any(r.request_id in self.failed_recving_kv_req_ids for r in members)
                committed.add(cohort)
                event('NATIVE_KV_COHORT_READY',cohort=cohort,requests=[r.request_id for r in members])
            return promote(self,request)
        Scheduler._try_promote_blocked_waiting_request=promote_cohort

    if cfg.get('reuse_prefill_token') and role=='decode':
        from dataclasses import replace
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import NixlPullConnectorWorker
        matched=NixlPullConnectorScheduler.get_num_new_matched_tokens
        def matched_original_prompt(self,request,num_computed_tokens):
            params=request.kv_transfer_params or {};prefix=params.get('dense_source_prefix_tokens')
            if prefix is None:return matched(self,request,num_computed_tokens)
            assert params.get('do_remote_prefill') and prefix==cfg['length']
            assert request.num_prompt_tokens==prefix+1 and prefix%16==0 and not self._has_mamba
            count=prefix-num_computed_tokens
            assert count>=0
            return count,count>0
        NixlPullConnectorScheduler.get_num_new_matched_tokens=matched_original_prompt
        read_blocks=NixlPullConnectorWorker._read_blocks
        def read_original_prompt(self,*args,**kwargs):
            rid=kwargs['request_id'];external=external_ids.get(rid,'')
            if external.startswith('q') and external[1:].isdigit():
                spec=kwargs['read_spec'];expected=cfg['length']//self.block_size
                assert self.block_size==16 and not self._has_mamba and not self._is_hma_required
                assert len(spec.local_block_ids)==len(spec.remote_block_ids)==1
                assert len(spec.remote_block_ids[0])==expected
                assert expected<=len(spec.local_block_ids[0])<=expected+1
                locals_before=len(spec.local_block_ids[0])
                # Keep the new-token KV block reserved, but do not read it
                # from P: that generated token has not run on P. Native READ
                # validation, completion notification, and ACK remain intact.
                from .token_protocol import valid_prefix_blocks
                kwargs['read_spec']=replace(spec,local_block_ids=[valid_prefix_blocks(
                    spec.local_block_ids[0],spec.remote_block_ids[0],cfg['length'],self.block_size)])
                event('NATIVE_REUSE_EXACT_PREFIX_READ',request_id=rid,external_request_id=external,
                    source_valid_blocks=expected,destination_reserved_blocks=locals_before,read_blocks=expected,
                    source_valid_tokens=cfg['length'])
            return read_blocks(self,*args,**kwargs)
        NixlPullConnectorWorker._read_blocks=read_original_prompt

    if mode=='MIGRATE' and role=='decode':
        update=Scheduler._update_waiting_for_remote_kv
        def remote_ready(self,request):
            assert request.request_id not in self.failed_recving_kv_req_ids,'Do not consume failed partial KV'
            update(self,request)
            source_id=request.kv_transfer_params['remote_request_id']
            assert source_id!=request.request_id,'Keep independent Native internal identities'
            payload=storage/(source_id+('.gpu.json' if cfg.get('hidden_transport')=='NIXL_GPU' else '.pt'))
            assert payload.exists(),'Hidden state must be complete before partial KV is advertised'
            assert not self.cache_config.enable_prefix_caching
            assert len(request.output_token_ids)==0
            native_computed=request.num_computed_tokens
            request.num_computed_tokens=0
            imports[request.request_id]=source_id
            event('PARTIAL_KV_READY',request_id=request.request_id,native_computed=native_computed,
                source_request_id=source_id,resumed_query_start=0,resumed_layer_start=cut,full_prompt_queries=request.num_prompt_tokens)
        Scheduler._update_waiting_for_remote_kv=remote_ready

        original_inputs=DefaultModelState.prepare_inputs
        def imported_inputs(self,batch,req_states):
            inputs=original_inputs(self,batch,req_states)
            prefilling=[bool(v) for v in batch.is_prefilling_np[:batch.num_reqs]]
            if not any(prefilling):return inputs
            if all(rid.startswith('_warmup_') for rid in batch.req_ids):return inputs
            assert all(prefilling),'Diagnostic partial Prefill cannot share a batch with full Decode'
            hidden=[];residual=[];start=time.monotonic()
            for i,rid in enumerate(batch.req_ids):
                assert batch.num_computed_tokens_np[i]==0
                source_id=imports[rid]
                if cfg.get('hidden_transport')=='NIXL_GPU':
                    h,r=transport['value'].get(source_id,cut)
                    assert h.shape[0]==int(batch.num_scheduled_tokens[i])
                    hidden.append(h);residual.append(r);continue
                value=torch.load(storage/(source_id+'.pt'),map_location='cpu',weights_only=True)
                count=int(batch.num_scheduled_tokens[i])
                assert value['request_id']==source_id and value['cut']==cut and value['version']==1
                assert value['hidden'].shape[0]==count and value['residual'].shape==value['hidden'].shape
                assert tensor_hash(value['hidden'])==value['hidden_SHA256']
                assert tensor_hash(value['residual'])==value['residual_SHA256']
                hidden.append(value['hidden']);residual.append(value['residual'])
            h=torch.cat(hidden).to(batch.input_ids.device);r=torch.cat(residual).to(batch.input_ids.device)
            assert h.shape[0]==batch.num_tokens_after_padding,'Initial qualification requires no token padding'
            if cfg.get('hidden_transport')!='NIXL_GPU':torch.cuda.synchronize()
            inputs['intermediate_tensors']=IntermediateTensors({'hidden_states':h,'residual':r})
            event('HIDDEN_IMPORTED',request_ids=list(batch.req_ids),
                host_staged_bytes=0 if cfg.get('hidden_transport')=='NIXL_GPU' else (h.numel()+r.numel())*h.element_size(),
                elapsed_ms=1000*(time.monotonic()-start),
                evidence='MEASURED_NIXL_GPU_HIDDEN_IMPORT' if cfg.get('hidden_transport')=='NIXL_GPU' else 'CPU_STAGED_HIDDEN_TRANSFER_DIAGNOSTIC')
            return inputs
        DefaultModelState.prepare_inputs=imported_inputs

        if cfg.get('partial_KV_wire',False):
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
            original_desc=NixlBaseConnectorWorker._compute_desc_ids
            def prefix_descriptors(self,block_ids,dst_num_blocks,block_size_ratio,physical_blocks_per_logical):
                ids=original_desc(self,block_ids,dst_num_blocks,block_size_ratio,physical_blocks_per_logical)
                assert not self._has_mamba and not self._is_hma_required
                assert self.num_regions%36==0 and self.num_regions>=36,'Require independent uniform layer regions'
                blocks=dst_num_blocks if block_size_ratio is None else int(dst_num_blocks*block_size_ratio)
                regions=ids//blocks
                selected=ids[regions<cut*(self.num_regions//36)]
                assert selected.size*36==ids.size*cut
                event('PARTIAL_KV_DESCRIPTORS',total_descriptors=int(ids.size),selected_descriptors=int(selected.size),
                    regions=self.num_regions,cut=cut)
                return selected
            NixlBaseConnectorWorker._compute_desc_ids=prefix_descriptors

    outer=Qwen3ForCausalLM.forward
    def observed_forward(self,*args,**kwargs):
        batch=active['batch'];runner=active['runner']
        # Dummy model profiling can follow a real batch. Runner execution marks
        # active inputs only for this model invocation and clears them below.
        if batch is None:
            capture=cfg.get('verify_layer_execution') and torch.cuda.is_current_stream_capturing()
            if capture:active['audit_layers']=[]
            result=outer(self,*args,**kwargs)
            if capture:
                actual=active.pop('audit_layers');expected=list(range(36))
                event('CAPTURED_FULL_GRAPH_LAYERS',actual=actual,expected=expected,passed=actual==expected)
                assert actual==expected,'Decode Graph must contain all original layers'
            return result['hidden_states'] if isinstance(result,IntermediateTensors) else result
        synthetic=all(rid.startswith('_warmup_') for rid in batch.req_ids)
        if synthetic:
            active['batch']=None;result=outer(self,*args,**kwargs)
            return result['hidden_states'] if isinstance(result,IntermediateTensors) else result
        start=time.monotonic();prefilling=bool(batch.is_prefilling_np[:batch.num_reqs].any());reuse_notices=[]
        if cfg.get('reuse_prefill_token') and role=='decode' and prefilling:
            assert mode!='MIGRATE' and notify is not None
            for i,rid in enumerate(batch.req_ids):
                external=external_ids[rid]
                if not (external.startswith('q') and external[1:].isdigit()):continue
                if not bool(batch.is_prefilling_np[i]):continue
                assert int(batch.num_computed_tokens_np[i])==cfg['length'],'The complete original prompt KV must be imported'
                assert rid not in released_tokens
                reuse_notices.append((rid,external,int(batch.num_computed_tokens_np[i])))
        if mode=='MIGRATE' and prefilling:
            assert all(bool(v) for v in batch.is_prefilling_np[:batch.num_reqs])
            assert all(int(v)==0 for v in batch.num_computed_tokens_np[:batch.num_reqs])
        begin=end=None
        if cfg.get('execution_events'):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        if cfg.get('verify_layer_execution'):active['audit_layers']=[]
        suffix=mode=='MIGRATE' and role=='decode' and prefilling and cfg.get('separate_suffix_runner')
        if suffix:
            imported=kwargs.get('intermediate_tensors')
            assert isinstance(imported,IntermediateTensors),'A suffix must consume transferred state'
            if not hasattr(self,'_dense_suffix'):
                import copy
                from vllm.config import set_current_vllm_config
                suffix_config=copy.copy(self.vllm_config)
                # The compilation config contains the live Attention context.
                # Deepcopy would duplicate GPU KV/weights. Keep those aliases.
                suffix_config.compilation_config=copy.copy(self.vllm_config.compilation_config)
                suffix_key=str(cut)+('_r32' if cfg.get('retain_fp32_boundary_residual') else '_r16')
                suffix_key+='_bridge' if cfg.get('residual_dtype_bridge') else '_direct'
                suffix_key+='_'+cfg.get('ar_execution_family','DEFAULT')
                suffix_config.compilation_config.cache_dir=str(ROOT/'tools'/'separate_suffix_cache_v14'/suffix_key)
                suffix_config.compilation_config.local_cache_dir=None
                suffix_config.compilation_config.traced_files=set()
                with set_current_vllm_config(suffix_config):
                    compiled_suffix=DenseSuffix(vllm_config=suffix_config,full=self.model)
                # Do not register a second copy in Native model weight loading.
                object.__setattr__(self,'_dense_suffix',compiled_suffix)
            out=self._dense_suffix(kwargs['positions'],imported['hidden_states'],imported['residual'])
        else:out=outer(self,*args,**kwargs)
        if cfg.get('verify_layer_execution'):
            expected=list(range(cut,36)) if mode=='MIGRATE' and role=='decode' and prefilling else list(range(cut if mode=='MIGRATE' and role=='prefill' else 36))
            actual=active.pop('audit_layers')
            measured=all(external_ids[rid].startswith('q') and external_ids[rid][1:].isdigit() for rid in batch.req_ids)
            event('ACTUAL_ATTENTION_LAYER_EXECUTION',requests=[external_ids[rid] for rid in batch.req_ids],
                prefilling=prefilling,expected=expected,actual=actual,passed=actual==expected,measured=measured,
                evidence='OBSERVED_NATIVE_ATTENTION_BACKEND_INVOCATIONS_NOT_KERNEL_DURATIONS')
            if measured:assert actual==expected,('Incorrect actual layer execution',expected,actual)
        if begin is not None:
            end.record();execution_events.append((begin,end,dict(host_submit=start,
                requests=[external_ids[rid] for rid in batch.req_ids],query_tokens=batch.num_tokens,
                prefilling=prefilling,layer_mode=mode,cut=cut,role=role)))
        for rid,external,cached in reuse_notices:
            released_tokens.add(rid)
            notify(dict(kind='prefill_token_ready',request_id=external,time=time.monotonic(),original_prompt_cached_tokens=cached))
            event('NATIVE_REUSED_TOKEN_RELEASE',request_id=rid,external_request_id=external,
                original_prompt_cached_tokens=cached,
                scope='The P token is valid. Release follows Native import and next full Decode forward submission. GPU completion is not assumed.')
        if mode=='MIGRATE' and role=='prefill':
            assert isinstance(out,IntermediateTensors)
            h=out['hidden_states'];r=out['residual'];transport_start=time.monotonic()
            for i,rid in enumerate(batch.req_ids):
                a=int(batch.query_start_loc_np[i]);b=int(batch.query_start_loc_np[i+1])
                if cfg.get('hidden_transport')=='NIXL_GPU':
                    transport['value'].put(rid,h[a:b],r[a:b],cut);continue
                hc=h[a:b].detach().cpu();rc=r[a:b].detach().cpu()
                value=dict(version=1,request_id=rid,cut=cut,hidden=hc,residual=rc,
                    hidden_SHA256=tensor_hash(hc),residual_SHA256=tensor_hash(rc))
                path=storage/(rid+'.pt');temporary=path.with_suffix('.tmp')
                torch.save(value,temporary);temporary.replace(path)
                event('HIDDEN_EXPORTED',request_id=rid,cut=cut,tokens=b-a,
                    bytes=(hc.numel()+rc.numel())*hc.element_size(),
                    hidden_SHA256=value['hidden_SHA256'],residual_SHA256=value['residual_SHA256'])
            event('HIDDEN_EXPORT_HOST_ENCLOSURE',elapsed_ms=1000*(time.monotonic()-transport_start),
                transport=cfg.get('hidden_transport','CPU_STAGING'))
            out=h # P sample is discarded by the existing Native PD coordinator.
        if mode and cfg.get('state_capture',True):
            from vllm.forward_context import get_forward_context
            context=get_forward_context();mappings=context.slot_mapping
            limit=cut if mode=='MIGRATE' and role=='prefill' else 36
            for i,rid in enumerate(batch.req_ids):
                if not bool(batch.is_prefilling_np[i]):continue
                if cfg.get('state_first_query_only') and int(batch.num_computed_tokens_np[i])!=0:continue
                a=int(batch.query_start_loc_np[i]);b=int(batch.query_start_loc_np[i+1])
                # Numeric diagnostics can capture a common query prefix. This
                # never changes model inputs, Attention, or KV computation.
                if cfg.get('state_capture_prefix_tokens'):b=min(b,a+int(cfg['state_capture_prefix_tokens']))
                hashes=[]
                last_hashes=[]
                raw_states=[]
                for layer in range(limit):
                    name=f'model.layers.{layer}.self_attn.attn'
                    attn=context.no_compile_layers[name];cache=attn.kv_cache
                    assert cache.ndim==4 and cache.shape[1]==8 and cache.shape[3]==256,tuple(cache.shape)
                    slots=mappings[name][a:b].long();block_size=cache.shape[2]
                    assert bool((slots>=0).all())
                    valid=cache[slots//block_size,:,slots%block_size,:]
                    hashes.append(dict(layer=layer,SHA256=tensor_hash(valid),bytes=valid.numel()*valid.element_size()))
                    if cfg.get('capture_last_prompt_state'):
                        last_hashes.append(dict(layer=layer,SHA256=tensor_hash(valid[-1:]),bytes=valid[-1:].numel()*valid.element_size()))
                    if cfg.get('save_state_tensors'):raw_states.append(valid.detach().cpu())
                if cfg.get('save_state_tensors'):
                    state_path=folder/'state_tensors';state_path.mkdir(exist_ok=True)
                    torch.save(dict(request_id=rid,external_request_id=external_ids[rid],
                        KV=raw_states,hidden=out[a:b].detach().cpu()),state_path/(role+'_'+external_ids[rid]+'.pt'))
                event('VALID_QUERY_STATE',request_id=rid,external_request_id=external_ids[rid],query_start=int(batch.num_computed_tokens_np[i]),
                    query_tokens=b-a,final_hidden_SHA256=tensor_hash(out[a:b]),KV=hashes,
                    state_scope='Only queried valid KV cells. Unallocated and unused cache cells are excluded.')
                if cfg.get('capture_last_prompt_state'):
                    event('LAST_QUERY_STATE',request_id=rid,external_request_id=external_ids[rid],
                        absolute_position=int(batch.num_computed_tokens_np[i])+b-a-1,
                        KV=last_hashes,final_hidden_SHA256=tensor_hash(out[b-1:b]))
        if mode=='MIGRATE' and role=='decode' and prefilling and cfg.get('numerical_native_reference'):
            # Correctness diagnostic only. Keep the candidate KV and output.
            # Execute the independent original full model, compare, then
            # restore candidate suffix KV before its real Decode can start.
            from vllm.forward_context import get_forward_context
            context=get_forward_context();snapshots={};prefix_hashes={}
            def valid_layer(layer):
                name=f'model.layers.{layer}.self_attn.attn'
                cache=context.no_compile_layers[name].kv_cache
                slots=context.slot_mapping[name][:batch.num_tokens].long();block=cache.shape[2]
                return cache,slots,cache[slots//block,:,slots%block,:]
            for layer in range(36):
                cache,slots,value=valid_layer(layer)
                if layer<cut:prefix_hashes[layer]=tensor_hash(value)
                else:snapshots[layer]=(slots,value.clone())
            active['audit_layers']=[]
            reference_inputs=dict(kwargs,intermediate_tensors=None)
            reference=outer(self,*args,**reference_inputs)
            actual_reference_layers=active.pop('audit_layers')
            assert actual_reference_layers==list(range(36)),'Reference must execute the independent original full model'
            def difference(a,b):
                delta=(a.float()-b.float()).abs();limit=.02+.01*a.float().abs()
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                bad=int((delta>limit).sum())
                return dict(elements=a.numel(),maximum_absolute_error=float(delta.max()),
                    RMSE=float(delta.square().mean().sqrt()),tolerance_exceedances=bad,
                    bitwise_equal=bool(torch.equal(a,b)),finite=finite,passed=finite and bad==0,atol=.02,rtol=.01)
            comparisons=[]
            for layer in range(36):
                cache,slots,value=valid_layer(layer)
                if layer<cut:
                    assert tensor_hash(value)==prefix_hashes[layer],'The full reference must not change imported prefix state'
                else:comparisons.append(dict(layer=layer,**difference(value,snapshots[layer][1])))
            hidden_check=difference(reference,out)
            for i,rid in enumerate(batch.req_ids):
                a=int(batch.query_start_loc_np[i]);b=int(batch.query_start_loc_np[i+1]);hashes=[]
                for layer in range(36):
                    _,_,value=valid_layer(layer)
                    hashes.append(dict(layer=layer,SHA256=tensor_hash(value[a:b]),bytes=value[a:b].numel()*value.element_size()))
                event('FULL_NATIVE_REFERENCE_STATE',external_request_id=external_ids[rid],KV=hashes,
                    final_hidden_SHA256=tensor_hash(reference[a:b]),query_tokens=b-a,
                    evidence='CORRECTNESS_REFERENCE_ONLY_EXTRA_WORK_NOT_METHOD_WORK')
            restoration=[]
            for layer,(slots,saved) in snapshots.items():
                cache,_,_=valid_layer(layer);block=cache.shape[2]
                cache[slots//block,:,slots%block,:]=saved
                _,_,restored=valid_layer(layer);restoration.append(tensor_hash(saved)==tensor_hash(restored))
            passed=all(x['passed'] for x in comparisons) and hidden_check['passed'] and all(restoration)
            event('NUMERIC_SUFFIX_REFERENCE',requests=[external_ids[rid] for rid in batch.req_ids],
                measured=all(external_ids[rid].startswith('q') for rid in batch.req_ids),KV=comparisons,
                final_hidden=hidden_check,restoration_pass=all(restoration),passed=passed,
                actual_reference_layers=actual_reference_layers,
                scope='Extra original full-model reference execution is diagnostic only. Candidate suffix KV is restored. The returned output remains the candidate output. This work is never used for performance or offload totals.')
            assert passed,'Suffix numerical reference exceeds preregistered tolerance'
        if cfg.get('runtime_observation') or (mode and (prefilling or not cfg.get('minimal_serving_trace'))):
            event('MODEL_FORWARD_HOST_ENCLOSURE',request_ids=list(batch.req_ids),prefilling=prefilling,
                start=start,end=time.monotonic(),scope='Host enclosure. Not CUDA kernel union time.')
        if mode=='MIGRATE' and role=='decode' and prefilling and cfg.get('hidden_transport')=='NIXL_GPU':
            transport['value'].retire([imports[rid] for rid in batch.req_ids])
        active['batch']=None
        return out
    if mode or cfg.get('runtime_observation') or cfg.get('execution_events') or (cfg.get('reuse_prefill_token') and role=='decode'):Qwen3ForCausalLM.forward=observed_forward
    event('EXTENSION_INSTALLED',mode=mode,cut=cut,partial_KV_wire=cfg.get('partial_KV_wire',False),
        CUDA_graph_mode=cfg['ar_cudagraph_mode'],state_capture=cfg.get('state_capture',True))
