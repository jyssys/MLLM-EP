"""Row-aware causal continuation for pure axes and genuine two-dimensional partitions.

The stock model handles decode-only batches and graph capture. Partitioned
prefills use the same original layers with a per-row execution interval.
Each interval builds fresh attention metadata from the original KV ownership.
No attention or layer operation is replaced by an approximate operator.
"""
import atexit
import copy
import hashlib
import time
from .common import ROOT,write
from .geometry import Partition,aligned_suffix,active_spans


class Adapter:
    def __init__(self,cfg,role,info):
        self.cfg,self.role,self.info=cfg,role,info
        self.folder=ROOT/'raw'/cfg['name'];self.rows=[];self.states={}
        self.specs={};self.imports={};self.active={};self.transport=None
        self.closed=False;self.active_read=None;self.capture_layers=None
        self.install();atexit.register(self.close)

    @staticmethod
    def external(rid):return rid.split('-')[0]

    def partition(self,rid):return self.specs[self.external(rid)]

    def command(self,command):
        if command.get('kind') not in ('add','cleanup_remote'):return command
        command=dict(command);rid=command['request_id'];length=command['original_length']
        action=self.cfg.get('axis_actions_by_request',{}).get(rid,{})
        kind=action.get('kind',command.get('axis_action',self.cfg['axis_kind']))
        upper=int(action.get('cut',command.get('axis_cut',self.cfg.get('axis_cut',0))));requested_cut=upper
        suffix=0
        if kind=='Token':
            suffix=aligned_suffix(length,upper);upper=0
        elif kind=='Hybrid':
            fraction=action.get('suffix_fraction',self.cfg['hybrid_suffix_fraction'])
            suffix=0 if fraction==0 else max(16,int(round(length*fraction/16))*16)
            suffix=min(suffix,(length-1)//16*16)
        elif kind=='Native':upper=0
        else:assert kind=='Layer',kind
        exact=self.cfg.get('suffix_by_request',{})
        if kind in ('Token','Hybrid') and rid in exact:suffix=int(exact[rid])
        part=Partition(length,upper,suffix);self.specs[rid]=part
        command.update(axis_action=kind,axis_cut=requested_cut,axis_full_length=length,
            axis_prefix_length=part.prefix,axis_actual_upper=part.upper,axis_suffix_tokens=part.suffix)
        self.rows.append(dict(event='PARTITION',request_id=rid,axis=kind,**part.record()))
        if self.role=='prefill' and suffix:
            assert len(command['input_ids'])==length
            command.update(input_ids=command['input_ids'][:part.prefix],original_length=part.prefix)
        return command

    def attach(self,llm):
        needs_hidden=self.cfg.get('axis_cut',0)>0 or any(
            action.get('kind') in ('Layer','Hybrid') and action.get('cut',0)>0
            for action in self.cfg.get('axis_actions_by_request',{}).values())
        if self.cfg['axis_kind'] in ('Layer','Hybrid') and needs_hidden:
            from .hidden_transport import HiddenTransport
            self.transport=HiddenTransport(self.cfg,self.role,self.folder)

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.transport is not None:self.transport.close()
        finally:
            write(self.folder/f'axis_events_{self.role}.json',dict(rows=self.rows,states=self.states,
                evidence='CORRECTNESS_ONLY' if self.cfg.get('correctness_only') else 'MEASURED_EXECUTION_COVERAGE',
                scope='Per-request query intervals and actual layer segments. Padding has no valid query or KV slot. '
                    'Host boundaries do not measure GPU-exclusive execution time. Decode graph replay counts are in the standard worker log.'))

    def capture_state(self,runner,batch,index,length,layers,hidden,key):
        import torch
        if not self.cfg.get('axis_state_capture',False) or key in self.states:return
        table=runner.block_tables.input_block_tables[0][index]
        capture_begin=time.monotonic()
        positions=torch.arange(length,device='cuda');records=[];samples=[];hash_jobs=[]
        hash_workers=int(self.cfg.get('axis_hash_workers',8))
        from .tensor_hash import submit
        part=self.partition(batch.req_ids[index])
        sample_positions=sorted(set(p for p in (0,1,part.prefix-1,part.prefix,part.prefix+1,length-1) if 0<=p<length))
        dumps=[]
        for number in range(layers):
            attn=runner.vllm_config.compilation_config.static_forward_context[f'model.layers.{number}.self_attn.attn']
            block=attn.kv_cache.shape[2]
            assert block==self.cfg['KV_block_tokens'],('KV token axis differs from the configured block size',attn.kv_cache.shape)
            value=attn.kv_cache[table[positions//block].long(),:,positions%block,:].detach().contiguous()
            assert tuple(value.shape)==(length,8,256) and value.dtype==torch.bfloat16 and value.numel()*value.element_size()==length*4096,(
                'Unexpected packed HND KV value',tuple(value.shape),str(value.dtype))
            if number==0 and not getattr(self,'cache_schema_written',False):
                write(self.folder/f'KV_cache_layout_{self.role}.json',dict(shape=list(attn.kv_cache.shape),strides=list(attn.kv_cache.stride()),
                    dtype=str(value.dtype),token_axis=2,block_tokens=block,valid_per_token_shape=[8,256],valid_bytes_per_token=4096,
                    scope='Actual packed FlashAttention HND cache. The final256 channels contain both K and V for each128-dimensional KV head.'))
                self.cache_schema_written=True
            if self.cfg.get('gpu_state_digest'):
                from .gpu_digest import gpu
                records.append(dict(layer=number,**gpu(value,verify=self.cfg.get('verify_GPU_digest_with_CPU',False))))
            else:
                value=value.cpu();hash_jobs.append(submit(value,hash_workers))
                records.append(dict(layer=number,bytes=value.numel()*value.element_size()))
            samples.append(value[sample_positions].clone().cpu())
            if self.cfg.get('dump_full_KV'):dumps.append(value.cpu())
        for record,job in zip(records,hash_jobs):record['SHA256']=job.result()
        state=dict(length=length,layers=layers,KV=records,sample_positions=sample_positions,
            correctness_CPU_hash_workers=0 if self.cfg.get('gpu_state_digest') else hash_workers,
            GPU_digest_observer=bool(self.cfg.get('gpu_state_digest')),
            hidden_SHA256=hashlib.sha256(hidden.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() if hidden is not None else None)
        tensor_path=self.folder/f'state_{self.role}_{key}.pt'
        saved_hidden=hidden[-1:] if hidden is not None and self.cfg.get('axis_capture_hidden_last_only') else hidden
        torch.save(dict(KV_samples=samples,KV_full=dumps or None,
            hidden=saved_hidden.detach().cpu() if saved_hidden is not None else None,positions=sample_positions,
            hidden_scope='LAST_QUERY_ONLY' if self.cfg.get('axis_capture_hidden_last_only') else 'ENTIRE_CURRENT_QUERY_CHUNK'),tensor_path)
        state['tensor_file']=str(tensor_path);state['correctness_capture_host_ms']=1000*(time.monotonic()-capture_begin)
        self.states[key]=state
        if self.role=='prefill' and not part.upper and not part.suffix and key==self.external(batch.req_ids[index]):
            self.capture_state(runner,batch,index,length-1,layers,None,'native_export_'+key)

    def capture_import(self,runner,batch,index):
        if self.role!='decode' or not self.cfg.get('axis_import_hash_after_model'):return
        rid=batch.req_ids[index];part=self.partition(rid)
        length=part.prefix-1 if not part.upper and not part.suffix else part.prefix
        self.capture_state(runner,batch,index,length,part.lower if part.upper else 36,None,'import_'+self.external(rid))

    def completed_full_graph(self,runner,execution='FULL36_CAPTURED_GRAPH_REPLAY'):
        frame=runner.execute_model_state
        assert frame is not None and frame.hidden_states is not None
        batch=frame.input_batch
        for i,rid in enumerate(batch.req_ids):
            if self.external(rid) not in self.specs:continue
            part=self.partition(rid);start=int(batch.num_computed_tokens_np[i]);count=int(batch.num_scheduled_tokens[i])
            if self.role=='prefill':assert not part.upper,'Partial prefill must not replay the full36 graph'
            else:assert not part.upper or start>=part.prefix,'Prefix continuation must remain eager'
            self.rows.append(dict(event='MODEL_VISIT',request_id=self.external(rid),start=start,count=count,
                layers=list(range(36)),actual_request_batch=int(batch.num_reqs),stage_request_batch=int(batch.num_reqs),
                stage_query_rows=int(batch.num_tokens),execution=execution,time=time.monotonic()))
            length=part.prefix if self.role=='prefill' else part.length
            if start<length<=start+count:
                left=int(batch.query_start_loc_np[i]);right=int(batch.query_start_loc_np[i+1])
                self.capture_import(runner,batch,i)
                self.capture_state(runner,batch,i,length,36,frame.hidden_states[left:right],self.external(rid))

    def metadata(self,spans):
        """Build exact request tails without reusing the full batch's FA schedule."""
        import torch
        from vllm.forward_context import get_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
        batch=self.active['batch'];runner=self.active['runner'];context=get_forward_context()
        assert isinstance(context.attn_metadata,dict) and isinstance(context.slot_mapping,dict)
        assert len(runner.kv_cache_config.kv_cache_groups)==1
        first=next(iter(context.attn_metadata.values()))
        assert first.__class__.__name__=='FlashAttentionMetadata' and not first.use_cascade
        assert not first.max_dcp_context_kv_len and first.causal is True
        requests=[r for r,_,_ in spans];counts=[right-left for _,left,right in spans]
        flat=[position for _,left,right in spans for position in range(left,right)]
        full=flat==list(range(int(batch.num_tokens)))
        if full and not self.active.get('metadata_rebuilt'):
            return None,context
        self.active['metadata_rebuilt']=True
        index=torch.tensor(flat,dtype=torch.long,device='cuda')
        req_index=torch.tensor(requests,dtype=torch.long,device='cuda')
        offsets=[0]
        for count in counts:offsets.append(offsets[-1]+count)
        cpu=torch.tensor(offsets,dtype=torch.int32)
        gpu=cpu.to('cuda',non_blocking=True)
        slots=next(iter(context.slot_mapping.values())).index_select(0,index)
        # Every selected interval is a tail. Its final position stays unchanged.
        # Therefore the original request seq_len remains valid for causal attention.
        upper=batch.seq_lens_cpu_upper_bound[requests]
        metadata=build_attn_metadata(runner.attn_groups,len(requests),len(flat),gpu,cpu,max(counts),
            first.seq_lens.index_select(0,req_index),int(upper.max()),
            (first.block_table.index_select(0,req_index),),slots.unsqueeze(0),runner.kv_cache_config,
            seq_lens_cpu_upper_bound=upper,positions=batch.positions.index_select(0,index))
        local=copy.copy(context);local.attn_metadata=metadata
        local.slot_mapping={name:slots for name in context.slot_mapping}
        local.is_padding=None;local.skip_compiled=True;local.batch_descriptor=None
        from vllm.config import CUDAGraphMode
        local.cudagraph_runtime_mode=CUDAGraphMode.NONE
        return None if full else index,local

    def partial_forward(self,module,input_ids,positions):
        import torch
        from vllm.forward_context import override_forward_context
        batch=self.active['batch'];runner=self.active['runner'];n=int(batch.num_tokens)
        self.active['partitioned']=True
        starts=[int(x) for x in batch.query_start_loc_np[:batch.num_reqs+1]]
        computed=[int(x) for x in batch.num_computed_tokens_np[:batch.num_reqs]]
        parts=[self.partition(rid) for rid in batch.req_ids]
        prefilling=[bool(x) for x in batch.is_prefilling_np[:batch.num_reqs]]
        prefixes=[part.prefix if prefilling[i] and part.upper and self.role=='decode' else 0 for i,part in enumerate(parts)]
        cuts=[part.lower if prefilling[i] and part.upper else (36 if self.role=='prefill' else 0) for i,part in enumerate(parts)]
        if self.role=='prefill':assert all(prefilling),'The producer must never run decode'
        h=module.embed_input_ids(input_ids[:n]);r=torch.empty_like(h)
        if self.role=='decode':
            for i,rid in enumerate(batch.req_ids):
                if not prefixes[i]:continue
                left,right=starts[i:i+2];start=computed[i]
                count=max(0,min(right-left,prefixes[i]-start))
                if count:
                    assert rid in self.imports,('Missing partial-state ownership',rid)
                    incoming_h,incoming_r=self.transport.get(self.imports[rid],parts[i].lower)
                    h[left:left+count].copy_(incoming_h[start:start+count])
                    r[left:left+count].copy_(incoming_r[start:start+count])
        boundaries=sorted(set([0,36]+[cut for cut in cuts if 0<cut<36]))
        for begin,end in zip(boundaries,boundaries[1:]):
            spans=active_spans(starts,computed,prefixes,cuts,begin,'P' if self.role=='prefill' else 'D')
            if not spans:continue
            index,context=self.metadata(spans)
            local_h=h if index is None else h.index_select(0,index)
            local_r=None if begin==0 else r if index is None else r.index_select(0,index)
            local_positions=positions[:n] if index is None else positions.index_select(0,index)
            with override_forward_context(context):
                for layer in range(begin,end):local_h,local_r=module.layers[layer](local_positions,local_h,local_r)
            assert local_r is not None
            if index is None:h,r=local_h,local_r
            else:h.index_copy_(0,index,local_h);r.index_copy_(0,index,local_r)
            for i,left,right in spans:
                query_start=computed[i]+left-starts[i]
                self.rows.append(dict(event='MODEL_VISIT',request_id=self.external(batch.req_ids[i]),
                    start=query_start,count=right-left,layers=list(range(begin,end)),
                    actual_request_batch=int(batch.num_reqs),stage_request_batch=len(spans),
                    stage_query_rows=len(local_h),time=time.monotonic()))
        if self.role=='prefill':
            for i,rid in enumerate(batch.req_ids):
                part=parts[i];left,right=starts[i:i+2]
                if part.upper:self.transport.put_part(rid,computed[i],part.prefix,h[left:right],r[left:right],part.lower)
                if computed[i]<part.prefix<=computed[i]+right-left:
                    self.capture_state(runner,batch,i,part.prefix,part.lower if part.upper else 36,h[left:right],self.external(rid))
            # Producer samples are discarded. Norm is required only for full36 producers.
            full=[i for i,part in enumerate(parts) if not part.upper]
            for i in full:
                left,right=starts[i:i+2];h[left:right],_=module.norm(h[left:right],r[left:right])
        else:
            h,_=module.norm(h,r)
            for i,rid in enumerate(batch.req_ids):
                part=parts[i];left,right=starts[i:i+2]
                if computed[i]<part.length<=computed[i]+right-left:
                    self.capture_import(runner,batch,i)
                    self.capture_state(runner,batch,i,part.length,36,h[left:right],self.external(rid))
                if part.upper and computed[i]<part.prefix<=computed[i]+right-left:
                    self.transport.retire(self.imports.pop(rid))
        padded=int(batch.num_tokens_after_padding)
        if padded>n:
            h=torch.cat((h,torch.zeros((padded-n,h.shape[1]),device=h.device,dtype=h.dtype)))
            self.rows.append(dict(event='UNSCHEDULED_PADDING_ROWS',valid=n,padded=padded,computed_partial_rows=0))
        return h

    def install(self):
        import torch
        from vllm.model_executor.models.qwen3 import Qwen3Model,Qwen3ForCausalLM,Qwen3DecoderLayer
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import NixlPullConnectorWorker
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
        me=self
        prepare=GPUModelRunner.prepare_inputs
        def prepared(runner,*a,**kw):
            batch=prepare(runner,*a,**kw);me.active.update(batch=batch,runner=runner,partitioned=False,layer_calls=[],metadata_rebuilt=False,graph_replayed=False,python_observed=False);return batch
        GPUModelRunner.prepare_inputs=prepared
        execute=GPUModelRunner.execute_model
        def executed(runner,*a,**kw):
            me.dispatch_schedule=a[0] if a else kw.get('scheduler_output')
            try:
                result=execute(runner,*a,**kw)
                if me.active.get('graph_replayed') and not me.active.get('python_observed'):me.completed_full_graph(runner)
                return result
            finally:me.active.clear();me.dispatch_schedule=None
        GPUModelRunner.execute_model=executed
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        replay=CudaGraphManager.run_fullgraph
        def replayed(manager,*a,**kw):
            if me.active:me.active['graph_replayed']=True
            return replay(manager,*a,**kw)
        CudaGraphManager.run_fullgraph=replayed
        attention=GPUModelRunner.prepare_attn
        def prepared_attention(runner,batch,*a,**kw):
            result=attention(runner,batch,*a,**kw)
            if me.role=='decode' and me.cfg.get('axis_state_capture',False) and not me.cfg.get('axis_import_hash_after_model'):
                for i,rid in enumerate(batch.req_ids):
                    if me.external(rid) not in me.specs:continue
                    part=me.partition(rid)
                    length=part.prefix-1 if not part.upper and not part.suffix else part.prefix
                    me.capture_state(runner,batch,i,length,part.lower if part.upper else 36,None,'import_'+me.external(rid))
            return result
        GPUModelRunner.prepare_attn=prepared_attention
        if self.role=='decode':
            matched=NixlPullConnectorScheduler.get_num_new_matched_tokens
            def matching(connector,request,computed):
                part=me.partition(request.request_id)
                if not part.upper and not part.suffix:
                    # Preserve the normal Native last-prompt-query recomputation.
                    return matched(connector,request,computed)
                if request.kv_transfer_params and request.kv_transfer_params.get('do_remote_prefill'):
                    assert request.kv_transfer_params['remote_num_tokens']==part.prefix
                    return max(0,part.prefix-computed),part.prefix>computed
                return matched(connector,request,computed)
            NixlPullConnectorScheduler.get_num_new_matched_tokens=matching
            ready=Scheduler._update_waiting_for_remote_kv
            def remote_ready(scheduler,request):
                result=ready(scheduler,request);part=me.partition(request.request_id)
                if not part.upper:return result
                source=request.kv_transfer_params['remote_request_id']
                assert (me.folder/'hidden_transport'/(source+'.json')).exists()
                me.imports[request.request_id]=source;request.num_computed_tokens=0
                me.rows.append(dict(event='PARTIAL_KV_REMOTE_READY',request_id=me.external(request.request_id),prefix=part.prefix,lower=part.lower,time=time.monotonic()))
                return result
            Scheduler._update_waiting_for_remote_kv=remote_ready
        read=NixlPullConnectorWorker._read_blocks
        def reading(worker,*a,**kw):
            me.active_read=kw['request_id']
            try:return read(worker,*a,**kw)
            finally:me.active_read=None
        NixlPullConnectorWorker._read_blocks=reading
        desc=NixlBaseConnectorWorker._compute_desc_ids
        def descriptors(worker,block_ids,dst_num_blocks,block_size_ratio,physical_blocks_per_logical):
            value=desc(worker,block_ids,dst_num_blocks,block_size_ratio,physical_blocks_per_logical)
            if me.active_read is None:return value
            part=me.partition(me.active_read)
            if not part.upper:return value
            assert worker.num_regions%36==0 and not worker._has_mamba and not worker._is_hma_required
            blocks=dst_num_blocks if block_size_ratio is None else int(dst_num_blocks*block_size_ratio)
            selected=value[value//blocks<part.lower*(worker.num_regions//36)]
            assert len(selected)*36==len(value)*part.lower
            me.rows.append(dict(event='PARTIAL_KV_DESCRIPTORS',total=len(value),selected=len(selected),lower=part.lower,time=time.monotonic()))
            return selected
        NixlBaseConnectorWorker._compute_desc_ids=descriptors
        original=Qwen3Model.forward
        def model(module,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
            batch=me.active.get('batch')
            if batch is None or not any(batch.is_prefilling_np[:batch.num_reqs]):return original(module,input_ids,positions,intermediate_tensors,inputs_embeds)
            if not all(me.external(rid) in me.specs for rid in batch.req_ids):return original(module,input_ids,positions,intermediate_tensors,inputs_embeds)
            parts=[me.partition(rid) for rid in batch.req_ids]
            if not any(part.upper for part in parts):return original(module,input_ids,positions,intermediate_tensors,inputs_embeds)
            if me.role=='decode' and not any(part.upper and bool(batch.is_prefilling_np[i]) and int(batch.num_computed_tokens_np[i])<part.prefix for i,part in enumerate(parts)):
                return original(module,input_ids,positions,intermediate_tensors,inputs_embeds)
            assert intermediate_tensors is None and inputs_embeds is None
            return me.partial_forward(module,input_ids,positions)
        Qwen3Model.forward=model
        layer_forward=Qwen3DecoderLayer.forward
        def layer(module,*a,**kw):
            number=int(module.self_attn.attn.layer_name.split('.')[2])
            if me.capture_layers is not None:me.capture_layers.append(number)
            if me.active:me.active['layer_calls'].append(number)
            return layer_forward(module,*a,**kw)
        Qwen3DecoderLayer.forward=layer
        outer=Qwen3ForCausalLM.forward
        def observed(module,*a,**kw):
            if me.active:me.active['python_observed']=True
            capturing=torch.cuda.is_current_stream_capturing()
            if capturing:me.capture_layers=[]
            result=outer(module,*a,**kw)
            if capturing:
                layers=me.capture_layers;me.capture_layers=None
                assert layers==list(range(36)),('Decode graph must remain full36',layers)
                me.rows.append(dict(event='CAPTURED_FULL_DECODE_GRAPH',layers=layers,passed=True))
            batch=me.active.get('batch')
            if batch is not None and not me.active['partitioned'] and all(me.external(rid) in me.specs for rid in batch.req_ids):
                assert me.active['layer_calls']==list(range(36)),('Full-model path changed coverage',me.active['layer_calls'])
                for i,rid in enumerate(batch.req_ids):
                    part=me.partition(rid);start=int(batch.num_computed_tokens_np[i]);count=int(batch.num_scheduled_tokens[i])
                    me.rows.append(dict(event='MODEL_VISIT',request_id=me.external(rid),start=start,count=count,
                        layers=list(range(36)),actual_request_batch=int(batch.num_reqs),stage_request_batch=int(batch.num_reqs),
                        stage_query_rows=int(batch.num_tokens),time=time.monotonic()))
                    length=part.prefix if me.role=='prefill' else part.length
                    if start<length<=start+count:
                        left=int(batch.query_start_loc_np[i]);right=int(batch.query_start_loc_np[i+1])
                        me.capture_import(me.active['runner'],batch,i)
                        me.capture_state(me.active['runner'],batch,i,length,36,result[left:right],me.external(rid))
            return result
        Qwen3ForCausalLM.forward=observed
        from .chunk_boundary import install as install_boundary
        install_boundary(self)
        from .graph_guard import install as install_graph_guard
        install_graph_guard(self)
