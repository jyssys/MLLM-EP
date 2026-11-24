"""Read-only LLM input, prompt KV and greedy-margin probes. Correctness only."""
import hashlib
import os
import re
import time

from .common import ROOT,write,digest


def capture_input_allowed(decode_role,include_decode_inputs,event_index,limit):
    return (not decode_role or include_decode_inputs) and event_index<limit


def capture_attention_allowed(layer_name,event,limit,attention_layer=0):
    return (bool(re.search(rf'(?:^|\.)layers\.{attention_layer}\.self_attn\.attn$',layer_name))
            and event is not None and event['index']<limit and event['query_tokens']>1)


def capture_gemm_allowed(event,last_attention_layer,limit):
    return (last_attention_layer==0 and event is not None and event['index']<limit
            and event['query_tokens']>1 and len(event.get('first_layer_GEMM',[]))<4)


def gather_prompt_kv(cache, block_ids, length):
    import torch
    assert cache.ndim==4 and length>0
    positions=torch.arange(length,device=cache.device)
    blocks=torch.tensor(block_ids,device=cache.device,dtype=torch.long)
    size=cache.shape[2]
    assert len(block_ids)*size>=length
    return cache[blocks[positions//size],:,positions%size,:].contiguous()


class ForwardProbe:
    def __init__(self,worker,name,max_logits,include_decode_inputs=False,include_attention=False,include_gemm=False,attention_layer=0):
        assert re.fullmatch('[A-Za-z0-9_.-]+',name) and name not in ('.','..')
        self.runner=worker.model_runner;self.model=self.runner.model;self.max_logits=max_logits
        self.decode_role=os.environ['MLLM_EPD_ROLE'] in ('decode','reference_decode')
        self.include_decode_inputs=include_decode_inputs
        self.include_attention=include_attention
        assert isinstance(attention_layer,int) and 0<=attention_layer<28
        assert not include_gemm or attention_layer==0,'Generated GEMM call-site capture is qualified only for layer0'
        self.attention_layer=attention_layer
        assert not include_gemm or include_attention
        self.include_gemm=include_gemm;self.last_attention_layer=None
        self.folder=ROOT/'raw/llm_reference'/name
        assert not self.folder.exists();self.folder.mkdir(parents=True)
        assert len(self.runner.kv_caches)==28,'This probe requires the actual Qwen2.5-VL 28-layer cache'
        self.events=[];self.logits=[];self.tensors={};self.snapshot=None;self.current=None

    def tensor(self,name,value):
        # Dedicated CPU storage avoids serializing a larger backing allocation.
        self.tensors[name]=value.detach().cpu().contiguous().clone()

    def begin(self):
        from vllm.forward_context import get_forward_context
        import torch
        self.original_execute=self.runner.execute_model
        self.original_forward=self.runner._model_forward
        self.original_logits=self.model.compute_logits
        self.original_slots=[(obj,key,key in vars(obj),vars(obj).get(key)) for obj,key in
            [(self.runner,'execute_model'),(self.runner,'_model_forward'),(self.model,'compute_logits')]]
        me=self
        def execute(output,*args,**kwargs):
            me.current=output
            return me.original_execute(output,*args,**kwargs)
        def forward(*args,**kwargs):
            output=me.current;batch=me.runner.input_batch
            ids=list(batch.req_ids)
            assert len(ids)==1,'Use a single-request correctness probe, not a serving run'
            rid=ids[0];i=0;start=int(batch.num_computed_tokens_cpu[i]);n=int(output.num_scheduled_tokens[rid])
            length=int(batch.num_prompt_tokens[i]);context=get_forward_context()
            event=dict(index=len(me.events),request_id=rid,start=start,query_tokens=n,prompt_tokens=length,
                cudagraph_mode=str(context.cudagraph_runtime_mode),batch_descriptor=str(context.batch_descriptor))
            me.last_attention_layer=None
            for key in ('input_ids','inputs_embeds','positions'):
                value=kwargs.get(key)
                if isinstance(value,torch.Tensor):
                    event[key+'_shape']=list(value.shape)
                    # Do not introduce a D-side pre-forward GPU synchronization.
                    if capture_input_allowed(me.decode_role,me.include_decode_inputs,len(me.events),me.max_logits):
                        data=value[:n] if key!='positions' else value[...,:n]
                        me.tensor(key+'_'+str(len(me.events)),data)
            me.events.append(event)
            return me.original_forward(*args,**kwargs)
        def logits(hidden,*args,**kwargs):
            result=me.original_logits(hidden,*args,**kwargs)
            event=me.events[-1]
            if event['start']+event['query_tokens']>=event['prompt_tokens'] and len(me.logits)<me.max_logits:
                values,indices=result.float().topk(32,dim=-1)
                me.logits.append(dict(event=event['index'],token_ids=indices.cpu().tolist(),values=values.cpu().tolist()))
                me.tensor('hidden_'+str(len(me.logits)-1),hidden)
                if me.snapshot is None:
                    batch=me.runner.input_batch;table=batch.block_table[0]
                    length=event['prompt_tokens'];num_blocks=int(table.num_blocks_per_row[0])
                    blocks=table.block_table.np[0,:num_blocks].tolist()
                    signatures=[]
                    for index,cache in enumerate(me.runner.kv_caches):
                        assert cache.shape[1]==4 and cache.shape[3]==256,(index,cache.shape)
                        data=gather_prompt_kv(cache,blocks,length)
                        me.tensor('prompt_KV_'+str(index),data)
                        signatures.append(dict(layer=index,shape=list(data.shape),cache_shape=list(cache.shape),
                            cache_stride=list(cache.stride()),dtype=str(data.dtype)))
                    request=me.runner.requests[event['request_id']]
                    token_ids=request.prompt_token_ids
                    me.tensor('prompt_token_ids',torch.tensor(token_ids,dtype=torch.int64))
                    if request.mrope_positions is not None:me.tensor('full_prompt_mrope',request.mrope_positions[...,:length])
                    me.snapshot=dict(event=event,block_ids=blocks,kernel_block_size=int(table.block_size),layers=signatures,
                        mrope_delta=request.mrope_position_delta,
                        scope='All valid prompt KV positions after first complete prompt query. '
                            'For D,positions below event.start are imported. The remaining positions may be recomputed.')
            return result
        self.runner.execute_model=execute;self.runner._model_forward=forward;self.model.compute_logits=logits
        if self.include_attention:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
            original_attention=FlashAttentionImpl.forward
            self.original_slots.append((FlashAttentionImpl,'forward',True,original_attention))
            def attention(impl,layer,query,key,value,kv_cache,metadata,output,*args,**kwargs):
                event=me.events[-1] if me.events else None
                match=re.search(r'(?:^|\.)layers\.(\d+)\.self_attn\.attn$',getattr(layer,'layer_name',''))
                me.last_attention_layer=int(match.group(1)) if match else None
                selected=capture_attention_allowed(getattr(layer,'layer_name',''),event,me.max_logits,me.attention_layer)
                if selected:
                    n=metadata.num_actual_tokens;index=event['index']
                    event['attention_0']=dict(layer_name=layer.layer_name,
                        implementation=type(impl).__name__,version=impl.vllm_flash_attn_version,
                        **{name:getattr(metadata,name) for name in ['num_actual_tokens','max_query_len',
                            'max_seq_len','use_cascade','common_prefix_len','max_num_splits']})
                    for name,tensor in [('query',query),('key',key),('value',value)]:
                        me.tensor(f'attention_0_{name}_{index}',tensor[:n])
                    for name in ['query_start_loc','seq_lens','scheduler_metadata','block_table','slot_mapping']:
                        tensor=getattr(metadata,name,None)
                        if tensor is not None:
                            me.tensor(f'attention_0_{name}_{index}',tensor)
                result=original_attention(impl,layer,query,key,value,kv_cache,metadata,output,*args,**kwargs)
                if selected:
                    me.tensor(f'attention_0_output_{index}',result[:n])
                return result
            FlashAttentionImpl.forward=attention
        if self.include_gemm:
            import inspect
            from torch._inductor.select_algorithm import extern_kernels
            def gemm_wrapper(kind,original):
                def captured(*args,**kwargs):
                    event=me.events[-1] if me.events else None
                    selected=capture_gemm_allowed(event,me.last_attention_layer,me.max_logits)
                    if selected:
                        calls=event.setdefault('first_layer_GEMM',[]);index=len(calls)
                        matrix=args[0] if kind=='mm' else args[1]
                        weight=args[1] if kind=='mm' else args[2]
                        calls.append(dict(index=index,kind=kind,input_shape=list(matrix.shape),
                            weight_shape=list(weight.shape)))
                        me.tensor(f'gemm_0_input_{event["index"]}_{index}',matrix)
                        if index in (0,1):
                            frame=inspect.currentframe().f_back
                            try:
                                if index==0:
                                    for name,local in [('residual','arg4_1'),('weight','arg3_1')]:
                                        tensor=frame.f_locals.get(local)
                                        assert isinstance(tensor,torch.Tensor), 'The actual generated norm call-site layout changed'
                                        me.tensor(f'norm_0_{name}_{event["index"]}',tensor)
                                else:
                                    runtime=[]
                                    for name in ['triton_red_fused_fused_add_rms_norm_0',
                                                 'triton_red_fused_fused_add_rms_norm_2']:
                                        kernel=frame.f_globals.get(name)
                                        def configs(values):
                                            return [dict(kwargs=dict(c.kwargs),num_warps=c.num_warps,
                                                num_stages=c.num_stages) for c in (values or []) if c is not None]
                                        launchers=getattr(kernel,'launchers',[]) or []
                                        runtime.append(dict(name=name,type=type(kernel).__name__,
                                            configs=configs(getattr(kernel,'configs',[])),
                                            launcher_configs=configs([getattr(l,'config',None) for l in launchers]),
                                            launcher_count=len(launchers)))
                                    event['norm_runtime']=dict(caller_file=frame.f_code.co_filename,
                                        caller_function=frame.f_code.co_name,kernels=runtime)
                            finally:
                                del frame
                    result=original(*args,**kwargs)
                    if selected:
                        me.tensor(f'gemm_0_output_{event["index"]}_{index}',result)
                    return result
                return captured
            for kind in ['mm','addmm']:
                original=getattr(extern_kernels,kind)
                self.original_slots.append((extern_kernels,kind,True,original))
                setattr(extern_kernels,kind,gemm_wrapper(kind,original))
        return dict(folder=str(self.folder),scope='CORRECTNESS_ONLY. Original forward and logits passthrough. CPU copies invalidate timing.')

    def close(self):
        from safetensors.torch import save_file
        for obj,key,existed,value in self.original_slots:
            if existed:setattr(obj,key,value)
            else:delattr(obj,key)
        file=self.folder/'tensors.safetensors';save_file(self.tensors,str(file))
        value=dict(time=time.time(),events=self.events,logits=self.logits,snapshot=self.snapshot,
            decode_input_copies_enabled=self.include_decode_inputs,
            first_layer_attention_copies_enabled=self.include_attention,
            captured_attention_layer=self.attention_layer,
            first_layer_GEMM_copies_enabled=self.include_gemm,
            tensor_file=str(file),tensor_SHA256=digest(file),evidence='CORRECTNESS_ONLY',
            scope='No model input,output,cache or sampled token was changed. This run cannot establish latency or throughput.')
        write(self.folder/'manifest.json',value)
        return dict(folder=str(self.folder),events=len(self.events),logits=len(self.logits),KV_snapshot=self.snapshot is not None)
