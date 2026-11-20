"""Qualify exact partial continuation without the old cohort scheduler.

Initial execution is serial and correctness-only. The stock Native path never
installs this module. Online use needs a separate safe-admission qualification.
"""
import atexit
import hashlib
import time
from .common import ROOT,write


def token_prefix(length,cut):
    target=length*(36-cut)/36
    return min((length-1)//16*16,max(16,int(round(target/16))*16))


class Adapter:
    def __init__(self,cfg,role,info):
        self.cfg,self.role,self.info=cfg,role,info
        self.kind=cfg['axis_kind'];self.m=cfg['axis_cut'];self.cut=36-self.m
        self.folder=ROOT/'raw'/cfg['name'];self.active={};self.imports={};self.rows=[];self.states={}
        self.transport=None;self.closed=False;self.install()

    def attach(self,llm):
        if self.kind=='Layer':
            from .variable_hidden import HiddenTransport
            self.transport=HiddenTransport(self.cfg,self.role,self.folder)

    def action(self,request_id):
        info=self.info.get(request_id.split('-')[0],{})
        kind=info.get('axis_action',self.kind)
        return kind,36-int(info.get('axis_cut',self.m))

    def command(self,command):
        kind=command.get('axis_action',(command.get('kv_params') or {}).get('BI0_axis_action',self.kind))
        cut=command.get('axis_cut',(command.get('kv_params') or {}).get('BI0_axis_cut',self.m))
        command=dict(command,axis_action=kind,axis_cut=cut)
        if kind=='Token' and self.role=='prefill':
            length=command['original_length'];prefix=token_prefix(length,cut)
            assert len(command['input_ids'])==length and 0<prefix<length
            self.rows.append(dict(event='TOKEN_PREFIX_SUBMITTED',request_id=command['request_id'],
                original_length=length,prefix=prefix,suffix=length-prefix,
                actual_D_GEMM_rows=36*(length-prefix),target_D_GEMM_rows=length*cut))
            return dict(command,input_ids=command['input_ids'][:prefix],axis_original_length=length,
                original_length=prefix)
        return command

    def close(self):
        if self.closed:return
        if self.transport is not None:self.transport.close()
        write(self.folder/f'axis_events_{self.role}.json',dict(rows=self.rows,states=self.states,
            evidence='CORRECTNESS_ONLY',scope='Actual query and layer coverage. Partial KV transport, raw hidden/residual and stock full36 decode. No online performance claim.'))
        self.closed=True

    def install(self):
        import torch
        from vllm.model_executor.models.qwen3 import Qwen3Model,Qwen3ForCausalLM,Qwen3DecoderLayer
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
        me=self
        def valid_KV(runner,batch,index,length,layers=36):
            positions=torch.arange(length,device='cuda');table=runner.block_tables.input_block_tables[0][index];result=[]
            for number in range(layers):
                attn=runner.vllm_config.compilation_config.static_forward_context[f'model.layers.{number}.self_attn.attn']
                block=attn.kv_cache.shape[2]
                value=attn.kv_cache[table[positions//block].long(),:,positions%block,:].detach().contiguous().view(torch.uint8).cpu()
                result.append(dict(layer=number,SHA256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),bytes=value.numel()))
            return result
        prepare=GPUModelRunner.prepare_inputs
        def prepared(runner,*a,**kw):
            batch=prepare(runner,*a,**kw);me.active.update(batch=batch,runner=runner,layers=[])
            return batch
        GPUModelRunner.prepare_inputs=prepared
        execute=GPUModelRunner.execute_model
        def executed(runner,*a,**kw):
            try:return execute(runner,*a,**kw)
            finally:me.active.clear()
        GPUModelRunner.execute_model=executed
        layer_forward=Qwen3DecoderLayer.forward
        me.capture_layers=None
        def layer(module,*a,**kw):
            if me.active:me.active['layers'].append(int(module.self_attn.attn.layer_name.split('.')[2]))
            if me.capture_layers is not None:me.capture_layers.append(int(module.self_attn.attn.layer_name.split('.')[2]))
            return layer_forward(module,*a,**kw)
        Qwen3DecoderLayer.forward=layer
        if self.role=='decode':
            matched=NixlPullConnectorScheduler.get_num_new_matched_tokens
            def matching(connector,request,computed):
                kind,cut=me.action(request.request_id)
                if kind=='Token' and (request.kv_transfer_params or {}).get('do_remote_prefill'):
                    prefix=token_prefix(request.num_prompt_tokens,36-cut)
                    assert request.kv_transfer_params['remote_num_tokens']==prefix
                    return max(0,prefix-computed),prefix>computed
                return matched(connector,request,computed)
            NixlPullConnectorScheduler.get_num_new_matched_tokens=matching
        if self.kind=='Layer':
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import NixlPullConnectorWorker
            read=NixlPullConnectorWorker._read_blocks
            def reading(worker,*a,**kw):
                me.active_read=kw['request_id']
                try:return read(worker,*a,**kw)
                finally:me.active_read=None
            NixlPullConnectorWorker._read_blocks=reading
            me.active_read=None
            original_desc=NixlBaseConnectorWorker._compute_desc_ids
            def descriptors(worker,block_ids,dst_num_blocks,block_size_ratio,physical_blocks_per_logical):
                value=original_desc(worker,block_ids,dst_num_blocks,block_size_ratio,physical_blocks_per_logical)
                if me.active_read is None:return value
                kind,cut=me.action(me.active_read)
                if kind!='Layer':return value
                assert worker.num_regions%36==0 and not worker._has_mamba and not worker._is_hma_required
                blocks=dst_num_blocks if block_size_ratio is None else int(dst_num_blocks*block_size_ratio)
                selected=value[value//blocks<cut*(worker.num_regions//36)]
                assert len(selected)*36==len(value)*cut
                me.rows.append(dict(event='PARTIAL_KV_DESCRIPTORS',total=len(value),selected=len(selected),cut=cut,time=time.monotonic()))
                return selected
            NixlBaseConnectorWorker._compute_desc_ids=descriptors
            if self.role=='decode':
                ready=Scheduler._update_waiting_for_remote_kv
                def remote_ready(scheduler,request):
                    result=ready(scheduler,request)
                    if me.action(request.request_id)[0]!='Layer':return result
                    assert request.request_id not in scheduler.failed_recving_kv_req_ids
                    source=request.kv_transfer_params['remote_request_id']
                    assert (me.folder/'hidden_transport'/(source+'.json')).exists()
                    me.imports[request.request_id]=source;request.num_computed_tokens=0
                    return result
                Scheduler._update_waiting_for_remote_kv=remote_ready
                inputs=DefaultModelState.prepare_inputs
                def prepared_inputs(model_state,batch,req_states):
                    result=inputs(model_state,batch,req_states)
                    flags=batch.is_prefilling_np[:batch.num_reqs]
                    if not any(flags):return result
                    if not any(rid in me.imports for rid in batch.req_ids):return result
                    assert all(flags),'Do not silently mix partial-layer prefills with full-model decode'
                    hs=[];rs=[]
                    for i,rid in enumerate(batch.req_ids):
                        source=me.imports[rid];h,r=me.transport.get(source,me.action(rid)[1])
                        start=int(batch.num_computed_tokens_np[i]);count=int(batch.num_scheduled_tokens[i])
                        hs.append(h[start:start+count]);rs.append(r[start:start+count])
                    from vllm.sequence import IntermediateTensors
                    h=torch.cat(hs);r=torch.cat(rs)
                    valid=len(h);padded=int(batch.num_tokens_after_padding)
                    assert valid==int(batch.num_tokens) and valid<=padded
                    if valid<padded:
                        h=torch.cat((h,torch.zeros((padded-valid,h.shape[1]),device=h.device,dtype=h.dtype)))
                        r=torch.cat((r,torch.zeros((padded-valid,r.shape[1]),device=r.device,dtype=r.dtype)))
                        me.rows.append(dict(event='UNSCHEDULED_PADDING_ROWS',valid=valid,padded=padded,
                            scope='Padding has no request token, valid KV position or output. Its compute is recorded separately.'))
                    result['intermediate_tensors']=IntermediateTensors(dict(hidden_states=h,residual=r))
                    return result
                DefaultModelState.prepare_inputs=prepared_inputs
            model_forward=Qwen3Model.forward
            def model(module,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
                batch=me.active.get('batch')
                prefilling=batch is not None and bool(batch.is_prefilling_np[:batch.num_reqs].any())
                if batch is not None and not all(rid.split('-')[0] in me.info for rid in batch.req_ids):prefilling=False
                if not prefilling:return model_forward(module,input_ids,positions,intermediate_tensors,inputs_embeds)
                kind,cut=me.action(batch.req_ids[0])
                assert all(me.action(rid)==(kind,cut) for rid in batch.req_ids),'Group only equal layer actions'
                if kind!='Layer':return model_forward(module,input_ids,positions,intermediate_tensors,inputs_embeds)
                assert all(batch.is_prefilling_np[:batch.num_reqs])
                if me.role=='prefill':
                    assert intermediate_tensors is None
                    hidden=module.embed_input_ids(input_ids);residual=None
                    for index in range(cut):hidden,residual=module.layers[index](positions,hidden,residual)
                    for i,rid in enumerate(batch.req_ids):
                        a=int(batch.query_start_loc_np[i]);b=int(batch.query_start_loc_np[i+1])
                        length=me.info[rid.split('-')[0]]['original_length']
                        me.transport.put_part(rid,int(batch.num_computed_tokens_np[i]),length,hidden[a:b],residual[a:b],cut)
                    return hidden  # Producer sample is explicitly discarded by the official proxy protocol.
                assert intermediate_tensors is not None
                hidden=intermediate_tensors['hidden_states'];residual=intermediate_tensors['residual']
                for index in range(cut,36):hidden,residual=module.layers[index](positions,hidden,residual)
                hidden,_=module.norm(hidden,residual)
                return hidden
            Qwen3Model.forward=model
        outer=Qwen3ForCausalLM.forward
        def observed(module,*a,**kw):
            capture=torch.cuda.is_current_stream_capturing()
            if capture:me.capture_layers=[]
            value=outer(module,*a,**kw)
            if capture:
                actual=me.capture_layers;me.capture_layers=None
                assert actual==list(range(36)),'Decode graph must capture all36 original layers'
                me.rows.append(dict(event='CAPTURED_FULL_DECODE_GRAPH',layers=actual,passed=True,
                    scope='Captured graph definition. Actual replay counts are logged by the standard worker.'))
            batch=me.active.get('batch')
            if batch is None or not all(rid.split('-')[0] in me.info for rid in batch.req_ids):return value
            prefilling=bool(batch.is_prefilling_np[:batch.num_reqs].any())
            kind,cut=me.action(batch.req_ids[0])
            expected=list(range(cut)) if kind=='Layer' and me.role=='prefill' and prefilling else list(range(cut,36)) if kind=='Layer' and prefilling else list(range(36))
            assert me.active['layers']==expected,('Incorrect layer coverage',me.active['layers'],expected)
            for i,rid in enumerate(batch.req_ids):
                external=rid.split('-')[0]
                if external not in me.info:continue
                length=me.info[external]['original_length'];start=int(batch.num_computed_tokens_np[i]);count=int(batch.num_scheduled_tokens[i])
                me.rows.append(dict(event='MODEL_VISIT',request_id=external,start=start,count=count,
                    layers=list(me.active['layers']),axis_action=kind,axis_cut=36-cut,
                    actual_batch=int(batch.num_reqs),time=time.monotonic()))
                if start<length<=start+count and me.cfg.get('axis_state_capture',True):
                    row=int(batch.query_start_loc_np[i])+length-start-1
                    state=dict(length=length,KV=valid_KV(me.active['runner'],batch,i,length,
                        cut if kind=='Layer' and me.role=='prefill' else 36),
                        last_hidden_SHA256=hashlib.sha256(value[row].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                    me.states[external]=state
                if start<length<=start+count and kind=='Layer' and me.role=='decode':me.transport.retire(me.imports[rid])
            return value
        Qwen3ForCausalLM.forward=observed
        if self.role=='decode':
            attention=GPUModelRunner.prepare_attn
            def prepared_attention(runner,batch,*a,**kw):
                result=attention(runner,batch,*a,**kw)
                if not me.cfg.get('axis_state_capture',True):return result
                for i,rid in enumerate(batch.req_ids):
                    external=rid.split('-')[0]
                    if external not in me.info:continue
                    key='import_'+external
                    if key in me.states:continue
                    length=me.info[external]['original_length']
                    kind,cut=me.action(rid)
                    prefix=token_prefix(length,36-cut) if kind=='Token' else length
                    me.states[key]=dict(length=prefix,KV=valid_KV(runner,batch,i,prefix,cut if kind=='Layer' else 36))
                return result
            GPUModelRunner.prepare_attn=prepared_attention
        atexit.register(self.close)
