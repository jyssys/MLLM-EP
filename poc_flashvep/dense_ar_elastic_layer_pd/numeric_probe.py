"""Observe one Decode query without changing its model computation."""
import atexit
from .common import *


def install(cfg,role):
    if not cfg.get('numeric_probe'):return
    import torch
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM,Qwen3DecoderLayer
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    assert cfg.get('ar_enforce_eager') and cfg.get('ar_cudagraph_mode')=='NONE'
    folder=ROOT/'raw'/cfg['name'];active=dict(batch=None,runner=None,done=False,selected=False)
    result=dict(role=role,layers={},Attention={},scope='Diagnostic tensor observations. CPU copies alter timing. No serving performance claim.')
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        batch=prepare(self,*a,**kw);active.update(batch=batch,runner=self,selected=False)
        for i,rid in enumerate(batch.req_ids):
            if rid.split('-')[0]==cfg.get('numeric_target','q1') and not active['done']:
                if int(batch.num_computed_tokens_np[i])==cfg['length'] and int(batch.num_scheduled_tokens[i])==1:
                    row=int(batch.query_start_loc_np[i]);active.update(selected=True,index=i,row=row)
                    result['batch']=dict(requests=batch.req_ids,query_lengths=batch.num_scheduled_tokens.tolist(),
                        computed=batch.num_computed_tokens_np.tolist(),query_row=row,
                        token=int(batch.input_ids[row]),position=int(batch.positions[row]))
        return batch
    GPUModelRunner.prepare_inputs=prepared
    forward=Qwen3ForCausalLM.forward
    def model(self,*a,**kw):
        if not active.get('hooks_installed'):
            active['hooks_installed']=True
            for name,module in self.named_modules():
                if not name.startswith('model.layers.23.'):continue
                if name.rsplit('.',1)[-1] not in ('o_proj','input_layernorm','post_attention_layernorm','gate_up_proj','act_fn','down_proj'):continue
                def observed(module,inputs,output,name=name):
                    if not active['selected']:return
                    row=active['row'];items=output if isinstance(output,tuple) else (output,)
                    result.setdefault('submodules',{})[name]=dict(
                        inputs=[v[row].detach().cpu() if isinstance(v,torch.Tensor) else v for v in inputs],
                        outputs=[v[row].detach().cpu() if isinstance(v,torch.Tensor) else v for v in items])
                    if name.endswith('down_proj'):
                        result['GEMM_fixture']=dict(input=inputs[0].detach().cpu(),weight=module.weight.detach().cpu(),
                            output=items[0].detach().cpu(),row=row)
                module.register_forward_hook(observed)
        if active['selected']:active['layer_ids']={id(layer):i for i,layer in enumerate(self.model.layers)}
        value=forward(self,*a,**kw)
        if active['selected']:
            runner=active['runner'];i=active['index'];table=runner.block_tables.input_block_tables[0][i]
            positions=torch.arange(cfg['length']+1,device=value.device);planes=[]
            for index in range(36):
                attn=runner.vllm_config.compilation_config.static_forward_context[f'model.layers.{index}.self_attn.attn']
                cache=attn.kv_cache;valid=cache[table[positions//16].long(),:,positions%16,:].detach().cpu()
                prefix=valid[:-1].contiguous().view(torch.uint8).numpy().tobytes()
                entry=dict(layer=index,prefix_SHA256=hashlib.sha256(prefix).hexdigest(),
                    sample=valid[torch.tensor(list(range(16))+list(range(cfg['length']-16,cfg['length']+1)))].clone())
                if index in cfg.get('save_full_KV_layers',(0,1)):entry['all_valid_KV']=valid.clone()
                planes.append(entry)
            result['KV']=planes;result['last_hidden']=value[active['row']].detach().cpu()
        return value
    Qwen3ForCausalLM.forward=model
    original_layer=Qwen3DecoderLayer.forward
    def layer(self,*a,**kw):
        selected=active['selected'];row=active.get('row',0)
        if selected:
            hidden=a[1] if len(a)>1 else kw['hidden_states'];residual=a[2] if len(a)>2 else kw.get('residual')
            data=dict(input_hidden=hidden[row].detach().cpu(),input_residual=residual[row].detach().cpu() if residual is not None else None)
        out=original_layer(self,*a,**kw)
        if selected:
            data.update(output_hidden=out[0][row].detach().cpu(),output_residual=out[1][row].detach().cpu())
            result['layers'][active['layer_ids'][id(self)]]=data
        return out
    Qwen3DecoderLayer.forward=layer
    attention=FlashAttentionImpl.forward
    def attended(self,layer,query,key,value,cache,metadata,output,*a,**kw):
        selected=active['selected'];row=active.get('row',0)
        if selected:
            data=dict(Q=query[row].detach().cpu(),K=key[row].detach().cpu(),V=value[row].detach().cpu(),
                max_query_len=metadata.max_query_len,max_seq_len=metadata.max_seq_len)
        out=attention(self,layer,query,key,value,cache,metadata,output,*a,**kw)
        if selected:
            data['output']=output[row].detach().cpu();result['Attention'][layer.layer_name]=data
        return out
    FlashAttentionImpl.forward=attended
    compute=Qwen3ForCausalLM.compute_logits
    def logits(self,*a,**kw):
        value=compute(self,*a,**kw)
        if active['selected'] and not active['done']:
            i=int(active['batch'].cu_num_logits_np[active['index']])
            result['logits']=value[i].detach().cpu();result['top10']=torch.topk(result['logits'],10)
            active['done']=True;active['selected']=False
        return value
    Qwen3ForCausalLM.compute_logits=logits
    def close():
        if 'logits' in result:
            path=folder/f'numeric_probe_{role}.pt';torch.save(result,path)
            write(folder/f'numeric_probe_{role}.json',dict(SHA256=digest(path),batch=result['batch'],
                layers=len(result['layers']),Attention_layers=len(result['Attention']),
                greedy_token=int(result['logits'].argmax()),evidence='CORRECTNESS_ONLY'))
    atexit.register(close)
