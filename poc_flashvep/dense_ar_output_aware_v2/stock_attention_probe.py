"""Read the original FA3 call at an opaque attention boundary for diagnosis."""
import hashlib
import inspect
import time
from poc_flashvep.dense_ar_longbench400_pd.common import ROOT,write

def install(cfg,role,info):
    if not cfg.get('stock_attention_probe'):return
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    original_prepare=GPUModelRunner.prepare_inputs;active={};seen=set();folder=ROOT/'raw'/cfg['name']
    def prepare(runner,*args,**kwargs):
        batch=original_prepare(runner,*args,**kwargs);active.update(batch=batch);return batch
    GPUModelRunner.prepare_inputs=prepare
    original=FlashAttentionImpl.forward;signature=inspect.signature(original)
    def forward(self,*args,**kwargs):
        bound=signature.bind(self,*args,**kwargs).arguments;batch=active.get('batch');layer=bound['layer']
        layer_name=getattr(layer,'layer_name',getattr(layer,'prefix',None))
        eligible=batch is not None and int(batch.num_reqs)==1 and batch.req_ids[0].split('-')[0]=='q0'
        eligible=eligible and layer_name=='model.layers.0.self_attn.attn' and not torch.cuda.is_current_stream_capturing()
        count=int(batch.num_scheduled_tokens[0]) if eligible else 0
        start=int(batch.num_computed_tokens_np[0]) if eligible else 0
        key=(start,count)
        eligible=eligible and count>1 and key not in seen
        # Run the unmodified attention first. Snapshot kernels never precede it.
        result=original(self,*args,**kwargs)
        if eligible:
            seen.add(key);metadata=bound['attn_metadata'];cache=bound['kv_cache']
            length=int(metadata.seq_lens[0].item());positions=torch.arange(length,device=cache.device)
            table=metadata.block_table[0];block=cache.shape[2]
            packed=cache[table[positions//block].long(),:,positions%block,:].detach().cpu()
            q=bound['query'][:count].detach().cpu();out=bound['output'][:count].detach().cpu()
            fields={}
            for field in ('num_actual_tokens','max_query_len','max_seq_len','max_num_splits','causal','use_cascade','sliding_window',
                    'query_start_loc','seq_lens','scheduler_metadata'):
                value=getattr(metadata,field,None)
                if isinstance(value,torch.Tensor):value=value.detach().cpu().tolist()
                fields[field]=value
            path=folder/f'FA3_layer0_{role}_start{start}_count{count}.pt'
            torch.save(dict(query=q,attention_output=out,packed_valid_KV=packed,query_start=start,query_count=count,
                scale=self.scale,num_heads=self.num_heads,num_kv_heads=self.num_kv_heads,head_size=self.head_size,
                metadata=fields),path)
            write(path.with_suffix('.json'),dict(role=role,layer=layer_name,query_start=start,query_count=count,
                query_shape=list(q.shape),output_shape=list(out.shape),valid_KV_shape=list(packed.shape),
                metadata=fields,time=time.monotonic(),evidence='DIAGNOSTIC',
                scope='Snapshot after the original opaque FA3 call. No input,output,weight,cache or model dispatch is corrected. '
                    'CPU copies perturb timing. Recheck whether the original numerical mismatch remains.'))
        return result
    FlashAttentionImpl.forward=forward
