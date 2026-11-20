"""Save selected actual tensors without modifying the BI0 model math."""
import atexit
import time
from .common import ROOT,write


def computed_layers(role,metadata,cfg):
    kind=metadata.get('axis_action',cfg.get('axis_kind'))
    count=36-int(metadata.get('axis_cut',cfg.get('axis_cut',0))) if role=='prefill' and kind=='Layer' else 36
    assert 0<count<=36
    return count


def install(cfg,role,info):
    if cfg.get('token_teacher_cut'):
        from .token_geometry_teacher import install as teacher
        teacher(cfg,role,info)
    if not cfg.get('numerical_probe'):
        return
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    folder=ROOT/'raw'/cfg['name'];seen=set();rows=[]
    descriptor={}
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*args,**kwargs):
        if len(args)>1:descriptor['value']=str(args[1])
        return prepare(self,*args,**kwargs)
    GPUModelRunner.prepare_inputs=prepared
    original=GPUModelRunner.execute_model
    def execute(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if kwargs.get('dummy_run') or torch.cuda.is_current_stream_capturing():
            return result
        frame=self.execute_model_state
        if frame is None or frame.hidden_states is None:
            return result
        if cfg.get('numerical_probe_sync'):torch.cuda.synchronize()
        batch=frame.input_batch
        for index,rid in enumerate(batch.req_ids):
            external=rid.split('-')[0];metadata=info.get(external)
            if metadata is None or not external.startswith('q') or external in seen:
                continue
            length=metadata['original_length'];start=int(batch.num_computed_tokens_np[index]);count=int(batch.num_scheduled_tokens[index])
            if not start<length<=start+count:
                continue
            offset=int(batch.query_start_loc_np[index])+length-start-1
            hidden=frame.hidden_states[offset].detach().clone().cpu()
            positions=torch.linspace(0,length-1,min(length,32),device='cuda').round().long().unique()
            table=self.block_tables.input_block_tables[0][index];planes=[]
            valid_layers=computed_layers(role,metadata,cfg)
            for layer in range(valid_layers):
                attn=self.vllm_config.compilation_config.static_forward_context[f'model.layers.{layer}.self_attn.attn']
                block=attn.kv_cache.shape[2]
                planes.append(attn.kv_cache[table[positions//block].long(),:,positions%block,:].detach().clone().cpu())
            path=folder/f'numerical_{role}_{external}.pt'
            meta={}
            for key,value in (frame.attn_metadata or {}).items():
                summary={}
                for field in ('num_actual_tokens','max_query_len','max_seq_len','num_decode_reqs','num_prefill_reqs',
                    'num_decode_tokens','num_prefill_tokens','max_num_splits','use_cascade','common_prefix_len','causal','sliding_window',
                    'seq_lens','query_start_loc'):
                    part=getattr(value,field,None)
                    if isinstance(part,torch.Tensor):part=part.detach().cpu().tolist() if part.numel()<1024 else dict(shape=list(part.shape))
                    if isinstance(part,(str,int,float,bool,list,tuple,dict)) or part is None:summary[field]=part
                meta[key]=summary
                if len(meta)>=2:break
            torch.save(dict(hidden_last=hidden,KV_samples=torch.stack(planes),positions=positions.cpu(),
                            length=length,query_start=start,query_count=count,
                            valid_layer_indices=list(range(valid_layers)),
                            device_synchronized=cfg.get('numerical_probe_sync',False),batch_descriptor=descriptor.get('value'),attention_metadata=meta),path)
            rows.append(dict(request_id=external,path=path.name,positions=positions.cpu().tolist(),
                length=length,query_start=start,query_count=count,time=time.monotonic()))
            seen.add(external)
        return result
    GPUModelRunner.execute_model=execute
    atexit.register(lambda:write(folder/f'numerical_probe_{role}.json',dict(rows=rows,
        evidence='CORRECTNESS_ONLY',scope='Last prompt hidden and32 valid KV positions/computed layer after prompt completion. '
        'Producer partial-layer probes never read uncomputed suffix-layer cache. '
        'D includes its final prompt query recomputation. Sampled tolerance is not a full-KV comparison. CPU copies invalidate timings.')))
