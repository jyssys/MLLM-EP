"""Public worker extension. Control patches apply only inside this experiment process."""
import os
from poc_looped_multigpu_kv_placement.worker_probe import KVWorkerProbe,inspect_worker
from poc_looped_recurrence_tp_4models.contract import *

if os.environ.get('POC4_FAMILY')=='tied':
    from poc_looped_recurrence_tp_4models.tied_tape import install
    install(int(os.environ['POC4_R']))


class FourModelProbe(KVWorkerProbe):
    def inspect_four(self):
        import hashlib
        import torch
        result=inspect_worker(self)
        model=self.get_model();layers=model.model.layers;geometry=[]
        for index,layer in enumerate(layers):
            attention=layer.self_attn;weight=attention.qkv_proj.weight
            q=attention.q_size;k=attention.kv_size
            def tensor_sha(value):return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            geometry.append(dict(layer=index,Q_heads=attention.num_heads,KV_heads=attention.num_kv_heads,
                head_dim=attention.head_dim,qkv_weight_shape=list(weight.shape),
                K_weight_sha256=tensor_sha(weight[q:q+k]),V_weight_sha256=tensor_sha(weight[q+k:q+2*k]),
                parameter_storage_ptr=weight.untyped_storage().data_ptr(),parameter_storage_bytes=weight.untyped_storage().nbytes()))
        result.update(physical_layers=len(layers),geometry=geometry,model_runner=type(self.model_runner).__module__,
            family=os.environ.get('POC4_FAMILY'),R=int(os.environ.get('POC4_R','1')),
            graph_manager=repr(getattr(self.model_runner,'cudagraph_manager',None)))
        if os.environ.get('POC4_MODEL')=='ouro':
            passes=int(os.environ['POC4_R'])
            result['execution_tape']=[dict(logical_layer=recurrent*48+layer,physical_layer=layer,
                loop=recurrent,module_identity=id(layers[layer]),
                qkv_storage_ptr=layers[layer].self_attn.qkv_proj.weight.untyped_storage().data_ptr(),
                cache_prefix=f'model.layers.{recurrent*48+layer}.self_attn.attn',
                residual_rule='Reset at pass start. Native norm after pass.')
                for recurrent in range(passes) for layer in range(48)]
        return result

    def teacher(self,tokens):
        """B1-only teacher forcing after native argmax, before native state update."""
        import torch
        sampler=self.model_runner.sampler
        assert sampler is not None and hasattr(sampler,'sample')
        self.teacher_tokens=None if tokens is None else torch.tensor(tokens,device=self.device,dtype=torch.int64)
        self.teacher_index=0
        if not hasattr(self,'original_sampler_sample'):
            self.original_sampler_sample=sampler.sample
            def sample(*args,**kwargs):
                sampled,processed=self.original_sampler_sample(*args,**kwargs)
                if self.teacher_tokens is not None:
                    assert sampled.numel()==1,'Hardware lane is strictly B1'
                    assert self.teacher_index<len(self.teacher_tokens)
                    sampled.copy_(self.teacher_tokens[self.teacher_index:self.teacher_index+1].view_as(sampled))
                    self.teacher_index+=1
                return sampled,processed
            sampler.sample=sample
        torch.cuda.synchronize()
        return True

    def start_parity_capture(self):
        self.captured_logits=[];self.captured_hidden=[]
        self.original_compute_logits=self.get_model().compute_logits
        def compute(hidden,*args,**kwargs):
            logits=self.original_compute_logits(hidden,*args,**kwargs)
            if len(self.captured_logits)<4:
                self.captured_logits.append(logits.detach().clone())
                self.captured_hidden.append(hidden.detach().clone())
            return logits
        self.get_model().compute_logits=compute
        return True

    def finish_parity_capture(self,label,prefix):
        import numpy as np
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        self.get_model().compute_logits=self.original_compute_logits
        torch.cuda.synchronize();runner=self.model_runner
        tables=runner.block_tables.input_block_tables
        assert len(tables)==1,'Primary full-attention layout has one cache group'
        block_table=tables[0][0].cpu().tolist()
        positions=[0,1,prefix-1,prefix,prefix+2]
        arrays={'logits':torch.cat(self.captured_logits).float().cpu().numpy(),
            'hidden':torch.cat(self.captured_hidden).float().cpu().numpy()}
        kv=[];names=[]
        for name,module in self.vllm_config.compilation_config.static_forward_context.items():
            cache=getattr(module,'kv_cache',None)
            if not isinstance(cache,torch.Tensor) or not cache.numel():continue
            assert cache.ndim==4 and cache.shape[-1]==256 and cache.shape[2]==16
            # Select actual written token positions through the native input block table.
            sample=torch.stack([cache[block_table[p//16],:min(2,cache.shape[1]),p%16,
                torch.tensor([0,1,2,3,124,125,126,127,128,129,130,131,252,253,254,255],device=cache.device)] for p in positions])
            kv.append(sample.float().cpu().numpy());names.append(name)
        arrays['KV']=np.stack(kv);rank=get_tensor_model_parallel_rank()
        path=Path(os.environ['LOOPED_KV_OUTPUT'])/f'{label}.rank{rank}.npz'
        np.savez_compressed(path,**arrays)
        return dict(rank=rank,path=str(path),sha256=sha(path),cache_slots=len(kv),
            KV_names=names,positions=positions,block_ids=[block_table[p//16] for p in positions],
            logits_calls=len(self.captured_logits),teacher_steps=self.teacher_index,
            scope='After-request CPU transfer. Full logits, selected hidden, and sampled written KV. Not clean timing.')

    def runtime_counts_start(self):
        import torch
        self.runtime_counts=dict(model_forwards=0,sampling_calls=0,graph_replays=0)
        self.count_originals=[]
        def wrap(obj,name,key):
            original=getattr(obj,name)
            def counted(*args,**kwargs):
                self.runtime_counts[key]+=1
                return original(*args,**kwargs)
            setattr(obj,name,counted);self.count_originals.append((obj,name,original))
        wrap(self.model_runner,'execute_model','model_forwards')
        wrap(self.model_runner,'sample_tokens','sampling_calls')
        wrap(torch.cuda.CUDAGraph,'replay','graph_replays')
        return True

    def runtime_counts_finish(self):
        for obj,name,original in reversed(self.count_originals):setattr(obj,name,original)
        return self.runtime_counts
