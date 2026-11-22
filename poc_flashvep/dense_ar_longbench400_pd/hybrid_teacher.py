"""Independent one-GPU three-wave teacher for short original prompts.

This diagnostic uses original Qwen layers and full local KV ownership. It does
not import the partition adapter or use NIXL. The first qualification is C1 and
the whole prompt must fit one B8192 prefill iteration.
"""
import copy
import numpy as np
import torch


def install(cfg,role,info):
    if 'local_hybrid_teacher_m' not in cfg:return
    assert role=='single' and cfg['serial'] and not cfg.get('axis_kind')
    from vllm.model_executor.models.qwen3 import Qwen3Model
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
    from vllm.forward_context import get_forward_context,override_forward_context
    from vllm.config import CUDAGraphMode
    original_prepare=GPUModelRunner.prepare_inputs;original_execute=GPUModelRunner.execute_model
    original_model=Qwen3Model.forward;active={}
    def prepare(runner,*a,**kw):
        batch=original_prepare(runner,*a,**kw);active.update(runner=runner,batch=batch);return batch
    def execute(runner,*a,**kw):
        try:return original_execute(runner,*a,**kw)
        finally:active.clear()
    GPUModelRunner.prepare_inputs=prepare;GPUModelRunner.execute_model=execute
    def model(module,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        batch=active.get('batch')
        if batch is None or not any(batch.is_prefilling_np[:batch.num_reqs]):return original_model(module,input_ids,positions,intermediate_tensors,inputs_embeds)
        if not all(rid.split('-')[0] in info for rid in batch.req_ids):return original_model(module,input_ids,positions,intermediate_tensors,inputs_embeds)
        assert batch.num_reqs==1 and int(batch.num_computed_tokens_np[0])==0
        length=info[batch.req_ids[0].split('-')[0]]['original_length']
        assert length<=8192 and int(batch.num_tokens)==length
        assert int(batch.num_tokens_after_padding)==length
        m=int(cfg['local_hybrid_teacher_m']);k=36-m
        s=max(16,int(round(length*cfg['local_hybrid_teacher_fraction']/16))*16);T=length-s
        runner=active['runner'];base=get_forward_context();first=next(iter(base.attn_metadata.values()))
        # Preserve the original full-query schedule. The builder owns a mutable AOT buffer.
        full=copy.copy(base);full.attn_metadata={}
        for name,value in base.attn_metadata.items():
            cloned=copy.copy(value)
            if cloned.scheduler_metadata is not None:cloned.scheduler_metadata=cloned.scheduler_metadata.clone()
            full.attn_metadata[name]=cloned
        def context(left,right,end):
            qcpu=torch.tensor([0,right-left],dtype=torch.int32);qgpu=qcpu.to('cuda')
            slots=next(iter(base.slot_mapping.values()))[left:right]
            seq_cpu=torch.tensor([end],dtype=torch.int32);seq_gpu=seq_cpu.to('cuda')
            metadata=build_attn_metadata(runner.attn_groups,1,right-left,qgpu,qcpu,right-left,seq_gpu,end,
                (first.block_table[:1],),slots.unsqueeze(0),runner.kv_cache_config,
                seq_lens_cpu_upper_bound=seq_cpu,positions=positions[left:right])
            local=copy.copy(base);local.attn_metadata=metadata
            local.slot_mapping={name:slots for name in base.slot_mapping};local.is_padding=None
            local.cudagraph_runtime_mode=CUDAGraphMode.NONE;local.batch_descriptor=None;local.skip_compiled=True
            return local
        prefix_h=module.embed_input_ids(input_ids[:T]);prefix_r=None
        with override_forward_context(context(0,T,T)):
            for layer in range(k):prefix_h,prefix_r=module.layers[layer](positions[:T],prefix_h,prefix_r)
        suffix_h=module.embed_input_ids(input_ids[T:length]);suffix_r=None
        with override_forward_context(context(T,length,length)):
            for layer in range(k):suffix_h,suffix_r=module.layers[layer](positions[T:length],suffix_h,suffix_r)
        hidden=torch.cat((prefix_h,suffix_h));residual=torch.cat((prefix_r,suffix_r))
        with override_forward_context(full):
            for layer in range(k,36):hidden,residual=module.layers[layer](positions,hidden,residual)
        hidden,_=module.norm(hidden,residual)
        return hidden
    Qwen3Model.forward=model
