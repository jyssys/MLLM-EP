"""Reuse audited DeepEP timing/ownership hooks, retaining only new rows."""
import os
import json
from pathlib import Path
import torch
from poc_speculative_moe_ep import instrumentation as shared


def install():
    os.environ['SD_EP_CONTROL']=os.environ['AGENT_EP_CONTROL']
    os.environ['SD_EP_OUTPUT']=os.environ['AGENT_EP_OUTPUT']
    shared.install()
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl as Kernel
    prepare=GPUModelRunner._prepare_inputs;apply=Kernel.apply
    original_flush=Worker.sd_ep_flush

    def flush(worker):
        result=original_flush(worker)
        rank=get_ep_group().rank_in_group
        free,total=torch.cuda.mem_get_info()
        (Path(os.environ['AGENT_EP_OUTPUT'])/f'runtime_memory.rank{rank}.json').write_text(json.dumps(dict(
            free_bytes=free,total_bytes=total,torch_allocated=torch.cuda.memory_allocated(),
            cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
            boundary='after serving wave + deferred drain; includes loaded model and8GiB KV reservation'))+'\n')
        return result
    Worker.sd_ep_flush=flush

    def inputs(self,scheduler_output,num_scheduled_tokens,*args,**kwargs):
        out=prepare(self,scheduler_output,num_scheduled_tokens,*args,**kwargs)
        assert self.input_batch.num_reqs==1, 'single semantic request trace required'
        start=int(self.input_batch.num_computed_tokens_cpu[0])
        count=int(num_scheduled_tokens[0])
        shared.TLS.query_start=start;shared.TLS.query_count=count
        return out

    def routed(self,*args,**kwargs):
        out=apply(self,*args,**kwargs)
        frame=getattr(shared.TLS,'frame',None)
        if frame and frame['lane']=='trace':
            row=frame['layers'][-1]
            ids,weights=row['route']
            c=shared.control();source_offset=frame['tp_rank']*row['source_rows']
            query_start=getattr(shared.TLS,'query_start',0)
            local_start=query_start+source_offset
            valid=row['valid_source_rows']
            lo=max(0,min(valid,c['exclusion_start']-local_start))
            hi=max(lo,min(valid,c['prompt_tokens']+c['decode_tokens']-local_start))
            if frame['dp_rank']==0 and not frame.get('scheduler',{}).get('scheduled_tokens'):
                lo=hi=0
            if frame['dp_rank']!=0:lo=hi=0
            row['route']=(ids[lo:hi].clone(),weights[lo:hi].clone())
            row.update(captured_rows=hi-lo,captured_position_start=local_start+lo,
                discarded_common_rows=lo,source_offset=source_offset,
                actual_query_start=query_start,actual_query_count=getattr(shared.TLS,'query_count',0))
        return out
    GPUModelRunner._prepare_inputs=inputs
    Kernel.apply=routed
