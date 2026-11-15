"""Update a GPU phase mask before replaying the original full-model graph."""
import atexit,collections
from .common import *


def install(cfg,role):
    assert not cfg.get('variable_input'),'Qualify variable-length mask metadata separately'
    import torch,numpy as np
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from .phase_stable_norm import phase_runs
    from .phase_mask_norm import load
    gate=json.loads((ROOT/'analysis/phase_mask_norm_GPU_gate.json').read_text())
    assert gate['passed'] and gate['kernel_source_SHA256']==digest(SOURCE/'phase_mask_norm.py')
    assert gate['original_body_SHA256']==digest(SOURCE/'exact_phase_norm.py')
    module=load();maximum=max(cfg['token_budget'],cfg.get('D_token_budget',0) if role=='decode' else 0,
        max(cfg.get('explicit_cudagraph_capture_sizes',[1])))
    phases=torch.zeros(maximum,device='cuda',dtype=torch.int32);retained=collections.deque()
    state=dict(batch=None,prefill_rows=0,decode_rows=0)
    counters=dict(phase_mask_updates=0,fullgraph_replays=0,mixed_fullgraph_replays=0)
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        batch=prepare(self,*a,**kw);state['batch']=batch
        assert batch.num_tokens<=maximum
        host=torch.ones(maximum,pin_memory=True,dtype=torch.int32);labels=host.numpy()
        runs=phase_runs(batch.num_computed_tokens_np,batch.num_scheduled_tokens,cfg['length'])
        state['prefill_rows']=0;state['decode_rows']=0
        for begin,end,phase in runs:
            labels[begin:end]=0 if phase=='PREFILL' else 1
            state['prefill_rows' if phase=='PREFILL' else 'decode_rows']+=end-begin
        phases.copy_(host,non_blocking=True)
        event=torch.cuda.Event();event.record();retained.append((host,event))
        while retained and retained[0][1].query():retained.popleft()
        counters['phase_mask_updates']+=1
        return batch
    GPUModelRunner.prepare_inputs=prepared
    @torch.library.custom_op('slo_pd::masked_add_rms_norm',mutates_args=('x','residual'))
    def op(x:torch.Tensor,residual:torch.Tensor,weight:torch.Tensor,mask:torch.Tensor,epsilon:float)->None:
        module.masked_norm(x,residual,weight,mask,epsilon)
    @op.register_fake
    def fake(x,residual,weight,mask,epsilon):return None
    unfused=None
    if cfg.get('opaque_unfused_norm'):
        from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
        @torch.library.custom_op('slo_pd::original_unfused_norm',mutates_args=())
        def original_unfused(x:torch.Tensor,weight:torch.Tensor,epsilon:float)->torch.Tensor:
            return rms_norm_batch_invariant(x,weight,epsilon)
        @original_unfused.register_fake
        def fake_unfused(x,weight,epsilon):
            return torch.empty(x.shape,device=x.device,dtype=x.dtype)
        unfused=original_unfused
    original=RMSNorm.forward_cuda
    def normalized(self,x,residual=None):
        if residual is None:
            if unfused is not None:
                assert self.pass_weight and self.variance_size_override is None
                return unfused(x,self.weight,self.variance_epsilon)
            return original(self,x,residual)
        assert self.pass_weight_add and self.variance_size_override is None and x.shape[-1]==4096
        op(x,residual,self.weight,phases,self.variance_epsilon);return x,residual
    RMSNorm.forward_cuda=normalized
    replay=ModelCudaGraphManager.run_fullgraph
    def observed(self,*a,**kw):
        value=replay(self,*a,**kw);batch=state['batch']
        if batch is not None and any(rid.split('-')[0].startswith('q') for rid in batch.req_ids):
            counters['fullgraph_replays']+=1
            if state['prefill_rows'] and state['decode_rows']:counters['mixed_fullgraph_replays']+=1
        return value
    ModelCudaGraphManager.run_fullgraph=observed
    def close():
        torch.cuda.synchronize();retained.clear()
        write(ROOT/'raw'/cfg['name']/f'GPU_phase_mask_{role}.json',dict(counters=counters,
            normalization_source_SHA256=digest(SOURCE/'phase_mask_norm.py'),evidence='STRUCTURAL_ACCOUNTING',
            scope='Existing full-model graph replays with phase-correct normalization. Replay counts are not kernel-union time. Fresh pinned source buffers remain owned until their copy event completes.'))
    atexit.register(close)
