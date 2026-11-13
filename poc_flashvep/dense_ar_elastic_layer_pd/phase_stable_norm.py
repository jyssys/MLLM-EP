"""Keep the existing Native normalization path for each token phase.

The installed fused normalization changes its reduction at256 rows despite
VLLM_BATCH_INVARIANT=1. Split only this pointwise/reduction operation. Keep
continuous scheduling, Attention, all GEMMs, and all model work unchanged.
"""
import atexit
from .common import *


def phase_runs(computed,counts,prompt_length):
    runs=[];cursor=0
    for start,count in zip(computed,counts):
        boundary=max(0,min(int(count),prompt_length-int(start)))
        for size,phase in ((boundary,'PREFILL'),(int(count)-boundary,'DECODE')):
            if not size:continue
            if runs and runs[-1][2]==phase:runs[-1]=(runs[-1][0],cursor+size,phase)
            else:runs.append((cursor,cursor+size,phase))
            cursor+=size
    return runs


def stable_fused(original,module,x,residual,runs):
    import torch
    assert x.ndim==2 and x.shape==residual.shape and runs[-1][1]==x.shape[0]
    for begin,end,phase in runs:
        a=x[begin:end];r=residual[begin:end];rows=end-begin
        if phase=='PREFILL' and rows<256:
            # Force the same original large-Prefill reduction. Padding affects
            # only independent normalization rows, never Attention or KV.
            padded_x=torch.empty((256,x.shape[1]),device=x.device,dtype=x.dtype)
            padded_r=torch.empty_like(padded_x)
            padded_x[:rows].copy_(a);padded_r[:rows].copy_(r)
            padded_x[rows:].zero_();padded_r[rows:].zero_()
            value,new_residual=original(module,padded_x,padded_r)
            a.copy_(value[:rows]);r.copy_(new_residual[:rows])
        elif phase=='DECODE' and rows>=256:
            for offset in range(begin,end,255):original(module,x[offset:min(offset+255,end)],residual[offset:min(offset+255,end)])
        else:original(module,a,r)
    return x,residual


def install(cfg,role):
    if not cfg.get('phase_stable_norm'):return
    import torch
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    folder=ROOT/'raw'/cfg['name'];active=dict(runs=None,tokens=0)
    counters=dict(prepared_mixed_batches=0,prepared_mixed_measurement_batches=0,
        measured_mixed_decode_rows=0,measured_mixed_prefill_rows=0,
        split_normalization_calls=0,padded_prefill_calls=0)
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        batch=prepare(self,*a,**kw)
        active['runs']=phase_runs(batch.num_computed_tokens_np,batch.num_scheduled_tokens,cfg['length'])
        active['tokens']=batch.num_tokens
        if len(active['runs'])>1:
            counters['prepared_mixed_batches']+=1
            if all(rid.split('-')[0].startswith('q') for rid in batch.req_ids):
                counters['prepared_mixed_measurement_batches']+=1
                for begin,end,phase in active['runs']:
                    counters['measured_mixed_'+phase.lower()+'_rows']+=end-begin
        return batch
    GPUModelRunner.prepare_inputs=prepared
    forward=RMSNorm.forward_cuda
    opaque=None
    if cfg.get('opaque_phase_norm'):
        from vllm import _custom_ops as ops
        assert cfg.get('ar_cudagraph_mode') in ('FULL_DECODE_ONLY','NONE')
        @torch.library.custom_op('elastic_pd::phase_add_rms_norm',mutates_args=('x','residual'))
        def phase_add_rms_norm(x:torch.Tensor,residual:torch.Tensor,weight:torch.Tensor,epsilon:float)->None:
            def operation(module,a,r):
                ops.fused_add_rms_norm(a,r,weight,epsilon);return a,r
            runs=active['runs']
            if not runs or x.ndim!=2 or x.shape[-1]!=4096 or x.shape[0]!=active['tokens']:
                operation(None,x,residual);return
            if len(runs)==1:
                rows=x.shape[0];phase=runs[0][2]
                if (phase=='DECODE' and rows<256) or (phase=='PREFILL' and rows>=256):
                    operation(None,x,residual);return
            counters['split_normalization_calls']+=1
            counters['padded_prefill_calls']+=sum(phase=='PREFILL' and end-begin<256 for begin,end,phase in runs)
            stable_fused(operation,None,x,residual,runs)
        @phase_add_rms_norm.register_fake
        def fake(x,residual,weight,epsilon):return None
        opaque=phase_add_rms_norm
    def normalized(self,x,residual=None):
        if opaque is not None and residual is not None:
            assert self.pass_weight_add and self.variance_size_override is None
            opaque(x,residual,self.weight,self.variance_epsilon)
            return x,residual
        runs=active['runs']
        if residual is None or not runs or x.ndim!=2 or x.shape[-1]!=4096 or x.shape[0]!=active['tokens']:
            return forward(self,x,residual)
        if len(runs)==1:
            rows=x.shape[0];phase=runs[0][2]
            if (phase=='DECODE' and rows<256) or (phase=='PREFILL' and rows>=256):
                return forward(self,x,residual)
        counters['split_normalization_calls']+=1
        counters['padded_prefill_calls']+=sum(phase=='PREFILL' and end-begin<256 for begin,end,phase in runs)
        return stable_fused(forward,self,x,residual,runs)
    RMSNorm.forward_cuda=normalized
    atexit.register(lambda:write(folder/f'phase_stable_norm_{role}.json',dict(counters=counters,opaque_custom_op=bool(opaque),
        evidence='STRUCTURAL_ACCOUNTING',scope='Normalization-only phase dispatch. Counts include warm-up. No model arithmetic, GEMM, Attention, token sampling, or cohort scheduling changes.')))
