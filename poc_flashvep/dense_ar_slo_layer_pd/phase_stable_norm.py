"""Keep the existing Native normalization path for each token phase.

The installed fused normalization changes its reduction at256 rows despite
VLLM_BATCH_INVARIANT=1. Split only this pointwise/reduction operation. Keep
continuous scheduling, Attention, all GEMMs, and all model work unchanged.
"""
import atexit
from .common import *
PREFIX_LENGTHS={}


def phase_runs(computed,counts,prompt_length):
    runs=[];cursor=0
    lengths=prompt_length if isinstance(prompt_length,(list,tuple)) else [prompt_length]*len(counts)
    for start,count,length in zip(computed,counts,lengths):
        boundary=max(0,min(int(count),length-int(start)))
        for size,phase in ((boundary,'PREFILL'),(int(count)-boundary,'DECODE')):
            if not size:continue
            if runs and runs[-1][2]==phase:runs[-1]=(runs[-1][0],cursor+size,phase)
            else:runs.append((cursor,cursor+size,phase))
            cursor+=size
    return runs


def stable_fused(original,module,x,residual,runs,direct_op=None):
    import torch
    assert x.ndim==2 and x.shape==residual.shape and runs[-1][1]==x.shape[0]
    for begin,end,phase in runs:
        a=x[begin:end];r=residual[begin:end];rows=end-begin
        if phase=='PREFILL' and rows<256:
            if direct_op is not None:
                assert module is not None and module.pass_weight_add and module.variance_size_override is None
                direct_op(a,r,module.weight,module.variance_epsilon)
                continue
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
    if cfg.get('gpu_phase_mask_norm') and role in cfg.get('phase_mask_norm_roles',
        ('prefill','decode','replica0','replica1')):
        from .phase_mask_runtime import install as masked
        return masked(cfg,role)
    import torch
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    direct_op=None
    if cfg.get('direct_phase_norm'):
        assert not cfg.get('opaque_phase_norm'),'Use the qualified no-compile normalization path'
        from .exact_phase_norm import load
        gate=json.loads((ROOT/'analysis/exact_phase_norm_GPU_gate.json').read_text())
        assert gate['passed'],'The explicit-launch kernel needs its independent exactness gate'
        assert gate['kernel_source_SHA256']==digest(SOURCE/'exact_phase_norm.py'),'The kernel changed after qualification'
        direct_op=load().phase_norm
    if cfg.get('variable_input'):
        from vllm.v1.engine.input_processor import InputProcessor
        assign=InputProcessor.assign_request_id
        def record_length(request):
            value=assign(request)
            params=(request.sampling_params.extra_args or {}).get('kv_transfer_params',{}) if request.sampling_params else {}
            PREFIX_LENGTHS[request.request_id]=int(params.get('dense_source_prefix_tokens',len(request.prompt_token_ids)))
            return value
        InputProcessor.assign_request_id=record_length
    folder=ROOT/'raw'/cfg['name'];active=dict(runs=None,tokens=0)
    counters=dict(prepared_mixed_batches=0,prepared_mixed_measurement_batches=0,
        measured_mixed_decode_rows=0,measured_mixed_prefill_rows=0,
        split_normalization_calls=0,padded_prefill_calls=0)
    prepare=GPUModelRunner.prepare_inputs
    def prepared(self,*a,**kw):
        batch=prepare(self,*a,**kw)
        lengths=([PREFIX_LENGTHS.get(rid,int(self.req_states.prompt_len.np[self.req_states.req_id_to_index[rid]]))
            for rid in batch.req_ids] if cfg.get('variable_input') else cfg['length'])
        active['runs']=phase_runs(batch.num_computed_tokens_np,batch.num_scheduled_tokens,lengths)
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
        return stable_fused(forward,self,x,residual,runs,direct_op)
    RMSNorm.forward_cuda=normalized
    atexit.register(lambda:write(folder/f'phase_stable_norm_{role}.json',dict(counters=counters,opaque_custom_op=bool(opaque),
        direct_256_thread_short_prefill=bool(direct_op),
        evidence='STRUCTURAL_ACCOUNTING',scope='Normalization-only phase dispatch. Counts include warm-up. No model arithmetic, GEMM, Attention, token sampling, or cohort scheduling changes.')))

