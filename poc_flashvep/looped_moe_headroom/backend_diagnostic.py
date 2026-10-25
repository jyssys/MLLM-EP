"""Diagnostic FP32 attention reference. This is not a serving baseline."""
import contextlib
import math
import torch
import torch.nn.functional as F

@contextlib.contextmanager
def fp32_reference(replace=False):
    import xllm.modules.causal_attention as source
    original=source.flash_attention
    records=[]
    def compared(q,k,v,cuq,cuk,maxq,maxk,scale,dropout,causal,deterministic,training):
        assert cuq is None and cuk is None and not training and dropout==0
        out=original(q,k,v,cuq,cuk,maxq,maxk,scale,dropout,causal,deterministic,training)
        # This reference changes arithmetic precision for diagnosis only.
        # It uses the actual pretrained model's post-RoPE Q/K/V.
        qh=q.transpose(1,2).float()
        kh=k.transpose(1,2).float().repeat_interleave(q.shape[2]//k.shape[2],dim=1)
        vh=v.transpose(1,2).float().repeat_interleave(q.shape[2]//v.shape[2],dim=1)
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            scores=qh@kh.transpose(-1,-2)*scale
            mask=torch.ones(q.shape[1],k.shape[1],dtype=torch.bool,device=q.device).triu(1)
            scores=scores.masked_fill(mask,float('-inf'))
            ref=(scores.softmax(-1)@vh).transpose(1,2)
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_tf32
        diff=out.float()-ref
        records.append(dict(index=len(records),shape=list(q.shape),dtype=str(out.dtype),
            relative_l2=float(diff.norm()/ref.norm().clamp_min(1e-8)),
            max_absolute_error=float(diff.abs().max()),
            rmse=float(diff.square().mean().sqrt())))
        return ref.to(out.dtype) if replace else out
    source.flash_attention=compared
    try:
        yield records
    finally:
        source.flash_attention=original

def run(a,ids):
    from .adapters import quality
    from .probes import Controller
    small=ids[:,:32]
    with Controller(a,'probe') as baseline_probe:
        native=a.forward(small)
        native_frames=baseline_probe.frames
    with fp32_reference(replace=False) as operator:
        untouched=a.forward(small)
    assert torch.equal(native,untouched),'Diagnostic observer changed the native output'
    with fp32_reference(replace=True):
        with Controller(a,'probe') as reference_probe:
            reference=a.forward(small)
            reference_frames=reference_probe.frames
        ref_generation=a.generate(small,8,stop_eos=False)
    native_generation=a.generate(small,8,stop_eos=False)
    # Freeze independent operator and token checks before their first execution.
    operator_pass=all(row['relative_l2']<=.01 for row in operator)
    route_differences=[]
    for left,right in zip(native_frames,reference_frames):
        assert (left['site'],left['pass_index'])==(right['site'],right['pass_index'])
        mismatch=int((left['full_ids']!=right['full_ids']).sum())
        if mismatch:
            route_differences.append(dict(site=left['site'],pass_index=left['pass_index'],
                changed_router_slots=mismatch,router_slots=left['full_ids'].numel()))
    return dict(operator_reference='Actual post-RoPE Q/K/V, explicit FP32 softmax attention, BF16 output boundary',
        operator_relative_l2_limit=.01,operator_pass=operator_pass,operator_records=operator,
        diagnostic_tf32=False,native_tf32_policy_restored=True,
        whole_model_fp32_attention_diagnostic=quality(native,reference,small),
        whole_model_relative_l2=float((native.float()-reference.float()).norm()/reference.float().norm()),
        native_generation=native_generation,fp32_attention_generation=ref_generation,
        greedy_8_token_parity=native_generation['ids']==ref_generation['ids'],
        observed_routing_differences=route_differences,
        pass_independent_checks=operator_pass and native_generation['ids']==ref_generation['ids'],
        limitation='Does not change the failed BF16-math whole-model L2<=0.05 result. Not a quality equivalence claim.')
