"""Same-input, bounded attention geometry and output-projection references."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu
from .numerical_trace import errors
from .contract import *


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    model=load_cpu().cuda();remote=__import__(type(model).__module__,fromlist=['ALL_ATTENTION_FUNCTIONS'])
    original=remote.ALL_ATTENTION_FUNCTIONS['sdpa'];current={};records=[];step=0
    def context(index):
        def before(module,args,kwargs):
            current.update(layer=index,loop=kwargs['current_ut'])
        return before
    handles=[layer.register_forward_pre_hook(context(i),with_kwargs=True) for i,layer in enumerate(model.model.layers)]
    def diagnostic(module,q,k,v,mask,**kw):
        result=original(module,q,k,v,mask,**kw)
        if step not in [0,18,19,26,27] or current['layer'] not in [7,8,9] or current['loop'] not in [1,2,3]:
            return result
        chunks=torch.cat([original(module,q[:,i:i+4],k[:,i:i+4],v[:,i:i+4],mask,**kw)[0] for i in range(0,16,4)],dim=2)
        causal=kw.get('is_causal')
        if causal is None:causal=q.shape[2]>1 and mask is None and getattr(module,'is_causal',True)
        with sdpa_kernel(SDPBackend.MATH):
            gold=F.scaled_dot_product_attention(q.float(),k.float(),v.float(),attn_mask=None if mask is None else mask.float(),
                dropout_p=0.,is_causal=causal,scale=kw.get('scaling')).transpose(1,2).contiguous()
        actual=result[0];W=module.o_proj.weight
        x=actual.reshape(*actual.shape[:-2],-1);xf=gold.reshape(*gold.shape[:-2],-1)
        reference=F.linear(xf.float(),W.float())
        bf16=F.linear(x,W)
        same_input_fp32=F.linear(x.float(),W.float())
        partials=sum(F.linear(x[...,i:i+512].float(),W[:,i:i+512].float()) for i in range(0,2048,512))
        record=dict(source=source_label,step=step,loop=current['loop']+1,layer_zero_based=current['layer'],query_length=q.shape[-2],key_length=k.shape[-2],
            scope='Same Q/K/V and weights. Local FP32 math attention and projection. Not a full-model FP32 reference.',
            reference_RMS=float(gold.square().mean().sqrt()),
            attention_16heads_vs_fp32=errors(actual,gold),attention_4heads_vs_fp32=errors(chunks,gold),
            attention_geometry_only=errors(chunks,actual),
            projection_monolithic_vs_same_input_fp32=errors(bf16,same_input_fp32),
            projection_fp32_partials_vs_same_input_fp32=errors(partials,same_input_fp32),
            combined_native_attention_projection_vs_fp32=errors(bf16,reference))
        records.append(record)
        return result
    remote.ALL_ATTENTION_FUNCTIONS.register('eight_seam_diagnostic',diagnostic)
    model.config._attn_implementation='eight_seam_diagnostic'
    fresh=torch.load(ROOT/'.tensor_cache/HF_P512_v0.pt',map_location='cpu',weights_only=False)
    historical=torch.load(PROJECT/'poc_flashvep/reports/ouro_tp4_parallelism_discovery/.tensor_cache/native_P512_O32.pt',map_location='cpu',weights_only=False)
    for source_label,source in [('new_synthetic_P512_v0',fresh),('historical_failed_P512',historical['request-0'])]:
        past=None
        for step in range(28):
            ids=source['input_ids'] if step==0 else [source['ids'][step-1]]
            value=model(torch.tensor([ids],device='cuda'),past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
            past=value.past_key_values
    for h in handles:h.remove()
    write(ROOT/'numerical/fp32_seams.json',records)
    assert len(records)==90,len(records)
    print('FP32_SEAMS',len(records),flush=True)


if __name__=='__main__':main()
