"""Bounded FP32 math check of masked-capacity versus exact-length KV semantics."""
import argparse
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu,StaticCache
from .contract import *
from .numerical_trace import errors


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',type=int,default=5);p.add_argument('--cap',type=int,default=8);a=p.parse_args()
    assert a.physical in [4,5,6,7]
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[a.physical-4]
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model=load_cpu().float().cuda();source=read(ROOT/'raw/probe_S4_v3/probe-0.json');P=len(source['prompt_ids'])
    gold=torch.load(ROOT/'.tensor_cache/HF_PP_probes.pt',map_location='cpu',weights_only=False)['probe-0']
    cache=StaticCache(P+a.cap-1);past=None;records=[]
    with sdpa_kernel(SDPBackend.MATH):
        for step in range(8):
            ids=torch.tensor([source['prompt_ids'] if step==0 else [gold['ids'][step-1]]],device='cuda')
            reference=model(ids,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1);past=reference.past_key_values
            h=model.model.embed_tokens(ids);position=0 if step==0 else P+step-1
            pos=torch.arange(P,device='cuda') if step==0 else torch.tensor([position],device='cuda')
            cache.prefill=step==0;cache.length=P+step
            mask=None if step==0 else torch.where(torch.arange(cache.capacity,device='cuda')<=position,0.,float('-inf')).view(1,1,1,-1)
            rotary=model.model.rotary_emb(h,pos.unsqueeze(0))
            for loop in range(4):
                for layer in model.model.layers:
                    h=layer(h,attention_mask=mask,position_ids=pos.unsqueeze(0),past_key_value=cache,
                        cache_position=pos,position_embeddings=rotary,current_ut=loop,use_cache=True)
                h=model.model.norm(h)
            z=model.lm_head(h[:,-1:,:])
            key=max(errors(k[:,:,:cache.length],past.key_cache[i])['relative_L2'] for i,k in enumerate(cache.key_cache))
            value=max(errors(v[:,:,:cache.length],past.value_cache[i])['relative_L2'] for i,v in enumerate(cache.value_cache))
            records.append(dict(step=step,logits=errors(z,reference.logits),K_max_slot_L2=key,V_max_slot_L2=value,
                argmax_equal=int(z[0,-1].argmax())==int(reference.logits[0,-1].argmax())))
    write(ROOT/'numerical'/f'fp32_mask_diagnostic_cap{a.cap}.json',dict(physical_GPU=a.physical,capacity=cache.capacity,
        precision='FP32 diagnostic expansion of the same BF16 checkpoint values',
        scope='P128, eight teacher-aligned positions, math SDPA. Not BF16 serving, task accuracy, or a full long-context numerical qualification.',rows=records))
    print('FP32_MASK_DIAGNOSTIC',records,flush=True)


if __name__=='__main__':main()
