"""Compare the official base checkpoint to the staged fixed-R4 execution."""
import os
import argparse
import numpy as np
import torch
from poc_decode_first_depth_aware_dp.contract import *


@torch.inference_mode()
def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--aligned-environment',action='store_true')
    parser.add_argument('--backend',default='sdpa',choices=['sdpa','flash_attention_3']);args=parser.parse_args()
    if args.aligned_environment:
        from poc_decode_first_depth_aware_dp.runtime import configure_imports
        configure_imports()
    from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
    source=args.source
    fixed=read(source/'q_None.json')
    config=AutoConfig.from_pretrained(MODEL,trust_remote_code=True)
    if not hasattr(config,'pad_token_id'):config.pad_token_id=None
    model=AutoModelForCausalLM.from_pretrained(MODEL,config=config,trust_remote_code=True,torch_dtype=torch.bfloat16,
        attn_implementation=args.backend).to('cuda:0').eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    prompt=tokenizer.encode('Q: What is 2 + 3?\nA:')
    logits=[];value=None
    for step in range(len(fixed['ids'])):
        ids=prompt if step==0 else [fixed['ids'][step-1]]
        value=model(torch.tensor([ids],device='cuda'),past_key_values=None if value is None else value.past_key_values,
            use_cache=True,exit_at_step=3,logits_to_keep=1)
        logits.append(value.logits[0,-1].cpu().float().numpy())
    gold=np.stack(logits);observed=np.load(source/'logits_q_None.npz')['logits']
    assert gold.shape==observed.shape
    diff=np.abs(gold-observed)
    cache=np.load(source/'fixed_KV.npz');num=0.;den=0.
    for name,tensors in [('keys',value.past_key_values.key_cache),('values',value.past_key_values.value_cache)]:
        a=torch.stack(tensors).squeeze(1).transpose(1,2).cpu().float().numpy()
        b=cache[name]
        assert a.shape==b.shape,(a.shape,b.shape)
        num+=float(np.square(a.astype(np.float64)-b).sum());den+=float(np.square(a.astype(np.float64)).sum())
    limits=read(ROOT/'artifacts/preregistration.json')['numerical_limits']
    relative=(num/den)**.5
    ids=torch.tensor([prompt],device='cuda')
    natural=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=16,
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,exit_at_step=3)[0,len(prompt):].tolist()
    metrics=dict(logits_max_abs=float(diff.max()),logits_mean_abs=float(diff.mean()),KV_relative_L2=relative,
        top1_matches=int((gold.argmax(-1)==observed.argmax(-1)).sum()),positions=len(gold),
        natural_ids=natural,natural_ids_equal=natural==fixed['ids'],KV_slots=len(value.past_key_values.key_cache),
        limits=limits,aligned_environment=args.aligned_environment,backend=args.backend,
        scope='One small pretrained probe, full logits and all 96-slot KV. Not benchmark quality.')
    qualified=metrics['logits_max_abs']<=limits['logits_max_abs'] and metrics['logits_mean_abs']<=limits['logits_mean_abs'] and relative<=limits['KV_relative_L2'] and metrics['natural_ids_equal']
    write(output/'reference_comparison.json',metrics)
    write(output/'status.json',dict(success=True,numerical_qualified=qualified))
    print('OFFICIAL_BASE_PARITY',qualified,metrics['logits_max_abs'],metrics['logits_mean_abs'],relative,flush=True)


if __name__=='__main__':main()
