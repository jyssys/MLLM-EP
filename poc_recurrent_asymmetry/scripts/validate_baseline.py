"""Revalidate exact reference and observer parity. No predictor execution."""
import copy
import gc
import importlib
import importlib.metadata
import os
import subprocess
import torch
from poc_recurrent_asymmetry.contract import *
from poc_async_recurrent_pp.scripts.baseline import load,token_ids
from poc_async_recurrent_pp.emulator import cache_signature
from poc_async_recurrent_pp.scoring import score


@torch.inference_mode()
def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    model,tokenizer=load()
    code=importlib.import_module(type(model).__module__)
    assert sha(code.__file__)==sha(MODEL/'modeling_ouro.py')
    data=read(PROJECT/'poc_async_recurrent_pp/outputs/manifests/data.json')
    env=dict(model_revision=REVISION,model_source_sha256=sha(code.__file__),
        config=model.config.to_dict(),torch=torch.__version__,cuda=torch.version.cuda,
        transformers=importlib.metadata.version('transformers'),nccl=torch.cuda.nccl.version(),
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),logical_cuda=0,
        UUID=str(torch.cuda.get_device_properties(0).uuid),
        unique_weight_bytes=sum({p.untyped_storage().data_ptr():p.untyped_storage().nbytes()
            for p in model.parameters()}.values()),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        versions=subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True))
    write(output/'environment.json',env)
    integrity=[]
    for batch in [1,2]:
        prompts=[token_ids(tokenizer,r,data['prompt_suffix']) for r in data['smoke'][:batch]]
        width=max(map(len,prompts))
        ids=torch.tensor([[2]*(width-len(p))+p for p in prompts],device='cuda')
        mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device='cuda')
        reference=model(ids,attention_mask=mask,use_cache=True,exit_at_step=3,logits_to_keep=1)
        observed=model(ids,attention_mask=mask,use_cache=True,exit_at_step=3,logits_to_keep=1)
        counts=[]
        handles=[]
        for l,layer in enumerate(model.model.layers):
            def before(module,args,kwargs,l=l):counts.append((kwargs['current_ut'],l))
            handles.append(layer.register_forward_pre_hook(before,with_kwargs=True))
        try:
            for step in range(4):
                counts.clear()
                token=reference.logits[:,-1].argmax(-1,keepdim=True)
                mask=torch.cat([mask,torch.ones((batch,1),device='cuda',dtype=mask.dtype)],dim=1)
                a=model(token,attention_mask=mask,past_key_values=reference.past_key_values,
                    use_cache=True,exit_at_step=3,logits_to_keep=1)
                observed_counts=list(counts);counts.clear()
                b=model(token,attention_mask=mask,past_key_values=observed.past_key_values,
                    use_cache=True,exit_at_step=3,logits_to_keep=1)
                assert observed_counts==[(r,l) for r in range(4) for l in range(24)]
                assert torch.equal(a.logits,b.logits)
                assert all(torch.equal(x,y) for x,y in zip(a.past_key_values.key_cache,b.past_key_values.key_cache))
                assert all(torch.equal(x,y) for x,y in zip(a.past_key_values.value_cache,b.past_key_values.value_cache))
                integrity.append(dict(batch=batch,cached_step=step,logical_executions=len(observed_counts),
                    logits_max_abs=0.,logits_mean_abs=0.,all_KV_bitwise=True,
                    cache=cache_signature(b.past_key_values)))
                reference,observed=a,b
        finally:
            for h in handles:h.remove()
        del reference,observed,a,b
        gc.collect();torch.cuda.empty_cache()
    old=read(PROJECT/'poc_async_recurrent_pp/outputs/baseline_v1/reference_parity.json')['natural']
    natural=[]
    for row in old[:2]:
        ids=torch.tensor([row['prompt_ids']],device='cuda')
        generated=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=2048,
            eos_token_id=2,pad_token_id=2,exit_at_step=3)[0,len(row['prompt_ids']):].tolist()
        assert generated==row['official_ids']
        natural.append(dict(id=row['id'],matched_tokens=len(generated),total_tokens=len(row['official_ids']),
            EOS=generated[-1]==2,output_ids=generated))
    selected=[row for dataset in ['gsm8k','math500'] for row in [r for r in data['pilot'] if r['dataset']==dataset][:2]]
    quality=[]
    for row in selected:
        prompt=token_ids(tokenizer,row,data['prompt_suffix'])
        inp=torch.tensor([prompt],device='cuda')
        tokens=model.generate(inp,attention_mask=torch.ones_like(inp),do_sample=False,max_new_tokens=8192,
            eos_token_id=2,pad_token_id=2,exit_at_step=3)[0,len(prompt):].tolist()
        result=dict(text=tokenizer.decode(tokens,skip_special_tokens=False),ids=tokens,
            finish_reason='eos' if tokens[-1]==2 else 'length')
        record=dict(id=row['id'],dataset=row['dataset'],prompt_ids=prompt,**result,**score(row,result))
        quality.append(record)
        write(output/(row['id'].replace('/','_')+'.json'),record)
        print('BASELINE_QUALITY',row['id'],record['answer_matches'],len(tokens),flush=True)
    write(output/'integrity.json',dict(rows=integrity,natural=natural,
        prior_reference='poc_async_recurrent_pp/outputs/baseline_v1/reference_parity.json',
        approximate_predictors_executed=False))
    write(output/'status.json',dict(success=True,exact_reference=True,quality=[dict(dataset=d,
        correct=sum(r['answer_matches'] for r in quality if r['dataset']==d),n=sum(r['dataset']==d for r in quality))
        for d in ['gsm8k','math500']],scope='Small exact-reference regression, not full benchmark accuracy.'))


if __name__=='__main__':
    main()
