"""Check full EOS-aware batched output parity with the official driver."""
import os
import torch
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.scripts.baseline import load,token_ids
from poc_async_recurrent_pp.emulator import generate_batch,StageEmulator


@torch.inference_mode()
def main():
    model,tokenizer=load()
    data=read(ROOT/'outputs/manifests/data.json')
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    prompts=[token_ids(tokenizer,row,data['prompt_suffix']) for row in data['smoke'][:4]]
    width=max(map(len,prompts))
    inputs=torch.tensor([[2]*(width-len(row))+row for row in prompts],device='cuda')
    mask=torch.tensor([[0]*(width-len(row))+[1]*len(row) for row in prompts],device='cuda')
    official=model.generate(inputs,attention_mask=mask,max_new_tokens=1024,do_sample=False,
        eos_token_id=2,pad_token_id=2,exit_at_step=3)
    reference=[]
    for row in official:
        ids=row[width:].tolist()
        if 2 in ids:
            ids=ids[:ids.index(2)+1]
        reference.append(ids)
    hooks=StageEmulator(model,'exact')
    try:
        candidate=generate_batch(model,tokenizer,prompts,1024,hooks)
    finally:
        hooks.close()
    records=[]
    for i,(native,custom) in enumerate(zip(reference,candidate)):
        matches=native==custom['ids']
        first=next((j for j,(a,b) in enumerate(zip(native,custom['ids'])) if a!=b),None)
        records.append(dict(smoke_id=data['smoke'][i]['id'],exact_token_parity=matches,
            official_ids=native,custom_ids=custom['ids'],first_divergent_token=first,
            official_EOS=native[-1]==2,custom_EOS=custom['ids'][-1]==2))
    passed=all(row['exact_token_parity'] for row in records)
    write(output/'parity.json',dict(rows=records,pass_gate=passed,
        comparison='Unmodified official generation vs instrumented exact stage hooks with the native fixed-batch generation driver.'))
    write(output/'status.json',dict(success=passed,n=4))
    assert passed,'Batch executor reference parity requires investigation'


if __name__=='__main__':
    main()
