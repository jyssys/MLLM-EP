"""Controlled B1 reference sampling. Different backend RNG streams need not match."""
import os
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5'
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    assert transformers.__version__=='4.57.1'
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    records=[]
    for row in read(ROOT/'results/fidelity_completion_inputs.json'):
        for repeat in range(2):
            torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
            inputs=torch.tensor([row['prompt_ids']],device='cuda')
            with torch.inference_mode():
                values=model.generate(inputs,max_new_tokens=512,do_sample=True,temperature=1.,top_p=.7,
                    top_k=0,eos_token_id=2,pad_token_id=2,exit_at_step=3)
            ids=values[0,len(row['prompt_ids']):].tolist()
            record=dict(id=row['id'],repeat=repeat,seed=SEEDS[0],temperature=1.,top_p=.7,
                top_k=0,top_k_semantics='disabled, equivalent filter setting to native -1',
                ids=ids,text=tokenizer.decode(ids,skip_special_tokens=False),
                finish_reason='stop' if ids and ids[-1]==2 else 'length')
            records.append(record)
            write(output/'controlled_sampling.json',records)
    checks=[dict(id=row['id'],same_seed_repeat_equal=next(r['ids'] for r in records if r['id']==row['id'] and r['repeat']==0)==next(r['ids'] for r in records if r['id']==row['id'] and r['repeat']==1)) for row in read(ROOT/'results/fidelity_completion_inputs.json')]
    write(output/'status.json',dict(success=True,checks=checks,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__),
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),scope='Reference sampling sanity, not public benchmark accuracy.'))

if __name__=='__main__':
    main()
