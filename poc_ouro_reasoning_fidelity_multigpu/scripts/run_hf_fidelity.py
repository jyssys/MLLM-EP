"""Official isolated Transformers reference. No serving speed claims."""
import sys
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--reference-version',choices=['4.54.1','4.57.1'],default='4.54.1')
parser.add_argument('--additional-only',action='store_true')
options=parser.parse_args()
if options.reference_version=='4.54.1':
    sys.path.insert(0,'/home/esjung/external/ouro-reasoning-fidelity-assets/hf454_deps')
import gc
import importlib
import os
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==('5' if options.additional_only else '4')
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    assert transformers.__version__==options.reference_version
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    remote=importlib.import_module(type(model).__module__)
    assert sha(remote.__file__)==sha(MODEL/'modeling_ouro.py')
    write(output/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,
        transformers_path=transformers.__file__,remote_source=remote.__file__,remote_sha256=sha(remote.__file__),
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
        parameter_bytes=sum({p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}.values()),
        config=model.config.to_dict(),eos=tokenizer.eos_token_id,attention='SDPA'))
    inputs=read(ROOT/('results/fidelity_additional_inputs.json' if options.additional_only else 'results/fidelity_inputs.json'))
    hidden=[]
    def capture(module,args,value):
        hidden[:] = [value[0].last_hidden_state[:,-1,:].detach()]
    handle=model.model.register_forward_hook(capture)
    with torch.inference_mode():
        for row in inputs:
            inp=torch.tensor([row['prompt_ids']],device='cuda')
            past=None;records=[]
            P=len(row['prompt_ids'])
            for step in range(1025):
                result=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                past=result.past_key_values
                if step in [0,1,4,32,128,1024]:
                    positions=sorted(set([0,P-1,P+step-1]));positions=[p for p in positions if p>=0]
                    coordinates=[0,1,2,3,124,125,126,127]
                    slots=[]
                    for key,val in zip(past.key_cache,past.value_cache):
                        k=key[0,:,positions,:].permute(1,0,2)[...,coordinates]
                        v=val[0,:,positions,:].permute(1,0,2)[...,coordinates]
                        slots.append(torch.cat([k,v],dim=-1).float().cpu().numpy())
                    assert len(slots)==192 and all(past.get_seq_length(i)==P+step for i in range(192))
                    np.savez_compressed(output/f'teacher_{file_id(row["id"])}.step{step}.npz',
                        logits=result.logits[:,-1,:].float().cpu().numpy(),
                        hidden=hidden[0].float().cpu().numpy(),KV=np.stack(slots))
                    records.append(dict(step=step,positions=positions,cache_lengths=[past.get_seq_length(i) for i in range(192)]))
                inp=torch.tensor([[row['teacher_ids'][step]]],device='cuda')
            write(output/f'teacher_{file_id(row["id"])}.json',records)
            del past,result
            gc.collect()
            print('HF teacher',row['id'],flush=True)
        handle.remove()
        if options.additional_only:
            write(output/'status.json',dict(success=True,scope='Supplemental MATH teacher states. No serving timing.'))
            return
        completion_inputs=read(ROOT/'results/fidelity_completion_inputs.json')
        for cap,rows in [(256,inputs),(2048,inputs[:2]),(8192,completion_inputs)]:
            for row in rows:
                inp=torch.tensor([row['prompt_ids']],device='cuda');past=None;ids=[]
                start=time.perf_counter()
                for step in range(cap):
                    result=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                    past=result.past_key_values
                    token=int(result.logits[:,-1,:].argmax(-1));ids.append(token)
                    if token==2:break
                    inp=torch.tensor([[token]],device='cuda')
                write(output/f'natural_{file_id(row["id"])}_cap{cap}.json',dict(id=row['id'],cap=cap,ids=ids,
                    text=tokenizer.decode(ids,skip_special_tokens=False),finish_reason='stop' if ids[-1]==2 else 'length',
                    wall_ms=(time.perf_counter()-start)*1000,scope='HF reference, not optimized serving timing'))
                del past,result
                gc.collect()
                print('HF natural',row['id'],cap,flush=True)
    write(output/'status.json',dict(success=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    main()
