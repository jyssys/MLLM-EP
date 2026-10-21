"""Official unmodified Single-GPU reference for the new synthetic PP probes."""
import torch
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu,tokenizer
from .contract import *


@torch.inference_mode()
def main():
    torch.set_num_threads(1);model=load_cpu().cuda();tok=tokenizer();outputs={}
    for i in range(4):
        text='Numerical calibration: '+('add integers 1 2 3. ' if i%2==0 else 'write a short Python function. ')
        ids=tok.encode(text,add_special_tokens=False);ids=(ids*((128+len(ids)-1)//len(ids)))[:128]
        actual=[];past=None
        for step in range(8):
            inp=torch.tensor([ids if step==0 else [actual[-1]]],device='cuda')
            value=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
            actual.append(int(value.logits[0,-1].argmax()));past=value.past_key_values
        outputs[f'probe-{i}']=dict(ids=actual,logits=value.logits.float().cpu(),
            KV={j:(k.cpu(),past.value_cache[j].cpu()) for j,k in enumerate(past.key_cache)})
    torch.save(outputs,ROOT/'.tensor_cache/HF_PP_probes.pt')
    write(ROOT/'numerical/HF_PP_probes.json',{rid:dict(ids=r['ids'],slots=len(r['KV'])) for rid,r in outputs.items()})
    print('HF_PP_REFERENCE',len(outputs),flush=True)


if __name__=='__main__':main()
