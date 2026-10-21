"""Fresh non-task numerical calibration prompts with layer and loop observations."""
import os
import torch
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu,tokenizer
from .numerical_trace import Trace
from .contract import *

@torch.inference_mode()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4';torch.set_num_threads(1)
    model=load_cpu().to('cuda');tok=tokenizer();folder=ROOT/'.tensor_cache';folder.mkdir(exist_ok=True)
    manifest=[]
    for P,O in [(128,8),(256,8),(512,32),(2048,8)]:
        for variant,text in enumerate(['Numerical calibration: add the integers in this sequence. 1 2 3 4. ',
                                        'Numerical calibration: explain a small Python function that reverses a list. ']):
            base=tok.encode(text,add_special_tokens=False);ids=(base*((P+len(base)-1)//len(base)))[:P]
            label=f'P{P}_v{variant}';target=folder/f'HF_{label}.pt';assert not target.exists()
            trace=Trace(model);past=None;inp=torch.tensor([ids],device='cuda');out=[];logits=[]
            for step in range(O):
                trace.step=step
                value=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                past=value.past_key_values;z=value.logits[0,-1];assert torch.isfinite(z).all()
                logits.append(z.float().cpu());token=int(z.argmax());out.append(token);inp=torch.tensor([[token]],device='cuda')
            trace.close();torch.save(dict(input_ids=ids,ids=out,logits=logits,frames=trace.frames,P=P,O=O,
                scope='Last token of each component only. Synthetic calibration, not benchmark accuracy.'),target)
            manifest.append(dict(label=label,P=P,O=O,path=str(target),sha256=sha(target)))
            print('HF_NUMERICAL',label,len(trace.frames),flush=True)
    write(ROOT/'numerical/hf_manifest.json',manifest)

if __name__=='__main__':main()
