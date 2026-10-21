"""Synthetic FP32 math forward. This diagnostic is not a serving or accuracy arm."""
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu
from .contract import *
from .numerical_trace import Trace


@torch.inference_mode()
def main():
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==UUIDS[1]
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model=load_cpu().float().cuda();manifest=[]
    with sdpa_kernel(SDPBackend.MATH):
        for row in read(ROOT/'numerical/hf_manifest.json'):
            source=torch.load(ROOT/'.tensor_cache'/f"HF_{row['label']}.pt",map_location='cpu',weights_only=False)
            trace=Trace(model);past=None;logits=[]
            for step in range(row['O']):
                trace.step=step;ids=source['input_ids'] if step==0 else [source['ids'][step-1]]
                result=model(torch.tensor([ids],device='cuda'),past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                past=result.past_key_values;z=result.logits[0,-1];assert torch.isfinite(z).all();logits.append(z.cpu())
            trace.close();file=ROOT/'.tensor_cache'/f"FP32_math_{row['label']}.pt"
            torch.save(dict(logits=logits,frames=trace.frames),file)
            manifest.append(dict(label=row['label'],P=row['P'],O=row['O'],physical_GPU=5,path=str(file),sha256=sha(file),
                scope='Synthetic teacher-aligned FP32 math reference. BF16 checkpoint values expanded to FP32. No TF32. Not task accuracy.'))
            print('FP32_MODEL',row['label'],row['O'],flush=True)
    write(ROOT/'numerical/fp32_model_manifest.json',manifest)


if __name__=='__main__':main()
