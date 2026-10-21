"""Teacher-aligned probe separates masked attention, graphs, and request sharing."""
import torch
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu
from .stage import Stage
from .numerical_trace import errors
from .contract import *


@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.cuda.set_device(0)
    gold=torch.load(ROOT/'.tensor_cache/HF_PP_probes.pt',map_location='cpu',weights_only=False)['probe-0']
    row=read(ROOT/'raw/probe_S4_v2/probe-0.json');ids=row['prompt_ids'];P=len(ids)
    model=load_cpu().cuda();parts=[Stage(0,'loop_pp4'),Stage(0,'loop_pp4')]
    parts[0].capture_enabled=False
    for part in parts:part.prepare([dict(id='r',input_ids=ids)],8,[(r,0,24) for r in range(4)])
    past=None;records=[]
    for step in range(8):
        inp=torch.tensor([ids if step==0 else [gold['ids'][step-1]]],device='cuda')
        ref=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1);past=ref.past_key_values
        for name,part in zip(['masked_eager','masked_graph'],parts):
            h=part.embedding(inp)
            for loop in range(4):h,_=part.execute(h,'r',step,0 if step==0 else P+step-1,loop,0,24)
            z=part.head(h[:,-1:,:]);cache=part.snapshot('r')
            if name=='masked_eager':eager_z=z.clone()
            e=max(errors(y,x.cpu())['relative_L2'] for index,kv in cache.items() for y,x in zip(kv,(past.key_cache[index],past.value_cache[index])))
            records.append(dict(step=step,mode=name,logits=errors(z,ref.logits),KV_max_slot_L2=e,
                graph_vs_eager=errors(z,eager_z),
                argmax=int(z[0,-1].argmax()),reference_argmax=int(ref.logits[0,-1].argmax())))
    write(ROOT/'numerical/graph_diagnostic_fixed.json',records)
    assert all(r['graph_vs_eager']['max_abs']==0 for r in records),records
    print('GRAPH_VS_EAGER_PASS',records,flush=True)


if __name__=='__main__':main()
