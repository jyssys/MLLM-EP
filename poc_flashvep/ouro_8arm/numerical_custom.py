"""Controlled sharding and unsharded-cache diagnostics on new calibration prompts."""
import argparse
import types
import os
import torch
import torch.distributed as dist
from poc_flashvep.ouro_tp4.tp_model import TPModel,KVView
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu
from .numerical_trace import Trace,errors
from .contract import *

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tp',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank)
    if a.tp:
        assert os.environ['WORLD_SIZE']=='4';dist.init_process_group('nccl',device_id=torch.device('cuda',rank));model=TPModel(rank)
        container=types.SimpleNamespace(model=types.SimpleNamespace(layers=model.layers))
    else:model=load_cpu().to('cuda');container=model
    summary=[];fp32=[]
    for row in read(ROOT/'numerical/hf_manifest.json'):
        ref=torch.load(row['path'],weights_only=False,map_location='cpu');P,O=row['P'],row['O']
        trace=Trace(container,rank=rank);logits=[];actual=[]
        if a.tp:model.configure([dict(id='r',input_ids=ref['input_ids'])],O)
        else:cache=torch.empty((96,2,1,16,P+O-1,128),device='cuda',dtype=torch.bfloat16)
        for step in range(O):
            trace.step=step;inp=torch.tensor([ref['input_ids']] if step==0 else [[ref['ids'][step-1]]],device=f'cuda:{rank}')
            if a.tp:z,token,h=model.forward(inp,'r',step,P,graphs=False);z=z[0,-1]
            else:
                L=P+step;kv=torch.empty((96,2,1,16,L,128),device='cuda',dtype=torch.bfloat16)
                if step:kv.copy_(cache[:,:,:,:,:L])
                hidden=model.model.embed_tokens(inp);pos=torch.arange(P,device='cuda') if step==0 else torch.tensor([L-1],device='cuda')
                rotary=model.model.rotary_emb(hidden,pos.unsqueeze(0))
                for loop in range(4):
                    for layer in model.model.layers:
                        hidden=layer(hidden,attention_mask=None,position_ids=pos.unsqueeze(0),past_key_value=KVView(kv,step==0),
                            cache_position=pos,position_embeddings=rotary,current_ut=loop,use_cache=True)
                    hidden=model.model.norm(hidden);model.model.early_exit_gate(hidden)
                cache[:,:,:,:,:L].copy_(kv);z=model.lm_head(hidden[:,-1:,:])[0,-1]
            assert torch.isfinite(z).all();actual.append(int(z.argmax()))
            if rank==0:logits.append(z.float().cpu())
        trace.close()
        if rank==0:
            name='Custom_TP4' if a.tp else 'Unsharded'
            path=ROOT/'.tensor_cache'/f'{name}_{row["label"]}.pt'
            torch.save(dict(frames=trace.frames,logits=logits),path)
            summary.append(dict(label=row['label'],ids=actual,reference_ids=ref['ids'],path=str(path),sha256=sha(path)))
            print('CUSTOM_NUMERICAL',name,row['label'],flush=True)
    if rank==0:write(ROOT/'numerical'/('custom_TP4_manifest.json' if a.tp else 'unsharded_manifest.json'),summary)
    if a.tp:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
