"""Diagnostic only: isolate exact-cache bridge math without TP or collectives."""
import torch
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu
from poc_flashvep.ouro_request_pp.reference import errors
from .contract import *
from .tp_model import KVView


@torch.inference_mode()
def main():
    mask();torch.set_num_threads(1)
    model=load_cpu().to('cuda:0');r=next(iter(torch.load(ROOT/'.tensor_cache/native_P512_O32.pt',weights_only=False).values()))
    cache=torch.empty((96,2,1,16,543,128),device='cuda:0',dtype=torch.bfloat16);records=[]
    for step in range(32):
        length=512+step;inp=torch.tensor([r['input_ids']] if step==0 else [[r['ids'][step-1]]],device='cuda:0')
        hidden=model.model.embed_tokens(inp);kv=torch.empty((96,2,1,16,length,128),device='cuda:0',dtype=torch.bfloat16)
        if step:kv.copy_(cache[:,:,:,:,:length])
        pos=torch.arange(512,device='cuda:0') if step==0 else torch.tensor([length-1],device='cuda:0')
        rotary=model.model.rotary_emb(hidden,pos.unsqueeze(0))
        for loop in range(4):
            for layer in model.model.layers:
                hidden=layer(hidden,attention_mask=None,position_ids=pos.unsqueeze(0),past_key_value=KVView(kv,step==0),
                    use_cache=True,cache_position=pos,position_embeddings=rotary,current_ut=loop)
            hidden=model.model.norm(hidden);model.model.early_exit_gate(hidden)
        cache[:,:,:,:,:length].copy_(kv);z=model.lm_head(hidden[:,-1:,:])
        records.append(dict(step=step,logits=errors(z[0,-1].float().cpu(),r['logits'][step]),
            hidden=errors(hidden[:,-1:,:].float().cpu(),r['hidden'][step])))
    write(ROOT/'correctness/unsharded_cache_probe.json',records)
    print('UNSHARDED',max(x['logits']['max_abs'] for x in records),max(x['hidden']['relative_L2'] for x in records),flush=True)


if __name__=='__main__':main()
