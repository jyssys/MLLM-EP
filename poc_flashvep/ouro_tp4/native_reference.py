"""Save exact native R4 numerical references before distributed correctness tests."""
import gc
import torch
from poc_flashvep.ouro_request_pp.model_adapter import load_cpu
from .contract import *


@torch.inference_mode()
def main():
    mask();torch.set_num_threads(1)
    target=ROOT/'.tensor_cache';target.mkdir(exist_ok=True)
    model=load_cpu().to('cuda:0');model.model.total_ut_steps=4
    # The preserved workload IDs are independent of the new TP outputs.
    pool=json.loads((OLD/'workload.json').read_text())['groups']
    summary=[]
    for P,O,C in [(128,8,2),(512,32,1),(2048,32,1)]:
        output={}
        for row in pool[str(P)][:C]:
            captured=[]
            hook=model.model.norm.register_forward_hook(lambda m,a,y:captured.append(y[:,-1:,:].detach().clone()))
            ids=[];logits=[];hidden=[];margins=[];past=None
            inp=torch.tensor([row['input_ids']],device='cuda:0')
            for step in range(O):
                captured.clear()
                y=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                past=y.past_key_values
                z=y.logits[0,-1].float();token=int(z.argmax());ids.append(token)
                logits.append(z.cpu());hidden.append(captured[-1].float().cpu())
                top=z.topk(2).values;margins.append(float(top[0]-top[1]))
                inp=torch.tensor([[token]],device='cuda:0')
            hook.remove()
            cache={i:(k.cpu(),past.value_cache[i].cpu()) for i,k in enumerate(past.key_cache)}
            assert len(cache)==96
            output[row['id']]=dict(input_ids=row['input_ids'],ids=ids,logits=logits,hidden=hidden,margins=margins,cache=cache)
            summary.append(dict(P=P,O=O,id=row['id'],ids=ids,margins=margins,KV_slots=len(cache),KV_length=P+O-1))
        file=target/f'native_P{P}_O{O}.pt';assert not file.exists();torch.save(output,file)
        write(target/f'native_P{P}_O{O}.json',dict(path=str(file),sha256=sha(file),bytes=file.stat().st_size))
        print('NATIVE_REFERENCE',P,O,C,flush=True)
        del y,past,cache,output;gc.collect();torch.cuda.empty_cache()
    # Operator reference at one shared physical layer for every loop and two KV states.
    records=[];current={};loop_index=[0]
    def block_pre(module,args,kwargs):
        loop_index[0]=kwargs['current_ut'];current[(step,loop_index[0])]=dict(step=step,loop=loop_index[0],
            hidden_in=args[0].cpu(),rotary=tuple(x.cpu() for x in kwargs['position_embeddings']))
    def attention(module,args,kwargs,result):
        current[(step,loop_index[0])].update(attention_in=kwargs['hidden_states'].cpu(),attention_out=result[0].cpu())
    def mlp(module,args,result):
        current[(step,loop_index[0])].update(mlp_in=args[0].cpu(),mlp_out=result.cpu())
    def block_post(module,args,result):current[(step,loop_index[0])]['hidden_out']=result.cpu()
    layer=model.model.layers[0]
    hooks=[layer.register_forward_pre_hook(block_pre,with_kwargs=True),
           layer.self_attn.register_forward_hook(attention,with_kwargs=True),layer.mlp.register_forward_hook(mlp),
           layer.register_forward_hook(block_post)]
    inp=torch.tensor([pool['16'][0]['input_ids']],device='cuda:0');past=None
    for step in range(2):
        y=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1);past=y.past_key_values
        for loop in range(4):
            r=current[(step,loop)];slot=24*loop
            r['KV']=(past.key_cache[slot].cpu(),past.value_cache[slot].cpu());records.append(r)
        inp=y.logits[:,-1].argmax(-1).view(1,1)
    for h in hooks:h.remove()
    torch.save(records,target/'operators.pt')
    write(ROOT/'native_reference.json',dict(model_revision=REVISION,records=summary,
        operators=dict(path='.tensor_cache/operators.pt',sha256=sha(target/'operators.pt'),records=len(records)),
        storage='Full KV tensors remain local in ignored .tensor_cache. Published errors and hashes identify the reference.'))
    print('OPERATORS_REFERENCE',len(records),flush=True)


if __name__=='__main__':main()
