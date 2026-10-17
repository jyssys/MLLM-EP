"""Minimize the batch-parity failure to one cached forward at the EOS boundary."""
import copy
import os
import torch
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.scripts.baseline import load,token_ids


@torch.inference_mode()
def main():
    model,tokenizer=load()
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    data=read(ROOT/'outputs/manifests/data.json')
    original=read(ROOT/'outputs/batch_reference_v1/parity.json')['rows']
    rows=data['smoke'][:4]
    prompts=[token_ids(tokenizer,row,data['prompt_suffix']) for row in rows]
    width=max(map(len,prompts))
    inputs=torch.tensor([[2]*(width-len(row))+row for row in prompts],device='cuda')
    mask=torch.tensor([[0]*(width-len(row))+[1]*len(row) for row in prompts],device='cuda')
    positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,1)
    value=model(inputs,attention_mask=mask,position_ids=positions,use_cache=True,exit_at_step=3,logits_to_keep=1)
    first_EOS=min(len(row['official_ids']) for row in original)
    for step in range(first_EOS-1):
        inputs=torch.tensor([[row['official_ids'][step]] for row in original],device='cuda')
        mask=torch.cat([mask,torch.ones((4,1),device='cuda',dtype=mask.dtype)],dim=1)
        positions=mask.sum(-1,keepdim=True)-1
        value=model(inputs,attention_mask=mask,position_ids=positions,past_key_values=value.past_key_values,
                    use_cache=True,exit_at_step=3,logits_to_keep=1)
    keep=[i for i,row in enumerate(original) if len(row['official_ids'])>first_EOS]
    selection=torch.tensor(keep,device='cuda')
    full=copy.deepcopy(value.past_key_values)
    small=copy.deepcopy(value.past_key_values)
    small.reorder_cache(selection)
    kv_before=all(torch.equal(a.index_select(0,selection),b) for a,b in zip(full.key_cache,small.key_cache))
    kv_before &= all(torch.equal(a.index_select(0,selection),b) for a,b in zip(full.value_cache,small.value_cache))
    inputs=torch.tensor([[row['official_ids'][first_EOS-1] if len(row['official_ids'])>=first_EOS else 2]
                         for row in original],device='cuda')
    mask=torch.cat([mask,torch.ones((4,1),device='cuda',dtype=mask.dtype)],dim=1)
    positions=mask.sum(-1,keepdim=True)-1
    captures=[]
    def observer(name):
        def capture(module,args,value):
            captures.append((name,value.detach().clone()))
        return capture
    modules=dict(input_norm=model.model.layers[0].input_layernorm,
                 q_projection=model.model.layers[0].self_attn.q_proj,
                 k_projection=model.model.layers[0].self_attn.k_proj,
                 v_projection=model.model.layers[0].self_attn.v_proj,
                 attention_output_projection=model.model.layers[0].self_attn.o_proj,
                 layer_output=model.model.layers[0])
    handles=[module.register_forward_hook(observer(name)) for name,module in modules.items()]
    a=model(inputs,attention_mask=mask,position_ids=positions,past_key_values=full,
            use_cache=True,exit_at_step=3,logits_to_keep=1)
    captured_full=captures[:];captures=[]
    b=model(inputs.index_select(0,selection),attention_mask=mask.index_select(0,selection),
            position_ids=positions.index_select(0,selection),past_key_values=small,
            use_cache=True,exit_at_step=3,logits_to_keep=1)
    for handle in handles:
        handle.remove()
    details=[]
    for (name,left),(other,right) in zip(captured_full,captures):
        assert name==other
        left=left.index_select(0,selection).float();right=right.float()
        details.append(dict(component=name,bitwise=torch.equal(left,right),
                            max_abs=float((left-right).abs().max()),
                            relative_L2=float((left-right).norm()/left.norm())))
    left=a.logits.index_select(0,selection).float();right=b.logits.float()
    write(output/'probe.json',dict(first_EOS_length=first_EOS,remaining_rows=keep,
        all_96_selected_KV_bitwise_before_forward=kv_before,
        matching_input_ids_mask_position_ids=True,weights_and_backend_unchanged=True,
        only_execution_batch_dimension_changed=[4,len(keep)],
        logits_bitwise=torch.equal(left,right),logits_max_abs=float((left-right).abs().max()),
        logits_relative_L2=float((left-right).norm()/left.norm()),
        top1_equal=torch.equal(left.argmax(-1),right.argmax(-1)),first_layer_components=details))
    assert kv_before
    print('EOS-boundary probe',first_EOS,details[:6],flush=True)


if __name__=='__main__':
    main()
