"""Fixed-R4 official Transformers reference, not a serving timing baseline."""
import os
import torch
from poc_looped_multigpu_kv_placement.contract import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['4','5']
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(model_path('ouro'),trust_remote_code=True,
        torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    ids=torch.tensor([[1,102,211,17,38,61]],device='cuda');cache=None;tokens=[];probs=[]
    storage={};slots=[];progress=[]
    with torch.inference_mode():
        for step in range(4):
            value=model(ids,past_key_values=cache,use_cache=True)
            cache=value.past_key_values;logits=value.logits[:,-1,:].float();next_id=logits.argmax(-1)
            tokens.append(int(next_id));top=logits.log_softmax(-1).topk(5)
            probs.append(dict(ids=top.indices[0].cpu().tolist(),logprobs=top.values[0].cpu().tolist()))
            for index,key in enumerate(cache.key_cache):
                progress.append(dict(forward=step,query_tokens=int(ids.shape[1]),loop=index//48,
                    physical_layer=index%48,cache_slot=index,cached_positions=int(key.shape[2]),
                    last_token_position=int(key.shape[2])-1,last_token_page16=(int(key.shape[2])-1)//16,
                    storage_ptr=key.untyped_storage().data_ptr(),storage_bytes=key.untyped_storage().nbytes()))
            ids=next_id[:,None]
        for index,(key,val) in enumerate(zip(cache.key_cache,cache.value_cache)):
            entries=[]
            for tensor in [key,val]:
                storage[tensor.untyped_storage().data_ptr()]=tensor.untyped_storage().nbytes()
                entries.append(dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),
                    logical_bytes=tensor.numel()*tensor.element_size(),storage_ptr=tensor.untyped_storage().data_ptr(),
                    storage_bytes=tensor.untyped_storage().nbytes(),offset=tensor.storage_offset()))
            slots.append(dict(slot=index,physical_layer=index%48,loop=index//48,KV=entries))
    write(output/'reference.json',dict(model=MODELS['ouro'],runtime='Official pinned remote HF code, Torch2.8, Transformers4.57.1, SDPA',
        total_ut_steps=model.config.total_ut_steps,physical_layers=len(model.model.layers),cache_type=type(cache).__name__,
        actual_cache_slots=len(slots),unique_cache_storage_bytes=sum(storage.values()),slot_map=slots,
        output_ids=tokens,logprobs=probs,precision='BF16',cache_progression=progress,
        physical_mask=os.environ['CUDA_VISIBLE_DEVICES'],GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),
        status='REAL_PRETRAINED_REFERENCE'))
    write(output/'status.json',dict(success=True))


if __name__=='__main__':main()
