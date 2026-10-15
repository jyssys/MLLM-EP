"""Official HF reference: exact task order, masks, and independent KV slots."""
import os
import importlib
import re
import numpy as np
import torch
from poc_looped_exact_depth_chunk_pipeline.contract import *
from poc_looped_exact_depth_chunk_pipeline.dag import diagonal_order,verify_order


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    assert len(model.model.layers)==48 and model.config.total_ut_steps==4
    assert all(layer.attention_type=='full_attention' for layer in model.model.layers)
    module=importlib.import_module(type(model).__module__);Cache=module.UniversalTransformerCache
    remote_code=dict(path=module.__file__,sha256=sha(module.__file__),matches_pinned_checkpoint_source=sha(module.__file__)==sha(MODEL/'modeling_ouro.py'))
    assert remote_code['matches_pinned_checkpoint_source']
    hardware=read(ROOT/'results/frozen_hardware.json')['requests'];results=[]
    # These are component probes, not benchmark completions or speed baselines.
    with torch.inference_mode():
        for row in hardware[:2]:
            ids=torch.tensor([row['prefix_token_ids']['1024'][:256]],device='cuda')
            P=ids.shape[1];base,states,gates=model.model(ids,use_cache=True)
            hidden=base.last_hidden_state;cache=base.past_key_values
            positions=sorted(set([0,63,64,127,128,P-1]))
            base_logits=model.lm_head(hidden[:,positions]).float()
            for B in [64,128]:
                M=(P+B-1)//B;order=diagonal_order(M);verify_order(order)
                new_cache=Cache(192);outputs={};publication=[]
                for r,j in order:
                    start=j*B;stop=min(P,(j+1)*B)
                    assert all(new_cache.get_seq_length(48*r+l)==start for l in range(48))
                    h=model.model.embed_tokens(ids[:,start:stop]) if r==0 else outputs[(r-1,j)]
                    pos=torch.arange(start,stop,device=ids.device);pids=pos[None,:]
                    rotary=model.model.rotary_emb(h,pids)
                    mask=(torch.arange(stop,device=ids.device)[None,:]<=pos[:,None])[None,None,:,:]
                    for layer in model.model.layers:
                        h=layer(h,attention_mask=mask,position_ids=pids,past_key_value=new_cache,
                            use_cache=True,cache_position=pos,position_embeddings=rotary,current_ut=r)
                    h=model.model.norm(h);outputs[(r,j)]=h
                    assert all(new_cache.get_seq_length(48*r+l)==stop for l in range(48))
                    publication.append(dict(loop=r,chunk=j,prefix_before=start,published_stop=stop))
                final=torch.cat([outputs[(3,j)] for j in range(M)],dim=1)
                logits=model.lm_head(final[:,positions]).float()
                relative=lambda a,b:float(torch.linalg.vector_norm(a.float()-b.float())/torch.linalg.vector_norm(a.float()).clamp_min(1e-20))
                KV=[dict(slot=i,K_relative_L2=relative(a,b),V_relative_L2=relative(c,d)) for i,(a,b,c,d) in enumerate(
                    zip(cache.key_cache,new_cache.key_cache,cache.value_cache,new_cache.value_cache))]
                value=dict(request_id=row['request_id'],P=P,chunk=B,actual_KV_slots=len(new_cache.key_cache),
                    logits_max_abs=float((base_logits-logits).abs().max()),hidden_relative_L2=relative(hidden,final),
                    all_selected_greedy_top1_equal=bool(torch.equal(base_logits.argmax(-1),logits.argmax(-1))),
                    KV=KV,publication=publication)
                value['PASS']=value['logits_max_abs']<=TOLERANCE['logits_max_abs'] and value['hidden_relative_L2']<=TOLERANCE['hidden_relative_L2'] and value['all_selected_greedy_top1_equal'] and all(
                    max(v['K_relative_L2'],v['V_relative_L2'])<=TOLERANCE['KV_relative_L2'] for v in KV)
                results.append(value)
            del base,states,gates,hidden,cache,outputs,new_cache
        # Match native TP4's exact first four natural-prefix states.
        comparisons=[];native_root=ROOT/'results/physics/TP4_R4_chunk0_v1'
        for index in range(2):
            manifest=read(native_root/f'parity_reference_{index}.json')
            native=np.load(native_root/f'parity_reference_{index}.rank0.npz')
            inp=torch.tensor([manifest['prompt_ids']],device='cuda');past=None;vectors=[];hidden_vectors=[];own_ids=[]
            def capture_hidden(module,inputs,outputs):
                hidden_vectors.append(outputs[0].last_hidden_state[:,-1,:].detach().float().cpu().numpy())
            hook=model.model.register_forward_hook(capture_hidden)
            for step,token in enumerate(manifest['output_ids']):
                value=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                past=value.past_key_values;vectors.append(value.logits[:,-1,:].float().cpu().numpy())
                own_ids.append(int(value.logits[:,-1,:].argmax(-1)))
                inp=torch.tensor([[token]],device='cuda')
            hook.remove()
            actual=np.concatenate(vectors)
            actual_hidden=np.concatenate(hidden_vectors)
            P=len(manifest['prompt_ids']);sample_positions=[0,1,P-1,P,P+2]
            coordinates=[0,1,2,3,124,125,126,127];sampled=[]
            for key,value_tensor in zip(past.key_cache,past.value_cache):
                k=key[0,:2,sample_positions,:].permute(1,0,2)[...,coordinates]
                v=value_tensor[0,:2,sample_positions,:].permute(1,0,2)[...,coordinates]
                sampled.append(torch.cat([k,v],dim=-1).float().cpu().numpy())
            native_slots=[int(re.search(r'layers\.(\d+)\.',name).group(1)) for name in manifest['evidence'][0]['KV_names']]
            assert sorted(native_slots)==list(range(192))
            # Native registration is physical-layer-major. HF cache slots are loop-major.
            actual_KV=np.stack([sampled[slot] for slot in native_slots])
            own_past=None;own_input=torch.tensor([manifest['prompt_ids']],device='cuda');natural_ids=[]
            for step in range(4):
                own_value=model(own_input,past_key_values=own_past,use_cache=True,exit_at_step=3,logits_to_keep=1)
                own_past=own_value.past_key_values
                selected=int(own_value.logits[:,-1,:].argmax(-1));natural_ids.append(selected)
                if selected==2:break
                own_input=torch.tensor([[selected]],device='cuda')
            comparisons.append(dict(problem_id=manifest['problem_id'],same_four_greedy_ids=own_ids==manifest['output_ids'],
                independent_HF_natural_ids=natural_ids,native_natural_ids=manifest['output_ids'],
                independent_four_token_natural_equal=natural_ids==manifest['output_ids'],
                full_logits_max_abs=float(np.max(np.abs(actual-native['logits']))),
                full_logits_relative_L2=float(np.linalg.norm(actual-native['logits'])/np.linalg.norm(native['logits'])),
                last_hidden_max_abs=float(np.max(np.abs(actual_hidden-native['hidden']))),
                last_hidden_relative_L2=float(np.linalg.norm(actual_hidden-native['hidden'])/np.linalg.norm(native['hidden'])),
                sampled_KV_max_abs=float(np.max(np.abs(actual_KV-native['KV']))),
                sampled_KV_relative_L2=float(np.linalg.norm(actual_KV-native['KV'])/np.linalg.norm(native['KV'])),
                native_registered_slot_order=native_slots,
                HF_cache_slots=len(past.key_cache),scope='Cross-engine SDPA versus native TP4 FA3. Not expected bitwise. Fixed final R4 state.'))
    write(output/'reference_component_correctness.json',dict(component_results=results,cross_engine=comparisons,
        tolerance=TOLERANCE,remote_code=remote_code,all_component_pass=all(r['PASS'] for r in results),runtime=dict(torch=torch.__version__,
        transformers=__import__('transformers').__version__),
        loaded_unique_parameter_bytes=sum({p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}.values()),
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Official pretrained BF16 HF reference on one GPU. Not an optimized serving latency comparison.'))
    write(output/'status.json',dict(success=True,component_gate=all(r['PASS'] for r in results)))


if __name__=='__main__':main()
