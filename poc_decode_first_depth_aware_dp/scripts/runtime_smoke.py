"""Check online gate decisions, actual core launches, and exit-KV propagation."""
import collections
import dataclasses
import gc
import os
import time
import torch
from poc_decode_first_depth_aware_dp.contract import *
from poc_decode_first_depth_aware_dp.runtime import load_engine,stats_dict


@torch.inference_mode()
def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    assert (ROOT/'artifacts/checkpoint.json').exists()
    model,tokenizer,engine=load_engine(None,graphs=False)
    import transformers
    write(output/'environment.json',dict(model=MODEL_ID,revision=REVISION,torch=torch.__version__,cuda=torch.version.cuda,
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),UUID=str(torch.cuda.get_device_properties(0).uuid),
        transformers=transformers.__version__,config=model.config.to_dict(),engine=dataclasses.asdict(engine.cdb_config),
        unique_weights=sum({p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}.values()),
        KV_storages=96,KV_bytes=sum(t.numel()*t.element_size() for t in engine.cache.key_cache+engine.cache.value_cache)))
    cases=[]
    for q in [None,.5]:
        engine.cdb_config.exit_threshold=q
        engine.exit_policy=dataclasses.replace(engine.exit_policy,threshold=q)
        engine.runner.exit_policy=engine.exit_policy
        engine.model_adapter.configure_exit_signal(q is not None)
        layers=collections.Counter();exits=[];copies=[];logits=[]
        for layer in model.model.layers:
            def before(module,args,kwargs):layers['calls']+=1
            layer._poc_handle=layer.register_forward_pre_hook(before,with_kwargs=True)
        original_exit=engine._record_exit
        def record(item,depth):
            exits.append(dict(request=item.state.request_id,position=item.token_position,depth=depth+1))
            return original_exit(item,depth)
        engine._record_exit=record
        original_copy=engine.cache.copy_kv_rows
        def copied(source,targets,rows):
            original_copy(source,targets,rows);torch.cuda.synchronize()
            check=all(torch.equal(engine.cache.key_cache[source].index_select(0,rows),engine.cache.key_cache[t].index_select(0,rows))
                and torch.equal(engine.cache.value_cache[source].index_select(0,rows),engine.cache.value_cache[t].index_select(0,rows)) for t in targets)
            assert check
            copies.append(dict(source=source,targets=targets,rows=rows.numel(),exact_identity=True))
        engine.cache.copy_kv_rows=copied
        original_retire=engine._retire_request
        def retire(state):
            if q is None:
                import numpy as np
                indices=torch.tensor(engine.cache.cache_allocator.get_write_indices(state.request_id,0,state.current_len()),device='cuda')
                torch.cuda.synchronize()
                np.savez_compressed(output/'fixed_KV.npz',
                    keys=torch.stack([t.index_select(0,indices) for t in engine.cache.key_cache]).cpu().float().numpy(),
                    values=torch.stack([t.index_select(0,indices) for t in engine.cache.value_cache]).cpu().float().numpy())
            return original_retire(state)
        engine._retire_request=retire
        head=model.lm_head.register_forward_hook(lambda module,args,result:logits.append(result.detach().reshape(-1,49152)[-1].cpu().float().numpy()))
        prompt=tokenizer.encode('Q: What is 2 + 3?\nA:')
        torch.cuda.synchronize();start=time.perf_counter()
        result=engine.generate_batch([prompt],max_new_tokens=16,eos_token_id=tokenizer.eos_token_id,warmup=False)[0]
        torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
        stats=stats_dict(engine.last_stats)
        assert stats['prefill_recurrent_steps']==4*len(prompt)
        assert stats['recurrent_steps']==sum(r['depth'] for r in exits)
        assert layers['calls']==24*(4+stats['recurrent_batches'])
        if q is None:assert all(r['depth']==4 for r in exits)
        case=dict(q=q,ids=result.generated_tokens,text=tokenizer.decode(result.generated_tokens,skip_special_tokens=False),
            wall_ms=wall,stats=stats,actual_layer_calls=layers['calls'],online_exit_events=exits,exit_copy_checks=len(copies),
            all_exit_copies_identical=all(r['exact_identity'] for r in copies),first_token_full_depth=True,
            scope='Diagnostic observer includes per-copy synchronization. Not clean serving latency.')
        cases.append(case);write(output/f'q_{q}.json',case)
        import numpy as np
        np.savez_compressed(output/f'logits_q_{q}.npz',logits=np.stack(logits))
        engine._record_exit=original_exit;engine.cache.copy_kv_rows=original_copy;engine._retire_request=original_retire;head.remove()
        for layer in model.model.layers:layer._poc_handle.remove();del layer._poc_handle
        print('SMOKE',q,stats['recurrent_steps'],stats['fixed_depth_recurrent_steps'],flush=True)
    write(output/'status.json',dict(success=True,actual_online_skipping=cases[1]['stats']['skipped_recurrent_steps']>0 if 'skipped_recurrent_steps' in cases[1]['stats'] else
        cases[1]['stats']['recurrent_steps']<cases[1]['stats']['fixed_depth_recurrent_steps'],
        random_or_replayed_exit_policy=False,quality='NOT_YET_EVALUATED',no_DP_scheduling_claim=True))


if __name__=='__main__':main()
