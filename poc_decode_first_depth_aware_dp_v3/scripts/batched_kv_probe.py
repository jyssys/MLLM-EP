"""Check terminal-KV publication for multiple independent CDB sequences."""
import collections
import os
import torch
from poc_decode_first_depth_aware_dp.runtime import load_engine,stats_dict
from poc_decode_first_depth_aware_dp_v3.contract import *

@torch.inference_mode()
def main():
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    model,tok,engine=load_engine(graphs=False,batch=4,blocks=2048,max_model_len=8192)
    rows=[r for r in read(ROOT/'artifacts/discovery_frozen.json')['rows'] if r['split']=='calibration'][:4]
    checks=[]
    calls=collections.Counter()
    handles=[layer.register_forward_pre_hook(lambda module,args:calls.update(['layer'])) for layer in model.model.layers]
    original=engine.cache.copy_kv_rows
    def copied(source,targets,indices):
        original(source,targets,indices)
        torch.cuda.synchronize()
        valid=all(torch.equal(engine.cache.key_cache[source].index_select(0,indices),engine.cache.key_cache[t].index_select(0,indices))
            and torch.equal(engine.cache.value_cache[source].index_select(0,indices),engine.cache.value_cache[t].index_select(0,indices))
            for t in targets)
        assert valid
        checks.append(dict(source=source,targets=targets,rows=indices.numel(),exact=True))
    engine.cache.copy_kv_rows=copied
    values=engine.generate_batch([tok.encode(r['prompt'],add_special_tokens=False) for r in rows],
        max_new_tokens=16,eos_token_id=tok.eos_token_id,warmup=False)
    stats=stats_dict(engine.last_stats)
    assert calls['layer']==24*(4*stats['prefill_batches']+stats['recurrent_batches'])
    assert stats['recurrent_steps']<stats['fixed_depth_recurrent_steps']
    for h in handles:h.remove()
    write(out/'batched_kv.json',dict(checks=checks,stats=stats,actual_layer_calls=calls['layer'],
        outputs=[dict(id=r['id'],tokens=v.generated_tokens) for r,v in zip(rows,values)],
        scope='Heavy correctness observer. Not a timing or task-accuracy result.'))
    write(out/'status.json',dict(success=True,checks=len(checks),batched_exact_copy=True,
        physical_work_saved=True,quality='NOT_EVALUATED'))

if __name__=='__main__':main()
