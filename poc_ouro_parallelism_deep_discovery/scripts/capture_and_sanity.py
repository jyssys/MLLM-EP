"""Both-size fixed/adaptive smoke and bounded real activation capture, not a quality benchmark."""
import argparse
import collections
import dataclasses
import datetime
import time
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports,load_engine
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',choices=MODELS,required=True)
    args=parser.parse_args(); output=Path(os.environ['LOOPED_KV_OUTPUT'])/args.size
    output.mkdir(parents=True,exist_ok=False)
    protocol=read(ROOT/'artifacts/probe_protocol.json')
    prompts=[r for r in protocol['rows'] if r['campaign_partition']=='calibration'][:2]
    model,tok,engine=load_engine(args.size,'adaptive',graphs=False,batch=1,blocks=192,max_model_len=2048)
    L=MODELS[args.size]['layers']; chosen=[0,L//2,L-1]
    activations={}; calls=collections.Counter(); depths={}; copy_checks=[]
    provenance={'name':'unstarted'}
    layer_calls=collections.Counter()
    handles=[]
    for l,layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(lambda module,arguments,_l=l:layer_calls.update([_l])))
    for l in chosen:
        def hook(module,arguments,_l=l):
            calls[_l]+=1
            x=arguments[0]
            # Bound each tensor and the count. These values are provenance for component tests.
            key=f'{provenance["name"]}_layer{_l}_call{calls[_l]}'
            if calls[_l]<=20:
                activations[key]=x.detach().reshape(-1,x.shape[-1])[-32:].contiguous().cpu()
        handles.append(model.model.layers[l].mlp.register_forward_pre_hook(hook))
    original=engine._record_exit
    def record(item,depth):
        depths.setdefault(item.state.request_id,[]).append(depth+1)
        return original(item,depth)
    engine._record_exit=record
    oldcopy=engine.cache.copy_kv_rows
    def copy(source,targets,indices):
        oldcopy(source,targets,indices); torch.cuda.synchronize()
        for target in targets:
            assert torch.equal(engine.cache.key_cache[source].index_select(0,indices),engine.cache.key_cache[target].index_select(0,indices))
            assert torch.equal(engine.cache.value_cache[source].index_select(0,indices),engine.cache.value_cache[target].index_select(0,indices))
            copy_checks.append({'source':source,'target':target,'rows':indices.numel(),'exact':True})
    engine.cache.copy_kv_rows=copy
    runs=[]
    for mode,min_R,q in [('fixed',4,1.),('adaptive',2,.5)]:
        engine.cdb_config.min_recurrent_steps=min_R; engine.cdb_config.exit_threshold=q
        # A new engine policy instance reads this configuration when each request starts.
        engine.exit_policy=dataclasses.replace(engine.exit_policy,min_recurrent_steps=min_R,threshold=q)
        engine.runner.exit_policy=engine.exit_policy
        for index,row in enumerate(prompts):
            depths.clear(); engine.reset(); calls.clear(); layer_calls.clear()
            provenance['name']=f'{mode}_{index}'
            ids=tok.encode(row['prompt'],add_special_tokens=False)
            assert len(ids)<1024
            before=time.perf_counter()
            result=engine.generate_batch([ids],max_new_tokens=16,eos_token_id=tok.eos_token_id,warmup=False)
            torch.cuda.synchronize()
            stat=engine.last_stats
            from poc_decode_first_depth_aware_dp.runtime import stats_dict
            summary=stats_dict(stat)
            observed=[d for ds in depths.values() for d in ds]
            if mode=='fixed': assert observed and set(observed)=={4},observed
            else: assert observed and set(observed)<={2,3,4},observed
            assert sum(layer_calls.values())==L*(summary['prefill_batches']*4+summary['recurrent_batches'])
            rendered=[dict(request_id=v.request_id,output_ids=v.generated_tokens,
                text=tok.decode(v.generated_tokens,skip_special_tokens=False)) for v in result]
            runs.append({'mode':mode,'id':row['id'],'prompt_tokens':len(ids),
                'result':rendered,'depths':dict(depths),'stats':summary,'observed_layer_hooks':dict(layer_calls),
                'observer_wall_ms':(time.perf_counter()-before)*1000,
                'scope':'Heavy correctness and activation capture. Not clean serving performance or benchmark accuracy.'})
            print('SANITY',args.size,mode,index,collections.Counter(observed),flush=True)
    for handle in handles:handle.remove()
    torch.save(activations,output/'actual_mlp_inputs.pt')
    write(output/'sanity.json',dict(size=args.size,model=MODELS[args.size]['id'],revision=MODELS[args.size]['revision'],
        mask=validate_mask(),uuid=str(torch.cuda.get_device_properties(0).uuid),slots=4*L,gate_original_tensor_match=True,
        runs=runs,KV_copy_checks=copy_checks,activation_shapes={k:list(v.shape) for k,v in activations.items()},
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),quality='NOT_RUN',
        evidence_class='QUALITY_ONLY',precision='BF16',runtime='Pinned CDB original-gate live path'))

if __name__=='__main__':main()
