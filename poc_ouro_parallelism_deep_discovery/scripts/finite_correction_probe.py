"""Actual pretrained finite-trajectory correction, serial numerical emulation.

Provisional states never become a new generated sequence. This is a bounded
correctness falsifier, not a new sampler, a task-quality result, or GPU speedup.
"""
import argparse
import datetime
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports,load_engine
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch

def clone_arguments(arguments):
    return {k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in arguments.items()}

def distance(a,b):
    delta=a.float()-b.float()
    return dict(max_abs=float(delta.abs().max()),relative_L2=float(delta.norm()/b.float().norm().clamp_min(1e-12)),
        bitwise_equal=bool(torch.equal(a,b)))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True);args=p.parse_args()
    out=Path(os.environ['LOOPED_KV_OUTPUT']);model,tok,engine=load_engine(args.size,'fixed',graphs=False,batch=1,blocks=128,max_model_len=1024)
    rows=[r for r in read(ROOT/'artifacts/probe_protocol.json')['rows'] if r['campaign_partition']=='calibration']
    selected=[next(r for r in rows if r['dataset']==d) for d in ['gsm8k','math500']]
    adapter=engine.model_adapter;native=adapter.recurrent_step;calls=[]
    def recorded(*args,**kwargs):
        torch.cuda.synchronize()
        snapshot=clone_arguments(kwargs)
        result=native(*args,**kwargs)
        torch.cuda.synchronize()
        calls.append(dict(arguments=snapshot,output=result[0].clone()))
        return result
    adapter.recurrent_step=recorded
    records=[]
    for row in selected:
        calls.clear()
        ids=tok.encode(row['prompt'],add_special_tokens=False)
        assert len(ids)<1024
        ordinary=engine.generate_batch([ids],max_new_tokens=2,eos_token_id=tok.eos_token_id,warmup=False)[0]
        assert len(calls)==4,(row['id'],len(calls))
        assert all(int(c['arguments']['kv_slot'])==i for i,c in enumerate(calls))
        h0=calls[0]['arguments']['hidden_states'].clone()
        reference=[c['output'].clone() for c in calls]
        for i in range(1,4):assert torch.equal(calls[i]['arguments']['hidden_states'],reference[i-1])
        position=int(calls[0]['arguments']['position_ids'][0,0])
        table=calls[0]['arguments']['block_table'][0,0]
        cache_row=int(table[position//engine.cache.block_size])*engine.cache.block_size+position%engine.cache.block_size
        K=[v[cache_row].clone() for v in engine.cache.key_cache]
        V=[v[cache_row].clone() for v in engine.cache.value_cache]
        reference_logits=model.lm_head(reference[-1]).float()
        target=int(reference_logits.argmax(-1).item())
        assert target==ordinary.generated_tokens[-1]
        for method in ['jacobi','identity_surrogate']:
            for warmup in [0,1]:
                # Only current-token embedding or one actual exact pass seeds
                # these guesses. Reference future states enter diagnostics only.
                seed=h0.clone() if not warmup else native(**calls[0]['arguments'])[0].clone()
                guesses=[seed.clone() for _ in range(4)]
                for iteration in range(1,5):
                    evaluated=[]
                    for r in range(4):
                        kwargs=dict(calls[r]['arguments'])
                        kwargs['hidden_states']=h0 if r==0 else guesses[r-1]
                        evaluated.append(native(**kwargs)[0])
                    if method=='jacobi':updated=[v.clone() for v in evaluated]
                    else:
                        residual=torch.stack([f.float()-x.float() for f,x in zip(evaluated,guesses)])
                        correction=residual.cumsum(0)
                        updated=[(x.float()+correction[r]).to(torch.bfloat16) for r,x in enumerate(guesses)]
                    logits=model.lm_head(updated[-1]).float()
                    logp=reference_logits.log_softmax(-1);logq=logits.log_softmax(-1)
                    kl=float((logp.exp()*(logp-logq)).sum(-1).mean())
                    kv_error=max(float((v[cache_row].float()-ref.float()).abs().max())
                        for tensors,refs in [(engine.cache.key_cache,K),(engine.cache.value_cache,V)]
                        for v,ref in zip(tensors,refs))
                    record=dict(id=row['id'],size=args.size,method=method,exact_warmup_passes=warmup,
                        correction_round=iteration,expensive_core_evaluations=warmup+4*iteration,
                        ideal_P4_core_rounds_without_overhead=warmup+iteration,
                        state_errors=[distance(x,y) for x,y in zip(updated,reference)],
                        logit_KL_reference_to_provisional=kl,greedy_token_agreement=int(logits.argmax(-1).item())==target,
                        current_token_KV_max_abs=kv_error,KV_scope='All physical layers and R4 current rows. Historical rows are not written by these one-token calls.',
                        causal_initialization=True,
                        final_boundary_does_not_certify_internal_KV=True,
                        execution='Serial one-GPU numerical emulation. No physical four-way solver or task-quality result.')
                    if method=='jacobi' and iteration==4:
                        assert all(v['bitwise_equal'] for v in record['state_errors'])
                        assert kv_error==0 and record['greedy_token_agreement']
                    records.append(record);guesses=updated
                    write(out/'finite_correction.json',dict(records=records,
                        label='QUALITY_ONLY bounded state/KV test. Not free-running candidate accuracy.',
                        source='Pinned original-gate CDB full-R4 model path. Actual pretrained BF16 checkpoint.',
                        model_revision=MODELS[args.size]['revision'],physical_gpu=validate_mask()[0],
                        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        cost_scope='Ideal core-round count omits all transport, synchronization, correction, validation, and KV repair. Not a measured speedup.',
                        prior='Jacobi finite-chain correction and identity-surrogate correction are existing mechanisms. See parallel_sampling_nearest_audit.md.'))
        print('FINITE_CORRECTION',args.size,row['id'],flush=True)
    adapter.recurrent_step=native
    write(out/'status.json',dict(success=True,cases=len(selected),records=len(records),new_candidate_accuracy='NOT_RUN',live_speedup='NOT_RUN'))

if __name__=='__main__':main()
