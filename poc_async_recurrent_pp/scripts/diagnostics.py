"""Teacher-forced injection diagnostics and cache/write audits, not accuracy."""
import argparse
import copy
import os
import torch
import numpy as np
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.scripts.baseline import load, token_ids
from poc_async_recurrent_pp.emulator import StageEmulator, generate_batch, cache_signature


def divergence(a,b):
    p=a.float().log_softmax(-1)
    q=b.float().log_softmax(-1)
    return dict(KL_exact_candidate_nats=float((p.exp()*(p-q)).sum()),
                KL_candidate_exact_nats=float((q.exp()*(q-p)).sum()),
                top1_agreement=bool(a.argmax()==b.argmax()),
                candidate_finite=bool(torch.isfinite(b).all()))


@torch.inference_mode()
def main():
    model,tokenizer=load()
    data=read(ROOT/'outputs/manifests/data.json')
    coef=read(ROOT/'outputs/manifests/predictor.json')
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    # Official batched generate vs manual left-padding/compaction semantics.
    rows=data['smoke'][:2]
    prompts=[token_ids(tokenizer,row,data['prompt_suffix']) for row in rows]
    width=max(map(len,prompts))
    inp=torch.tensor([[2]*(width-len(ids))+ids for ids in prompts],device='cuda')
    mask=torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for ids in prompts],device='cuda')
    official=model.generate(inp,attention_mask=mask,max_new_tokens=32,do_sample=False,
        eos_token_id=2,pad_token_id=2,exit_at_step=3)
    manual=generate_batch(model,tokenizer,prompts,32)
    for i,row in enumerate(manual):
        ids=official[i,width:].tolist()
        if 2 in ids:
            ids=ids[:ids.index(2)+1]
        assert ids==row['ids'], 'Batched generation differs from official generate'
    write(output/'batch_parity.json',dict(exact=True,n=2,cap=32,output_ids=[r['ids'] for r in manual]))
    # All targeted interventions use independent calibration prompts only.
    configurations=[('stale',None,None),('delta',None,None),('w0',None,None)]
    configurations += [('delta',[j],None) for j in [1,2,3]]
    configurations += [('delta',[1,2],None),('delta',[2,3],None),('delta',None,[1])]
    metrics=[]
    boundary=[]
    cache_audit=[]
    for row in data['calibration'][:2]:
        prompts=token_ids(tokenizer,row,data['prompt_suffix'])
        result=model(torch.tensor([prompts],device='cuda'),use_cache=True,exit_at_step=3,logits_to_keep=1)
        prefix=copy.deepcopy(result.past_key_values)
        teachers=[]
        exact_logits=[]
        exact_states=[]
        hook=StageEmulator(model,capture=True)
        token=int(result.logits[0,-1].argmax())
        for step in range(32):
            teachers.append(token)
            hook.captures=[]
            result=model(torch.tensor([[token]],device='cuda'),past_key_values=result.past_key_values,
                         use_cache=True,exit_at_step=3,logits_to_keep=1)
            exact_logits.append(result.logits[0,-1].detach().clone())
            exact_states.append({(v['round'],v['stage']):v['value'].numpy() for v in hook.captures})
            token=int(result.logits[0,-1].argmax())
        hook.close()
        exact_cache=result.past_key_values
        plain=copy.deepcopy(prefix)
        for step,token in enumerate(teachers):
            check=model(torch.tensor([[token]],device='cuda'),past_key_values=plain,
                        use_cache=True,exit_at_step=3,logits_to_keep=1)
            assert torch.equal(check.logits[0,-1],exact_logits[step])
        assert all(torch.equal(a,b) for a,b in zip(exact_cache.key_cache,plain.key_cache))
        assert all(torch.equal(a,b) for a,b in zip(exact_cache.value_cache,plain.value_cache))
        write(output/(row['id'].replace('/','_')+'_instrumented_exact_parity.json'),
              dict(teacher_positions=32,all_logits_bitwise=True,all_96_KV_keys_values_bitwise=True))
        for arm,stages,rounds in configurations:
            hook=StageEmulator(model,arm,coef,stages=stages or (1,2,3),rounds=rounds or (1,2,3),capture=True)
            past=copy.deepcopy(prefix)
            label=arm+'_stages'+''.join(map(str,sorted(hook.stages)))+'_rounds'+''.join(map(str,sorted(hook.rounds)))
            writes=[]
            original=past.update
            def observed_update(key,value,idx,kwargs=None):
                writes.append(idx)
                return original(key,value,idx,kwargs)
            past.update=observed_update
            for step,token in enumerate(teachers):
                hook.captures=[]
                before=len(writes)
                result=model(torch.tensor([[token]],device='cuda'),past_key_values=past,
                             use_cache=True,exit_at_step=3,logits_to_keep=1)
                assert writes[before:]==list(range(96)), 'Depth/cache write count or order differs'
                cache_signature(past)
                metrics.append(dict(id=row['id'],arm=label,step=step,**divergence(exact_logits[step],result.logits[0,-1])))
                if arm in ['stale','delta','w0'] and stages is None and rounds is None:
                    for state in hook.captures:
                        r,j=state['round'],state['stage']
                        if j==0:
                            continue
                        predicted=state['value'].numpy()
                        truth=exact_states[step][r,j]
                        boundary.append(dict(id=row['id'],arm=arm,step=step,round=r,stage=j,
                            relative_L2=float(np.linalg.norm(predicted-truth)/(np.linalg.norm(truth)+1e-12)),
                            cosine=float(np.sum(predicted*truth)/(np.linalg.norm(predicted)*np.linalg.norm(truth)+1e-12))))
            cache_audit.append(dict(id=row['id'],arm=label,positions=len(teachers),
                total_writes=len(writes),one_write_per_depth_layer_position=True,
                independent_candidate_cache=True,exact_KV_reset_after_decode=False))
            write(output/(row['id'].replace('/','_')+'_'+label+'_availability.json'),hook.events)
            trace=hook.resolved_gpu_trace()
            keyed={(event['token_position'],event['round'],event['stage'],event['kind']):event['GPU_ms_from_local_anchor'] for event in trace}
            checked=0
            checked_sources=0
            for event in trace:
                if event['kind']!='prediction_ready':
                    continue
                p,r,j=event['token_position'],event['round'],event['stage']
                consumer=keyed.get((p,r,j,'consumer_launch_boundary'))
                if consumer is not None:
                    assert event['GPU_ms_from_local_anchor']<=consumer
                    checked+=1
                sources=[(p,r,0,'round_input_ready')]
                if r>0:
                    sources += [(p,r-1,j-1,'stage_output_ready'),(p,r-1,0,'round_input_ready')]
                for key in sources:
                    if key in keyed:
                        assert keyed[key]<=event['GPU_ms_from_local_anchor']
                        checked_sources+=1
            write(output/(row['id'].replace('/','_')+'_'+label+'_GPU_availability.json'),
                  dict(events=trace,prediction_before_consumer_checks=checked,source_before_prediction_checks=checked_sources,
                       conclusion='Causal emulator availability only. No distributed concurrency claim.'))
            hook.close()
        print('diagnostics',row['id'],flush=True)
    write(output/'teacher_metrics.json',metrics)
    write(output/'boundary_errors.json',boundary)
    write(output/'cache_audit.json',cache_audit)
    write(output/'status.json',dict(success=True,calibration_prompts=2,teacher_positions=64,
        scope='Teacher-forced diagnostic and intervention ablations. Not task accuracy or pipeline latency.'))


if __name__=='__main__':
    main()
