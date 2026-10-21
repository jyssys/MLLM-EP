"""Pinned official Single-GPU dynamic reference, with actual online exit traces."""
import argparse
import time
import torch
from poc_decode_first_depth_aware_dp.runtime import load_engine,stats_dict
from .contract import *

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--limit',type=int)
    p.add_argument('--cap',type=int,default=4096);p.add_argument('--full',action='store_true')
    p.add_argument('--validate-KV',action='store_true');p.add_argument('--batch',type=int,default=1)
    p.add_argument('--probe',action='store_true');a=p.parse_args()
    assert __import__('os').environ['CUDA_VISIBLE_DEVICES']=='4'
    output=ROOT/'raw'/a.label;output.mkdir(exist_ok=False)
    model,tok,engine=load_engine(.5,graphs=True,batch=a.batch,blocks=max(640,512*a.batch),refill=False,max_model_len=8192)
    finite=torch.ones((),device='cuda',dtype=torch.bool)
    def finite_hook(module,inputs,value):finite.logical_and_(torch.isfinite(value).all())
    handle=model.lm_head.register_forward_hook(finite_hook)
    warm=tok.encode('Q: What is 1 + 1?\nA:')
    for batch in range(1,a.batch+1):
        engine.generate_batch([warm]*batch,max_new_tokens=8,eos_token_id=tok.eos_token_id,warmup=True)
    rows=dataset(not a.full)
    if a.probe:
        rows=[]
        for i in range(4):
            prior=read(ROOT/'raw/probe_S4_v3'/f'probe-{i}.json')
            rows.append(dict(id=prior['id'],dataset='calibration',prompt_sha256=prior['prompt_sha256'],
                input_ids=prior['prompt_ids'],stop_strings=[]))
    if a.limit:rows=rows[:a.limit]
    original=engine._record_exit;events=[];copies=[]
    def record(item,depth):
        events.append(dict(position=item.token_position,depth=depth+1,request=item.state.request_id,
            cdf=item.policy_state.exit_cdf,survival=item.policy_state.survival))
        return original(item,depth)
    engine._record_exit=record
    original_copy=engine.cache.copy_kv_rows
    if a.validate_KV:
        def copy(source,targets,positions):
            original_copy(source,targets,positions)
            exact=all(torch.equal(cache[target].index_select(0,positions),cache[source].index_select(0,positions))
                for target in targets for cache in [engine.cache.key_cache,engine.cache.value_cache])
            assert exact,(source,targets)
            copies.append(dict(source=source,targets=targets,rows=positions.cpu().tolist(),bitwise=True))
        engine.cache.copy_kv_rows=copy
    for offset in range(0,len(rows),a.batch):
        group=rows[offset:offset+a.batch];events.clear();copies.clear()
        inputs=[r.get('input_ids') or tok.encode(r['prompt'],add_special_tokens=False) for r in group]
        assert all(len(ids)+a.cap<=8192 for ids in inputs)
        assert all(r['stop_strings']==group[0]['stop_strings'] for r in group)
        stops=[tok.encode(s,add_special_tokens=False) for s in group[0]['stop_strings']]
        finite.fill_(True)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
        values=engine.generate_batch(inputs,max_new_tokens=a.cap,eos_token_id=tok.eos_token_id,stop_sequences=stops)
        torch.cuda.synchronize();end=time.perf_counter();stats=stats_dict(engine.last_stats)
        assert bool(finite),'Nonfinite logits in the official reference'
        assert len(events)==sum(len(v.generated_tokens)-1 for v in values)
        assert stats['recurrent_steps']==sum(v['depth'] for v in events)
        for row,ids,value in zip(group,inputs,values):
            out=value.generated_tokens;selected=[e for e in events if e['request']==value.request_id]
            assert len(selected)==len(out)-1
            finish='eos' if out[-1]==tok.eos_token_id else 'length' if len(out)==a.cap else 'stop'
            record_=dict(id=row['id'],dataset=row['dataset'],arm='SD',GPUs=[4],engine=f'Pinned official CDB, local capacity {a.batch}, no-refill, FA3',
                prompt_sha256=row['prompt_sha256'],prompt_ids=ids,output_ids=out,text=tok.decode(out,skip_special_tokens=False),
                finish_reason=finish,output_tokens=len(out),depths=[4]+[v['depth'] for v in selected],
                max_new_tokens=a.cap,evaluation_eligible=a.cap==4096 and not a.probe,
                mean_depth=(4+sum(v['depth'] for v in selected))/len(out),depth_events=selected,
                E2E_s=value.lifespan[1]-begin,TTFT_s=value.first_token_time-begin,
                TPOT_s=(value.lifespan[1]-value.first_token_time)/(len(out)-1) if len(out)>1 else None,
                cohort_index=offset//a.batch,cohort_C=len(group),peak_allocated=torch.cuda.max_memory_allocated(),status='REFERENCE_POLICY_LIVE',
                lane='ACTUAL_GREEDY_CDB_REFERENCE. Production-oriented research engine, not the custom FIFO comparison.')
            write(output/(row['id'].replace('/','_')+'.json'),record_)
            print('QUALITY SD',row['id'],len(out),finish,round(record_['mean_depth'],3),flush=True)
        write(output/f'cohort_{offset//a.batch:03d}.json',dict(ids=[r['id'] for r in group],makespan_s=end-begin,stats=stats,KV_checks=list(copies),
            peak_allocated=torch.cuda.max_memory_allocated(),batch_capacity=a.batch))
    handle.remove();write(output/'complete.json',dict(success=True,requests=len(rows),arm='SD'))

if __name__=='__main__':main()
