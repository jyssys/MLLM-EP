"""Free-running greedy quality and online depth traces from the existing CDB engine."""
import argparse
import collections
import dataclasses
import os
import time
import torch
from poc_decode_first_depth_aware_dp.contract import *
from poc_decode_first_depth_aware_dp.runtime import load_engine,stats_dict


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=['fixed','dynamic'],required=True)
    parser.add_argument('--part',type=int,choices=[0,1],required=True);parser.add_argument('--graphs',action='store_true')
    parser.add_argument('--cohort',choices=['pilot','full'],default='pilot')
    parser.add_argument('--protocol',type=Path,default=ROOT/'artifacts/quality_protocol_v2.json')
    parser.add_argument('--extension-only',action='store_true')
    parser.add_argument('--ids-only',nargs='+')
    parser.add_argument('--max-new-tokens',type=int,choices=[4096,8192],default=4096);args=parser.parse_args()
    output=Path(os.environ['LOOPED_KV_OUTPUT']);q=None if args.policy=='fixed' else .5
    max_model_len=8192 if args.max_new_tokens==4096 else 10240
    model,tokenizer,engine=load_engine(q,graphs=args.graphs,max_model_len=max_model_len)
    protocol=read(args.protocol);rows=protocol['rows']
    selected=[r for r in rows if args.cohort=='full' or r['split']=='pilot']
    if args.extension_only:
        extension=set(protocol['extension_ids'])
        selected=[r for r in selected if r['id'] in extension]
    if args.ids_only:
        assert set(args.ids_only)<={r['id'] for r in selected}
        selected=[r for r in selected if r['id'] in set(args.ids_only)]
    selected=[r for i,r in enumerate(selected) if i%2==args.part]
    write(output/'quality_contract.json',dict(policy=args.policy,q=q,min_depth=2,max_depth=4,KV='last_exited',
        graphs=args.graphs,online=True,replay=False,selected_ids=[r['id'] for r in selected],cap=args.max_new_tokens,
        max_model_len=max_model_len,
        model=MODEL_ID,revision=REVISION,CDB_revision=CDB_REVISION,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])))
    # Warmup is separate from all scored requests.
    engine.generate_batch([tokenizer.encode('Q: What is 1 + 1?\nA:')],max_new_tokens=4,
        eos_token_id=tokenizer.eos_token_id,warmup=True)
    original_record=engine._record_exit
    events=[]
    def record(item,depth):
        events.append(dict(token_position=item.token_position,exit_depth_executed=depth+1,
            scheduler_tick=engine.last_stats.scheduler_ticks,request_id=item.state.request_id))
        if len(events)%1024==0:
            print('DECODE_PROGRESS',args.policy,item.state.request_id,len(events),flush=True)
        return original_record(item,depth)
    engine._record_exit=record
    summaries=[]
    for row in selected:
        events.clear();ids=tokenizer.encode(row['prompt'],add_special_tokens=False)
        stops=[tokenizer.encode(s,add_special_tokens=False) for s in row['stop_strings']]
        assert len(ids)+args.max_new_tokens<=max_model_len,'Do not truncate the input or silently reduce the generation budget.'
        assert len(ids)+args.max_new_tokens<=640*16,'The actual paged KV allocator must fit the full bounded request.'
        torch.cuda.synchronize();start=time.perf_counter()
        value=engine.generate_batch([ids],max_new_tokens=args.max_new_tokens,eos_token_id=tokenizer.eos_token_id,
            stop_sequences=stops,warmup=False)[0]
        torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
        tokens=value.generated_tokens
        finish='eos' if tokens and tokens[-1]==tokenizer.eos_token_id else 'stop' if any(tokens[-len(s):]==s for s in stops if s) else 'length'
        stats=stats_dict(engine.last_stats)
        assert stats['prefill_recurrent_steps']==4*len(ids)
        assert stats['recurrent_steps']==sum(e['exit_depth_executed'] for e in events)
        assert len(events)==len(tokens)-1,(len(events),len(tokens))
        if q is None:assert all(e['exit_depth_executed']==4 for e in events)
        for index,event in enumerate(events,1):
            event.update(output_token_index=index,output_token_id=tokens[index],
                input_token_id=tokens[index-1],phase='decode',
                physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),logical_cuda_rank=0)
        record=dict(id=row['id'],dataset=row['dataset'],policy=args.policy,prompt_ids=ids,prompt_tokens=len(ids),
            output_ids=tokens,text=tokenizer.decode(tokens,skip_special_tokens=False),finish_reason=finish,
            first_token=dict(output_token_index=0,depth=4,phase='prefill',adaptive_decode=False),
            output_length=len(tokens),max_new_tokens=args.max_new_tokens,depth_events=list(events),
            mean_actual_decode_depth=statistics_mean([e['exit_depth_executed'] for e in events]),
            total_decode_loop_work=stats['recurrent_steps'],full_depth_decode_work=stats['fixed_depth_recurrent_steps'],
            wall_ms=wall,TTFT_ms=(value.first_token_time-value.created_time)*1000,
            decode_span_ms=(value.lifespan[1]-value.first_token_time)*1000,stats=stats,
            memory_allocated=torch.cuda.memory_allocated(),peak_memory=torch.cuda.max_memory_allocated(),
            label='MEASURED_ONLINE_FREE_RUNNING_GREEDY. Per-token trace is not a future admission feature.')
        write(output/(row['id'].replace('/','_')+'.json'),record)
        summaries.append(dict(id=row['id'],tokens=len(tokens),finish=finish,loops=stats['recurrent_steps'],wall_ms=wall))
        write(output/'progress.json',summaries)
        print('QUALITY',args.policy,row['id'],len(tokens),finish,round(record['mean_actual_decode_depth'],4),flush=True)
    write(output/'status.json',dict(success=True,policy=args.policy,part=args.part,requests=len(selected),
        quality_scoring='PENDING_OFFLINE_PAIRED_ANALYSIS',online_policy=True,graphs=args.graphs))


def statistics_mean(values):
    return sum(values)/len(values) if values else None


if __name__=='__main__':main()
