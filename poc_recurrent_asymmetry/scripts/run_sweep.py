"""Exact HF shape sweep. Clean timing and observer lanes remain separate."""
import argparse
import copy
import gc
import os
import time
import statistics
import torch
from poc_recurrent_asymmetry.contract import *
from poc_recurrent_asymmetry.observer import Observer
from poc_async_recurrent_pp.scripts.baseline import load


def memory(cache):
    tensors=cache.key_cache+cache.value_cache
    return dict(logical_KV_bytes=sum(t.numel()*t.element_size() for t in tensors),
        allocated_KV_storage_bytes=sum({t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}.values()),
        slots=len(cache.key_cache),key_storage_count=len({t.untyped_storage().data_ptr() for t in cache.key_cache}),
        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())


def call(model,ids,past=None):
    return model(ids,past_key_values=past,use_cache=True,exit_at_step=model.model.total_ut_steps-1,logits_to_keep=1)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--context',type=int,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'outputs/baseline_v1');args=parser.parse_args()
    assert read(args.baseline/'status.json')['exact_reference']
    output=Path(os.environ['LOOPED_KV_OUTPUT']);manifest=read(ROOT/'metrics/frozen_workload.json')
    model,tokenizer=load();P=args.context
    assert P in manifest['contexts']
    rows=manifest['requests'];repeats=manifest['repeats'];steps=manifest['decode_steps']
    info=dict(physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),UUID=str(torch.cuda.get_device_properties(0).uuid),
        model_revision=REVISION,torch=torch.__version__,context=P,backend='Official HF SDPA, dynamic depth KV, eager',
        sample_unit='A frozen request or fixed request group. Technical repeats are not independent requests.',
        generation='Fixed 32 measured cached forwards with frozen token ID 13. Not natural generation or user-visible E2E.')
    write(output/'contract.json',info)
    # Initial warmup is not a timed sample.
    warm=call(model,torch.tensor([rows[0]['prompts']['1024'][:128]],device='cuda'))
    for _ in range(3):warm=call(model,torch.tensor([[13]],device='cuda'),warm.past_key_values)
    del warm;gc.collect();torch.cuda.empty_cache()
    clean=[];skipped=[]
    for B in manifest['batches']:
        logical=(P+steps)*B*96*2*16*128*2
        total=torch.cuda.get_device_properties(0).total_memory
        if logical+12*1024**3>total*.90:
            skipped.append(dict(context=P,batch=B,status='CAPACITY_SAFETY_SKIP',
                logical_KV_bytes=logical,device_bytes=total,
                reason='Audited logical KV plus 12 GiB safety reserve exceeds 90% HBM. Not an observed OOM.'))
            continue
        for start in range(0,len(rows),B):
            selected=rows[start:start+B]
            ids=torch.tensor([r['prompts'][str(P)] for r in selected],device='cuda')
            token=torch.full((B,1),13,dtype=torch.long,device='cuda')
            for repeat in range(repeats):
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize();host=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();value=call(model,ids);b.record();torch.cuda.synchronize()
                prefill_wall=(time.perf_counter()-host)*1000;prefill_stream=a.elapsed_time(b)
                pre_mem=memory(value.past_key_values)
                pairs=[];torch.cuda.synchronize();host=time.perf_counter()
                for step in range(steps):
                    x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True)
                    x.record();value=call(model,token,value.past_key_values);y.record();pairs.append((x,y))
                torch.cuda.synchronize();decode_wall=(time.perf_counter()-host)*1000
                lengths=[t.shape[2] for t in value.past_key_values.key_cache]
                assert set(lengths)=={P+steps} and len(lengths)==96
                sample=dict(context=P,batch=B,request_ids=[r['id'] for r in selected],repeat=repeat,
                    prefill_wall_ms=prefill_wall,prefill_stream_ms=prefill_stream,
                    decode_wall_ms=decode_wall,decode_steps=steps,
                    cached_step_stream_ms=[x.elapsed_time(y) for x,y in pairs],
                    teacher_token_id=13,exact_R=4,prefill_memory=pre_mem,final_memory=memory(value.past_key_values),
                    complete_fixed_step_wall_ms=prefill_wall+decode_wall,
                    scope='Measured exact fixed-step hardware characterization, not natural-output latency.')
                clean.append(sample);write(output/f'clean_B{B}_group{start}_repeat{repeat}.json',sample)
                del value
            print('CLEAN',P,B,start,round(prefill_wall,3),round(decode_wall/steps,3),flush=True)
            del ids,token;gc.collect();torch.cuda.empty_cache()
    # One representative trace is separate from all clean observations.
    ids=torch.tensor([rows[0]['prompts'][str(P)]],device='cuda')
    for phase in ['prefill','decode']:
        previous=call(model,ids).past_key_values if phase=='decode' else None
        inp=torch.tensor([[13]],device='cuda') if phase=='decode' else ids
        # Observer parity begins from identical independent historical cache.
        plain_cache=copy.deepcopy(previous) if previous is not None else None
        reference=call(model,inp,plain_cache)
        with Observer(model) as observer:
            start=time.perf_counter();value=call(model,inp,previous);torch.cuda.synchronize();observed_wall=(time.perf_counter()-start)*1000
        timeline=observer.resolve()
        assert torch.equal(value.logits,reference.logits)
        assert all(torch.equal(x,y) for x,y in zip(value.past_key_values.key_cache,reference.past_key_values.key_cache))
        assert all(torch.equal(x,y) for x,y in zip(value.past_key_values.value_cache,reference.past_key_values.value_cache))
        relevant=[x for x in clean if x['batch']==1 and x['request_ids']==[rows[0]['id']]]
        clean_reference=statistics.median(x['prefill_wall_ms'] if phase=='prefill' else x['decode_wall_ms']/steps for x in relevant)
        write(output/f'{phase}_timeline.json',dict(events=timeline,logits_bitwise=True,all_KV_bitwise=True,
            observer_wall_ms=observed_wall,clean_reference_ms=clean_reference,
            observer_percent=(observed_wall/clean_reference-1)*100,
            scope='Bounded diagnostic GPU-stream intervals. Not headline clean timing.'))
        del value,reference,plain_cache,previous;gc.collect();torch.cuda.empty_cache()
    # Serial native-supported chunking is a component diagnostic, not a new pipeline engine.
    if P<=8192:
        chunk_records=[]
        reference=call(model,ids);reference_logits=reference.logits.clone();del reference;gc.collect();torch.cuda.empty_cache()
        for chunk in manifest['chunks']:
            clean_chunk=[]
            for repeat in range(5):
                past=None;torch.cuda.synchronize();start=time.perf_counter()
                for offset in range(0,P,chunk):
                    value=call(model,ids[:,offset:offset+chunk],past);past=value.past_key_values
                torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                clean_chunk.append(wall)
                write(output/f'chunk_{chunk}_clean_repeat{repeat}.json',dict(context=P,chunk_rows=chunk,
                    repeat=repeat,wall_ms=wall,scope='Measured exact serial chunked prefill. No component observer.'))
                del past,value;gc.collect();torch.cuda.empty_cache()
            for repeat in range(1):
                past=None
                with Observer(model,detail=False) as observer:
                    torch.cuda.synchronize();start=time.perf_counter()
                    for c,offset in enumerate(range(0,P,chunk)):
                        observer.chunk=c
                        value=call(model,ids[:,offset:offset+chunk],past)
                        past=value.past_key_values
                    torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                difference=(value.logits.float()-reference_logits.float()).abs()
                row=dict(context=P,chunk_rows=chunk,repeat=repeat,wall_ms=wall,
                    clean_wall_ms=clean_chunk,observer_percent=(wall/statistics.median(clean_chunk)-1)*100,
                    last_logit_max_abs=float(difference.max()),last_logit_mean_abs=float(difference.mean()),
                    last_token_equal=bool(value.logits.argmax()==reference_logits.argmax()),
                    events=observer.resolve(),scope='Measured serial chunk component lane. A different GEMM shape can change BF16 rounding.')
                chunk_records.append(row);write(output/f'chunk_{chunk}_repeat{repeat}.json',row)
                del past,value;gc.collect();torch.cuda.empty_cache()
            print('CHUNK',P,chunk,flush=True)
    # R1/R2 are diagnostic computation counts, not accuracy-equivalent candidates.
    if P==1024:
        recurrence=[]
        for R in [1,2,4]:
            model.model.total_ut_steps=R
            for repeat in range(5):
                torch.cuda.synchronize();start=time.perf_counter();value=call(model,ids);torch.cuda.synchronize()
                recurrence.append(dict(R=R,repeat=repeat,prefill_ms=(time.perf_counter()-start)*1000,
                    physical_weight_bytes=sum({p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in model.parameters()}.values()),
                    logical_executions=R*24,cache_slots=len(value.past_key_values.key_cache),
                    KV_bytes=memory(value.past_key_values)['logical_KV_bytes'],scope='Changed-depth diagnostic. Not quality-matched.'))
                del value;gc.collect();torch.cuda.empty_cache()
        model.model.total_ut_steps=4
        write(output/'recurrence_diagnostic.json',recurrence)
    write(output/'skipped.json',skipped)
    write(output/'status.json',dict(success=True,clean_samples=len(clean),skipped=skipped,exact_R4_primary=True))


if __name__=='__main__':
    main()
