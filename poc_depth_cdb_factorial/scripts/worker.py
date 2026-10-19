"""Drive either unchanged native loop. Change placement outside the local engine."""
import argparse
import collections
import dataclasses
import json
import os
import queue
import sys
import threading
import time
import torch
from poc_decode_first_depth_aware_dp.runtime import load_engine, stats_dict
from poc_depth_cdb_factorial.contract import *

def emit(value):
    print('DP_EVENT '+json.dumps(value), flush=True)

class Admission:
    def __init__(self,engine,messages,tokenizer,origin,quality,trace=False):
        self.engine,self.messages,self.tokenizer,self.origin=engine,messages,tokenizer,origin
        self.quality=quality
        self.trace=trace
        self.closed=False
        self.next_status=0.
        self.metadata={}
        self.seen=[]
        self.timeline=[]

    @property
    def pending(self):
        return not self.closed or not self.messages.empty()

    def release_due(self):
        from looped_cdb.continuous_depth_batching.requests import RequestState
        result=[]
        while True:
            try:m=self.messages.get_nowait()
            except queue.Empty:break
            if m['op']=='close':self.closed=True;continue
            assert m['op']=='request',m
            row=m['row']
            ids=self.tokenizer.encode(row['prompt'],add_special_tokens=False)
            assert len(ids)+m['cap']<=self.engine.cdb_config.max_model_len
            stops=[self.tokenizer.encode(s,add_special_tokens=False) for s in row.get('stop_strings',[])] if self.quality else []
            state=RequestState(request_id=row['id'],initial_tokens=ids,max_new_tokens=m['cap'],
                eos_token_id=self.tokenizer.eos_token_id,stop_sequences=stops)
            self.metadata[row['id']]=dict(row=row,arrival=m['arrival_s'],admission=time.perf_counter()-self.origin,
                cap=m['cap'],stops=stops)
            self.seen.append(row['id'])
            result.append(state)
        now=time.perf_counter()
        if now>=self.next_status:
            s=self.engine.scheduler
            row=dict(event='queue',waiting=len(s.waiting_requests),active=len(s.active_requests),
                ready=len(s.ready_queue),coda=len(s.coda_queue),t=now-self.origin)
            if self.trace:
                row['free_KV_blocks']=self.engine.cache.get_num_free_blocks()
                row['used_KV_blocks']=self.engine.cache.num_blocks-row['free_KV_blocks']
                row['min_free_slots']=s.min_free_slots
                row['free_resident_slots']=s.free_residency_slots()
                row['admissible_fresh_requests']=sum(state.request_id in s.waiting_requests for state in s.get_prefill_bucket())
                row['reservation_blocks_admission']=bool(row['waiting'] and row['free_resident_slots']>0
                    and row['admissible_fresh_requests']==0 and not s.block_new_requests)
            self.timeline.append(row)
            emit(row)
            self.next_status=now+.05
        return result

    def wait_for_next_arrival(self):
        time.sleep(.001)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--trace',action='store_true')
    p.add_argument('--validate',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model,tok,engine=load_engine(graphs=not args.validate,batch=args.batch,blocks=2048,max_model_len=10240)
    # Both modes share warmed native graph buckets and weights.
    for refill in [False,True]:
        engine.cdb_config.refill=refill
        for n in [1,2,4,8,16]:
            if n<=args.batch:
                engine.generate_batch([tok.encode('Q: What is 1 + 1?\nA:')]*n,
                    max_new_tokens=4,eos_token_id=tok.eos_token_id,warmup=True)
    messages=queue.Queue()
    def reader():
        for line in sys.stdin:messages.put(json.loads(line))
    threading.Thread(target=reader,daemon=True).start()
    emit(dict(event='ready',gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),uuid=str(torch.cuda.get_device_properties(0).uuid)))
    original_record=engine._record_exit
    original_retire=engine._retire_request
    original_advance=engine._advance_after_coda
    original_prefill=engine._advance_after_prefill
    methods=['compute_recurrent_batch','stage_prefill_batch','stage_coda_batch','compute_prelude_batch']
    originals={n:getattr(engine.runner,n) for n in methods}
    original_copy=engine.cache.copy_kv_rows
    original_flush=engine._flush_exit_kv_copies
    while True:
        command=messages.get()
        if command['op']=='shutdown':break
        assert command['op']=='case'
        case=command['case']
        engine.reset()
        engine.cdb_config.refill=ARMS[command['arm']]['refill']
        depths={}
        token_times={}
        stages=[]
        hist=collections.Counter()
        checks=[]
        layer_calls=collections.Counter()
        handles=[]
        origin=command['origin_s']
        admission=Admission(engine,messages,tok,origin,command['quality'],args.trace)
        anchor=None
        anchor_host=None
        anchor_uncertainty=None
        if args.trace:
            anchor=torch.cuda.Event(enable_timing=True)
            left=time.perf_counter()
            anchor.record(engine.inputs_and_outputs.compute_stream)
            anchor.synchronize()
            right=time.perf_counter()
            anchor_host=(left+right)/2-origin
            anchor_uncertainty=(right-left)*1000
        def record(item,depth):
            depths.setdefault(item.state.request_id,[]).append(dict(position=item.token_position,depth=depth+1))
            return original_record(item,depth)
        def advance(item,token):
            token_times.setdefault(item.state.request_id,[]).append(time.perf_counter()-origin)
            return original_advance(item,token)
        def prefill(state,token):
            token_times.setdefault(state.request_id,[]).append(time.perf_counter()-origin)
            return original_prefill(state,token)
        def retire(state):
            result=original_retire(state)
            emit(dict(event='done',id=state.request_id,t=time.perf_counter()-origin))
            return result
        engine._record_exit=record
        engine._retire_request=retire
        engine._advance_after_coda=advance
        engine._advance_after_prefill=prefill
        def copied(source,targets,indices):
            original_copy(source,targets,indices)
            if args.validate:
                torch.cuda.synchronize()
                valid=all(torch.equal(engine.cache.key_cache[source].index_select(0,indices),engine.cache.key_cache[t].index_select(0,indices))
                    and torch.equal(engine.cache.value_cache[source].index_select(0,indices),engine.cache.value_cache[t].index_select(0,indices)) for t in targets)
                assert valid
                checks.append(dict(source=source,targets=targets,rows=indices.numel(),exact=True))
        engine.cache.copy_kv_rows=copied
        if args.trace:
            def flush_trace():
                pending=len(engine._exit_kv_backlog)
                if not pending:return original_flush()
                local=engine.scheduler
                occupancy=dict(resident=len(local.active_requests),waiting=len(local.waiting_requests),
                    ready=len(local.ready_queue),coda=len(local.coda_queue))
                b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter()
                b.record(engine.inputs_and_outputs.compute_stream)
                result=original_flush()
                e.record(engine.inputs_and_outputs.compute_stream)
                stages.append(('exit_KV_copy',start-origin,(time.perf_counter()-start)*1000,b,e,pending,[],occupancy))
                return result
            engine._flush_exit_kv_copies=flush_trace
        else:engine._flush_exit_kv_copies=original_flush
        if args.validate:
            handles=[l.register_forward_pre_hook(lambda module,a:layer_calls.update(['layer'])) for l in model.model.layers]
        for name,call in originals.items():
            def wrapped(*a,_call=call,_name=name,**kw):
                items=a[0] if _name!='stage_prefill_batch' else a[1]
                hist[f'{_name}:{len(items)}']+=1
                if not args.trace:return _call(*a,**kw)
                metadata=[]
                if _name=='compute_recurrent_batch':
                    metadata=[dict(id=i.state.request_id,position=i.token_position,depth=i.recurrent_step+1) for i in items]
                local=engine.scheduler
                occupancy=dict(resident=len(local.active_requests),waiting=len(local.waiting_requests),
                    ready=len(local.ready_queue),coda=len(local.coda_queue))
                if _name=='compute_recurrent_batch':
                    occupancy['core_padded_rows']=engine.runner._recurrent_bucket_size(len(items))
                stream=engine.inputs_and_outputs.compute_stream
                b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter()
                b.record(stream)
                value=_call(*a,**kw)
                e.record(stream)
                stages.append((_name,start-origin,(time.perf_counter()-start)*1000,b,e,len(items),metadata,occupancy))
                return value
            setattr(engine.runner,name,wrapped)
        torch.cuda.reset_peak_memory_stats()
        emit(dict(event='case_ready',case=case))
        if engine.cdb_config.refill:
            engine._run_scheduler_loop(model_kwargs=None,arrivals=admission)
        else:
            engine._run_no_refill_loop(model_kwargs=None,arrivals=admission)
        engine.scheduler.close_steady_window()
        engine._free_finished_or_active()
        torch.cuda.synchronize()
        end=time.perf_counter()
        for h in handles:h.remove()
        outputs=[]
        for rid in admission.seen:
            value=engine.finished_outputs[rid]
            meta=admission.metadata[rid]
            events=depths.get(rid,[])
            tokens=value.generated_tokens
            assert len(events)==len(tokens)-1,(rid,len(events),len(tokens))
            reason='eos' if tokens[-1]==tok.eos_token_id else 'length' if len(tokens)>=meta['cap'] else 'stop'
            outputs.append(dict(id=rid,output_ids=tokens,text=tok.decode(tokens,skip_special_tokens=False),depths=events,
                prompt_tokens=len(value.prompt_ids),generated_tokens=len(tokens),finish_reason=reason,
                token_host_times_s=token_times.get(rid,[]),arrival_s=meta['arrival'],admission_s=meta['admission'],
                first_scheduled_s=value.lifespan[0]-origin,first_token_s=value.first_token_time-origin,
                finish_s=value.lifespan[1]-origin,gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])))
        stats=stats_dict(engine.last_stats)
        stats['generated_tokens']=sum(r['generated_tokens'] for r in outputs)
        assert sum(x['depth'] for r in outputs for x in r['depths'])==stats['recurrent_steps']
        if not engine.cdb_config.refill:assert stats['scheduler_refills']==0
        if args.validate:
            assert layer_calls['layer']==24*(4*stats['prefill_batches']+stats['recurrent_batches'])
        timings=[dict(stage=n,host_start_s=s,host_ms=h,gpu_ms=b.elapsed_time(e),batch=batch,items=items,occupancy=occupancy,
            gpu_start_s=anchor_host+anchor.elapsed_time(b)/1000,gpu_end_s=anchor_host+anchor.elapsed_time(e)/1000)
            for n,s,h,b,e,batch,items,occupancy in stages]
        write(args.output/f'{case}.json',dict(outputs=outputs,stats=stats,stages=timings,
            wall_ms=(end-origin)*1000,trace=args.trace,validate=args.validate,checks=checks,
            layer_calls=layer_calls['layer'],batch_histogram=dict(hist),queue_timeline=admission.timeline,
            GPU_clock_anchor_host_s=anchor_host,GPU_anchor_uncertainty_ms=anchor_uncertainty,
            label='MEASURED_ONLINE_LIVE',arm=command['arm'],refill=engine.cdb_config.refill,
            memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved(),
            peak_memory=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            cache_bytes=sum(t.numel()*t.element_size() for t in engine.cache.key_cache+engine.cache.value_cache),
            config=dataclasses.asdict(engine.cdb_config),homogeneous_slots=engine.cache.requires_slot_homogeneous_recurrent_batches))
        emit(dict(event='case_done',case=case,requests=len(outputs)))
    emit(dict(event='shutdown'))

if __name__=='__main__':main()
