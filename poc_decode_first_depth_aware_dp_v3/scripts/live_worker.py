"""Thin DP admission interface around the authors' unchanged local CDB loop."""
import argparse
import dataclasses
import json
import os
import queue
import sys
import threading
import time
import torch
from poc_decode_first_depth_aware_dp.runtime import load_engine, stats_dict
from poc_decode_first_depth_aware_dp_v3.contract import *

def emit(value):
    print('DP_EVENT '+json.dumps(value),flush=True)

class Admission:
    """Replace only the authors' wall-clock ArrivalQueue input source."""
    def __init__(self,engine,messages,tokenizer,origin):
        self.engine=engine
        self.messages=messages
        self.tokenizer=tokenizer
        self.origin=origin
        self.closed=False
        self.next_status=0.0
        self.metadata={}
        self.seen=[]

    @property
    def pending(self):
        return not self.closed or not self.messages.empty()

    def release_due(self):
        from looped_cdb.continuous_depth_batching.requests import RequestState
        result=[]
        while True:
            try:message=self.messages.get_nowait()
            except queue.Empty:break
            if message['op']=='close':
                self.closed=True
                continue
            assert message['op']=='request'
            r=message['row']
            ids=self.tokenizer.encode(r['prompt'],add_special_tokens=False)
            assert len(ids)+message['cap']<=self.engine.cdb_config.max_model_len
            state=RequestState(request_id=r['id'],initial_tokens=ids,max_new_tokens=message['cap'],
                eos_token_id=self.tokenizer.eos_token_id)
            self.metadata[r['id']]=dict(row=r,arrival=message['arrival_s'],admission=time.perf_counter()-self.origin)
            self.seen.append(r['id'])
            result.append(state)
        now=time.perf_counter()
        if now>=self.next_status:
            s=self.engine.scheduler
            emit(dict(event='queue',waiting=len(s.waiting_requests),active=len(s.active_requests),
                ready=len(s.ready_queue),coda=len(s.coda_queue),t=now-self.origin))
            self.next_status=now+.05
        return result

    def wait_for_next_arrival(self):
        # This is host idle, not GPU work. The native engine calls it only after draining work.
        time.sleep(.001)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model,tok,engine=load_engine(graphs=True,batch=args.batch,blocks=2048,max_model_len=8192)
    # Warm all graph bucket sizes that can occur. These outputs are not evaluation requests.
    for n in [1,2,4,8,16]:
        if n<=args.batch:
            engine.generate_batch([tok.encode('Q: What is 1 + 1?\nA:')]*n,
                max_new_tokens=4,eos_token_id=tok.eos_token_id,warmup=True)
    calibration=[r for r in read(ROOT/'artifacts/discovery_frozen.json')['rows'] if r['split']=='calibration']
    engine.generate_batch([tok.encode(r['prompt'],add_special_tokens=False) for r in calibration],
        max_new_tokens=16,eos_token_id=tok.eos_token_id,warmup=False)
    messages=queue.Queue()
    def reader():
        for line in sys.stdin:
            messages.put(json.loads(line))
    threading.Thread(target=reader,daemon=True).start()
    emit(dict(event='ready',gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),uuid=str(torch.cuda.get_device_properties(0).uuid)))
    original_record=engine._record_exit
    original_retire=engine._retire_request
    original_advance=engine._advance_after_coda
    original_prefill=engine._advance_after_prefill
    stage_methods=['_run_mixed_recurrent_batch','_run_prefill_batch','_stage_coda_batch']
    original_stages={name:getattr(engine,name) for name in stage_methods}
    while True:
        command=messages.get()
        if command['op']=='shutdown':break
        assert command['op']=='case'
        case=command['case']
        engine.reset()
        depths={}
        token_times={}
        stages=[]
        origin=command['origin_s']
        admission=Admission(engine,messages,tok,origin)
        def record(item,depth):
            depths.setdefault(item.state.request_id,[]).append(dict(position=item.token_position,depth=depth+1,
                tick=engine.last_stats.scheduler_ticks))
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
        for name,call in original_stages.items():
            if not args.trace:
                setattr(engine,name,call)
                continue
            def timed(*a,_call=call,_name=name,**kw):
                items=a[0]
                metadata=[dict(id=i.state.request_id,position=i.token_position,depth=i.recurrent_step,
                    context=i.token_position+1) for i in items] if isinstance(items,list) else []
                if _name=='_run_prefill_batch':
                    metadata=[dict(id=i.state.request_id,query=i.query_length,
                        past=i.state.position_offset,final_chunk=i.has_new_token) for i in items.requests]
                begin=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter()
                begin.record(engine.inputs_and_outputs.compute_stream)
                value=_call(*a,**kw)
                end.record(engine.inputs_and_outputs.compute_stream)
                if _name=='_run_mixed_recurrent_batch':
                    for record in metadata:
                        previous=depths.get(record['id'],[])
                        record['exited']=bool(previous and previous[-1]['position']==record['position'])
                stages.append((_name,start-origin,(time.perf_counter()-start)*1000,begin,end,metadata))
                return value
            setattr(engine,name,timed)
        emit(dict(event='case_ready',case=case))
        # This is the native local scheduler. No recurrence, KV, gate, or refill code is copied.
        engine._run_scheduler_loop(model_kwargs=None,arrivals=admission)
        engine.scheduler.close_steady_window()
        engine._free_finished_or_active()
        torch.cuda.synchronize()
        end=time.perf_counter()
        outputs=[]
        for rid in admission.seen:
            value=engine.finished_outputs[rid]
            meta=admission.metadata[rid]
            times=token_times.get(rid,[])
            events=depths.get(rid,[])
            assert len(events)==len(value.generated_tokens)-1,(rid,len(events),len(value.generated_tokens))
            outputs.append(dict(id=rid,output_ids=value.generated_tokens,depths=events,prompt_tokens=len(value.prompt_ids),
                generated_tokens=len(value.generated_tokens),token_host_times_s=times,
                arrival_s=meta['arrival'],admission_s=meta['admission'],
                first_token_s=value.first_token_time-origin,finish_s=value.lifespan[1]-origin,
                gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])))
        timing=[dict(stage=n,host_start_s=s,host_ms=h,gpu_ms=b.elapsed_time(e),items=items)
                for n,s,h,b,e,items in stages]
        stats=stats_dict(engine.last_stats)
        assert sum(x['depth'] for r in outputs for x in r['depths'])==stats['recurrent_steps']
        write(args.output/f'{case}.json',dict(outputs=outputs,stats=stats,stages=timing,
            wall_ms=(end-origin)*1000,trace=args.trace,label='MEASURED_ONLINE_LIVE_DP_RANK',
            memory_allocated=torch.cuda.memory_allocated(),peak_memory=torch.cuda.max_memory_allocated(),
            homogeneous_slots=engine.cache.requires_slot_homogeneous_recurrent_batches))
        emit(dict(event='case_done',case=case,requests=len(outputs)))
    emit(dict(event='shutdown'))

if __name__=='__main__':main()
