"""Measure after native request waits. Never read unfinished GPU tensors."""
import time
import torch

class ServiceObserver:
    def __init__(self,engine,metadata=None,prefill_only=False):
        self.engine=engine;self.metadata=metadata or {};self.rows=[];self.events=[];self.active=None
        self.prefill_only=prefill_only
        runner=engine.model_runner
        runner.synchronize()
        self.anchor=torch.cuda.Event(enable_timing=True)
        before=time.perf_counter();self.anchor.record();self.anchor.synchronize();after=time.perf_counter()
        self.origin=(before+after)/2;self.uncertainty_s=(after-before)/2
        self.original_submit=engine.executor.submit
        self.original_prepare=runner.prepare_on_gpu
        self.original_execute=runner._execute
        engine.executor.submit=self.submit
        runner.prepare_on_gpu=self.prepare
        runner._execute=self.execute

    def submit(self,batch):
        if self.prefill_only and not any(batch.phase_for(r).value.startswith('prefill') for r in batch.requests):
            return self.original_submit(batch)
        runner=self.engine.model_runner
        now=time.perf_counter()
        ready={r.request.request_id:(runner.events[r.request.request_id].query() if r.request.request_id in runner.events else True) for r in batch.requests}
        self.active=dict(index=len(self.rows),host_dispatch_begin_s=now,ready_before_issue=ready,
            pending_event_ready=[event.query() for event in runner.submissions.events],
            spans=[dict(id=r.request.request_id,phase=batch.phase_for(r).value,start=r.token_start,
                tokens=r.token_count,depth=r.depth if r.depth is not None else 0,
                G=4 if batch.phase_for(r).value=='prefill' else r.loop_count if batch.phase_for(r).value in ('prefilling_recurrent','recurrent') else 0)
                for r in batch.requests],
            graph_before={k:getattr(runner.graphs,k,0) for k in ('captures','replays','fallbacks')} if runner.graphs else {},
            context_max=max(r.token_start+r.token_count if batch.phase_for(r).value in ('prefill','prefilling_prelude','prefilling_recurrent','recurrent') else r.request.position+1 for r in batch.requests),
            staged_bytes=runner.prefill_states.numel()*runner.prefill_states.element_size() if runner.prefill_states is not None else 0,
            actual_rows=sum(r.token_count for r in batch.requests),
            state=[dict(id=r.request_id,phase=r.phase.value,issued_R=r.loops_done,position=r.position,
                prompt_tokens=len(r.prompt_token_ids),num_prefilled=r.num_prefilled_tokens,num_preluded=r.num_preluded_tokens,
                observed_outputs=len(r.generated_token_ids)) for r in self.engine.scheduler.requests.values()])
        ticket=self.original_submit(batch)
        self.active['host_dispatch_end_s']=time.perf_counter()
        self.active['graph_after']={k:getattr(runner.graphs,k,0) for k in ('captures','replays','fallbacks')} if runner.graphs else {}
        self.rows.append(self.active);self.active=None
        return ticket

    def prepare(self,prepared):
        if self.active is None:
            return self.original_prepare(prepared)
        # Native submit already issued its request wait_event dependencies.
        start=torch.cuda.Event(enable_timing=True);start.record(torch.cuda.current_stream())
        self.active['host_gpu_prepare_s']=time.perf_counter()
        self.events.append([start,None])
        return self.original_prepare(prepared)

    def execute(self,*args,**kwargs):
        result=self.original_execute(*args,**kwargs)
        if self.active is None:
            return result
        end=torch.cuda.Event(enable_timing=True);end.record(torch.cuda.current_stream())
        self.events[-1][1]=end
        return result

    def finish(self):
        runner=self.engine.model_runner;runner.synchronize()
        final=torch.cuda.Event(enable_timing=True)
        before=time.perf_counter();final.record();final.synchronize();after=time.perf_counter()
        drift=1000*((before+after)/2-self.origin)-self.anchor.elapsed_time(final)
        bound=abs(drift)+1000*(self.uncertainty_s+(after-before)/2)
        assert len(self.rows)==len(self.events)
        for row,(start,end) in zip(self.rows,self.events):
            assert end is not None
            begin_s=self.origin+self.anchor.elapsed_time(start)/1000
            end_s=self.origin+self.anchor.elapsed_time(end)/1000
            row.update(self.metadata,service_ms=start.elapsed_time(end),aligned_service_begin_s=begin_s,
                aligned_service_end_s=end_s,host_dispatch_ms=1000*(row['host_dispatch_end_s']-row['host_dispatch_begin_s']),
                host_prepare_before_service_ms=1000*(row['host_gpu_prepare_s']-row['host_dispatch_begin_s']),
                enqueue_to_service_start_ms=1000*(begin_s-row['host_dispatch_begin_s']),alignment_bound_ms=bound,
                prompt_token_loop_work=sum(s['tokens']*s['G'] for s in row['spans'] if s['phase'] in ('prefill','prefilling_recurrent')),
                completed_prompt_tokens=sum(s['tokens'] for s in row['spans'] if s['phase']=='prefill' or (s['phase']=='prefilling_recurrent' and s['depth']+s['G']==4)),
                evidence='MEASURED_OBSERVER_READY_STREAM_SERVICE_NOT_PURE_KERNEL_BUSY')
        return self.rows

    def close(self):
        e=self.engine
        e.executor.submit=self.original_submit
        e.model_runner.prepare_on_gpu=self.original_prepare
        e.model_runner._execute=self.original_execute
        self.engine=None;self.original_submit=None;self.original_prepare=None;self.original_execute=None
