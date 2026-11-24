"""CUDA-event forward enclosures. These are not exclusive kernel activity."""
import csv
import os
import re
import time

from .common import ROOT,write


class GPUEnclosures:
    def __init__(self,runner,name,state_interval_ms=50):
        assert re.fullmatch('[A-Za-z0-9_.-]+',name) and 0<=state_interval_ms<=1000
        self.runner=runner;self.interval_ns=int(state_interval_ms*1_000_000)
        self.folder=ROOT/'raw/gpu_enclosures'/name/os.environ['MLLM_EPD_ROLE']
        assert not self.folder.exists();self.folder.mkdir(parents=True)
        self.calls=[];self.states=[];self.last_state=0

    def anchor(self):
        import torch
        event=torch.cuda.Event(enable_timing=True)
        left=time.monotonic_ns();event.record();event.synchronize();right=time.monotonic_ns()
        return event,dict(host_lower_ns=left,host_upper_ns=right,
                          uncertainty_ns=right-left)

    def begin(self):
        import torch
        from vllm.v1.core.sched.scheduler import Scheduler
        torch.cuda.synchronize()
        self.first,self.left=self.anchor()
        self.slot=('execute_model' in vars(self.runner),vars(self.runner).get('execute_model'))
        self.original_execute=self.runner.execute_model
        self.original_schedule=Scheduler.schedule
        original_execute=self.original_execute;original_schedule=self.original_schedule;me=self
        def execute(output,*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            host_begin=time.monotonic_ns();start.record()
            result=original_execute(output,*args,**kwargs)
            end.record()
            me.calls.append(dict(start=start,end=end,host_begin_ns=host_begin,
                host_end_ns=time.monotonic_ns(),requests=dict(output.num_scheduled_tokens),
                scheduled_tokens=output.total_num_scheduled_tokens,
                encoder_items=sum(len(v) for v in output.scheduled_encoder_inputs.values())))
            return result
        def schedule(scheduler,*args,**kwargs):
            output=original_schedule(scheduler,*args,**kwargs)
            now=time.monotonic_ns()
            if now-me.last_state>=me.interval_ns:
                me.last_state=now
                waiting=list(scheduler.waiting);running=list(scheduler.running)
                me.states.append(dict(time_ns=now,waiting=len(waiting),running=len(running),
                    decode_active=sum(r.num_computed_tokens>=r.num_prompt_tokens for r in running),
                    prefill_active=sum(r.num_computed_tokens<r.num_prompt_tokens for r in running),
                    waiting_by_status={str(status):sum(r.status==status for r in waiting)
                        for status in {r.status for r in waiting}}))
            return output
        self.runner.execute_model=execute;Scheduler.schedule=schedule
        return dict(folder=str(self.folder),anchor=self.left,
            scope='GPU forward stream enclosures only. No per-step synchronization, kernel trace or SM counter.')

    def close(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        Scheduler.schedule=self.original_schedule
        if self.slot[0]:self.runner.execute_model=self.slot[1]
        else:delattr(self.runner,'execute_model')
        last,right=self.anchor()
        span_ms=self.first.elapsed_time(last)
        left_mid=(self.left['host_lower_ns']+self.left['host_upper_ns'])/2
        right_mid=(right['host_lower_ns']+right['host_upper_ns'])/2
        scale=(right_mid-left_mid)/(span_ms*1_000_000) if span_ms>0 else 1.
        rows=[];geometry=[]
        for index,call in enumerate(self.calls):
            start_ms=self.first.elapsed_time(call['start']);end_ms=self.first.elapsed_time(call['end'])
            assert 0<=start_ms<=end_ms<=span_ms+.1
            rows.append(dict(index=index,begin_ns=left_mid+start_ms*1_000_000*scale,
                end_ns=left_mid+end_ms*1_000_000*scale,GPU_enclosure_ms=end_ms-start_ms,
                host_begin_ns=call['host_begin_ns'],host_end_ns=call['host_end_ns'],
                scheduled_requests=len(call['requests']),scheduled_tokens=call['scheduled_tokens'],
                encoder_items=call['encoder_items']))
            geometry.append(dict(index=index,requests=call['requests']))
        path=self.folder/'forward_enclosures.csv'
        with path.open('x',newline='') as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(self.folder/'scheduler.json',dict(rows=self.states,sampling_interval_ms=self.interval_ns/1_000_000))
        write(self.folder/'geometry.json',dict(rows=geometry))
        value=dict(events=len(rows),folder=str(self.folder),left=self.left,right=right,
            CUDA_anchor_span_ms=span_ms,clock_scale=scale,
            evidence='DIAGNOSTIC until observer qualification',
            scope='Forward stream enclosures include gaps and dependencies, not GPU-exclusive kernels. '
                'Copies on unrelated streams and export work outside execute_model are not covered. '
                'Uncovered time is not automatically physical idle or actionable slack. '
                'Host alignment uses two bounded CUDA-event anchors and linear clock scaling.')
        write(self.folder/'anchors.json',value)
        return value
