"""Non-synchronizing operation spans. CUDA spans are not kernel-busy time."""
import collections
import time

ACTIVE=None

class Recorder:
    def __init__(self,enabled=True):
        self.enabled=enabled;self.label='setup';self.records=[];self.cpu=collections.defaultdict(lambda:[0,0.]);self.frontiers=[];self.seen=set()
    def phase(self,label):
        self.label=label;self.records=[];self.cpu=collections.defaultdict(lambda:[0,0.])
    def cpu_add(self,name,seconds):
        value=self.cpu[name];value[0]+=1;value[1]+=seconds
    def summary(self):
        # Call only after an existing phase boundary has drained device work.
        gpu=collections.defaultdict(lambda:[0,0.])
        for operation,start,end in self.records:
            value=gpu[operation];value[0]+=1;value[1]+=start.elapsed_time(end)
        return dict(label=self.label,CPU_calls={k:dict(count=v[0],wall_ms=1000*v[1]) for k,v in self.cpu.items()},
            CUDA_forward_spans={k:dict(count=v[0],duration_sum_ms=v[1]) for k,v in gpu.items()},
            scope='CUDA event spans include intra-forward launch gaps. They are not exclusive kernel busy or additive E2E components.')

def install():
    import torch
    from vllm_rlt.worker.executor import Executor,ModelRuntime
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import RequestPhase
    if getattr(Executor,'_bc_telemetry',False):return
    Executor._bc_telemetry=True
    forward=ModelRuntime.forward;submit=Executor.submit;schedule=Scheduler.schedule
    def measured_forward(self,batch):
        recorder=ACTIVE
        if recorder is None or not recorder.enabled:return forward(self,batch)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream());t=time.perf_counter()
        result=forward(self,batch)
        end.record(torch.cuda.current_stream());recorder.records.append((batch.operation.value,start,end))
        recorder.cpu_add('model_forward_submission',time.perf_counter()-t)
        return result
    def measured_submit(self,batch):
        recorder=ACTIVE
        if recorder is None:return submit(self,batch)
        now=time.monotonic()
        for item in batch.requests:
            request=item.request;phase=batch.phase_for(item)
            if phase in (RequestPhase.PREFILL,RequestPhase.PREFILLING_PRELUDE,RequestPhase.PREFILLING_RECURRENT):
                key=(request.request_id,'P_FIRST_SUBMISSION')
                if key not in recorder.seen:
                    recorder.seen.add(key);recorder.frontiers.append(dict(event=key[1],request_id=key[0],seed=request.sampling_params.seed,time=now,label=recorder.label))
        t=time.perf_counter();result=submit(self,batch)
        recorder.cpu_add('executor_submit',time.perf_counter()-t)
        return result
    def measured_schedule(self,*args,**kwargs):
        t=time.perf_counter();result=schedule(self,*args,**kwargs)
        if ACTIVE is not None:ACTIVE.cpu_add('scheduler',time.perf_counter()-t)
        return result
    ModelRuntime.forward=measured_forward;Executor.submit=measured_submit;Scheduler.schedule=measured_schedule

