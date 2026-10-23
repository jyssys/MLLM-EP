"""Opt-in fixed-budget controllers. Native weights and GPU kernels stay intact."""
import time
from .actions import ChunkActions
from .admission import BudgetAdmission
from .budget_controller import BudgetController
from .optimizer import BudgetSelector

class OnlineControl:
    def __init__(self,engine,model,mode,budget_ms,fixed_C=1024,objective='U2',
            yield_penalty_ms=0.,feedback='FIXED',target_ITL_ms=270.):
        self.engine=engine
        self.controller=BudgetController(feedback,budget_ms,target_ITL_ms)
        self.selector=BudgetSelector(model,budget_ms,mode,fixed_C,objective,yield_penalty_ms)
        self.actions=ChunkActions(engine,self.selector)
        self.admission=BudgetAdmission(engine,self.actions,self.selector)
        self.original_step=engine.step
        engine.step=self.step
        self.output_events=[];self.counts={}

    def step(self,*args,**kwargs):
        outputs=self.original_step(*args,**kwargs)
        scheduler=self.engine.scheduler
        waiting=any(r.phase.value.startswith('prefill') for r in scheduler.requests.values())
        pressure=sum(r.phase.value in ('prelude','recurrent','coda') for r in scheduler.requests.values())
        for output in outputs:
            count=len(output.token_ids);previous=self.counts.get(output.request_id,0)
            if count>previous:
                assert count==previous+1,'Native output cadence must not fabricate timestamps'
                now=time.perf_counter()
                value=self.controller.observe_output(output.request_id,now,waiting,pressure)
                self.selector.budget=value
                self.output_events.append(dict(host_s=now,id=output.request_id,tokens=count,budget_ms=value,
                    source='NATIVE_STEP_OUTPUT_DELIVERY_NOT_HTTP'))
                self.counts[output.request_id]=count
            if output.finished:
                self.counts.pop(output.request_id,None)
                self.controller.last.pop(output.request_id,None)
        return outputs

    def reset_cohort(self):
        assert not self.engine.has_unfinished_requests()
        self.counts.clear();self.controller.last.clear()
        self.selector.trace.clear();self.actions.decisions.clear();self.admission.trace.clear()
        self.output_events.clear()

    def close(self):
        self.engine.step=self.original_step
        self.admission.close();self.actions.close()
        self.engine=None;self.original_step=None
