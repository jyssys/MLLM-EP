"""Force the same legal prefix/suffix boundary on one full-model GPU."""
import atexit
import time
from .common import ROOT,write
from .partial_axis import token_prefix


def clipped_budget(limit,computed,prefix):
    return min(limit,prefix-computed) if computed<prefix else limit


def install(cfg,role,info):
    if not cfg.get('token_teacher_cut'):return
    assert role=='single' and cfg['serial'] and cfg['execution_family']=='MATCHED_NO_COMPILE'
    from vllm.v1.core.sched.scheduler import Scheduler
    original=Scheduler.schedule;rows=[]
    def schedule(self,*args,**kwargs):
        requests=list(self.running)+list(self.waiting)+list(self.skipped_waiting)
        requests=[r for r in requests if r.request_id.split('-')[0] in info and r.num_computed_tokens<r.num_prompt_tokens]
        assert len(requests)<=1,'The same-geometry teacher is serial'
        limit=self.max_num_scheduled_tokens
        if requests:
            request=requests[0];prefix=token_prefix(request.num_prompt_tokens,cfg['token_teacher_cut'])
            self.max_num_scheduled_tokens=clipped_budget(limit,request.num_computed_tokens,prefix)
            rows.append(dict(request_id=request.request_id,computed=request.num_computed_tokens,prefix=prefix,
                step_token_limit=self.max_num_scheduled_tokens,scheduler_upper_limit=limit,time=time.monotonic()))
        try:return original(self,*args,**kwargs)
        finally:self.max_num_scheduled_tokens=limit
    Scheduler.schedule=schedule
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'same_token_geometry_teacher.json',dict(rows=rows,
        evidence='CORRECTNESS_ONLY',scope='One local request retains its KV and executes the same prompt query chunks as the two-GPU Token arm. '
        'The step token limit never exceeds B8192. This diagnostic changes chunk boundaries, not attention, weights or output tokens. No performance claim.')))
