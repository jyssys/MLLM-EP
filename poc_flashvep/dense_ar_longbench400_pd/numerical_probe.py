"""Keep full-model single-GPU chunk teachers independent of the partition adapter."""
import atexit
import time
from .common import ROOT,write
from .geometry import aligned_suffix


def install(cfg,role,info):
    if cfg.get('token_teacher_cut'):
        assert role=='single' and cfg['serial']
        from vllm.v1.core.sched.scheduler import Scheduler
        original=Scheduler.schedule;rows=[]
        def schedule(scheduler,*a,**kw):
            requests=list(scheduler.running)+list(scheduler.waiting)+list(scheduler.skipped_waiting)
            requests=[r for r in requests if r.request_id.split('-')[0] in info and r.num_computed_tokens<r.num_prompt_tokens]
            assert len(requests)<=1
            budget=scheduler.max_num_scheduled_tokens
            if requests:
                request=requests[0];suffix=aligned_suffix(request.num_prompt_tokens,cfg['token_teacher_cut'])
                if 'teacher_suffix_fraction' in cfg:suffix=max(16,int(round(request.num_prompt_tokens*cfg['teacher_suffix_fraction']/16))*16)
                external=request.request_id.split('-')[0]
                if external in cfg.get('suffix_by_request',{}):suffix=int(cfg['suffix_by_request'][external])
                prefix=request.num_prompt_tokens-suffix
                if request.num_computed_tokens<prefix:scheduler.max_num_scheduled_tokens=min(budget,prefix-request.num_computed_tokens)
                rows.append(dict(request_id=request.request_id,computed=request.num_computed_tokens,prefix=prefix,step_budget=scheduler.max_num_scheduled_tokens,upper_B=budget))
            try:return original(scheduler,*a,**kw)
            finally:scheduler.max_num_scheduled_tokens=budget
        Scheduler.schedule=schedule
        atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'same_token_geometry_teacher.json',dict(rows=rows,
            evidence='CORRECTNESS_ONLY',scope='Unmodified original full36 model. Only the per-step scheduler budget is clipped to the legal prefix boundary. '
                'No model or partition adapter is installed. No transfer occurs. No performance claim.')))
    from ..dense_ar_bi0_shortanswer_pd.numerical_probe import install as original_probe
    options=dict(cfg);options.pop('token_teacher_cut',None);original_probe(options,role,info)
