"""Causal budget admission. Keep native dependencies and bound starvation."""
import time

PREFILL=('prefill','prefilling_prelude','prefilling_recurrent')
DECODE=('prelude','recurrent','coda')

class BudgetAdmission:
    def __init__(self,engine,actions,selector,clock=time.perf_counter):
        self.engine=engine;self.actions=actions;self.selector=selector;self.clock=clock
        self.previous_ready=engine.scheduler.execution_ready
        self.previous_take=engine.scheduler.take_phases
        self.waiting={};self.trace=[]
        self.checks=0;self.total_ns=0;self.max_ns=0
        engine.scheduler.execution_ready=self.ready
        engine.scheduler.take_phases=self.take

    def ready(self,request):
        before=time.perf_counter_ns()
        try:
            return self._ready(request)
        finally:
            elapsed=time.perf_counter_ns()-before
            self.checks+=1;self.total_ns+=elapsed;self.max_ns=max(self.max_ns,elapsed)

    def _ready(self,request):
        if self.previous_ready is not None and not self.previous_ready(request):
            return False
        if request.phase.value not in PREFILL:
            return True
        scheduler=self.engine.scheduler
        if request.phase.value=='prefilling_prelude' and not scheduler._prelude_window_open(request):
            return False
        template,state=self.actions.snapshot(request,request.phase,scheduler.config.max_num_batched_tokens)
        if state is None or template.token_count<=0:
            return False
        key=id(request);now=self.clock()
        # This is a new wait at each legal frontier, not a lifetime age.
        frontier=(request.phase.value,state['start'],state['depth'])
        previous=self.waiting.get(key)
        if previous is None or previous[0]!=frontier:
            self.waiting[key]=(frontier,now)
        choices=self.selector.choices(state)
        if choices:
            return True
        has_decode=any(r.phase.value in DECODE for r in scheduler.requests.values())
        expired=1000*(now-self.waiting[key][1])>=self.selector.starvation_ms
        if not has_decode or expired:
            self.trace.append(dict(host_s=now,id=request.request_id,state=state,budget_ms=self.selector.budget,
                reason='OVERBUDGET_IDLE_PROGRESS' if not has_decode else 'OVERBUDGET_BOUNDED_STARVATION_ESCAPE'))
            return True
        return False

    def take(self,phases,*,eligible=None):
        # The model estimates one prompt span. Do not silently pack additional
        # prefills into that budget. All adaptive one-dimensional controls use
        # the same packing rule. Native Mixed is a separate unmodified baseline.
        base=eligible if eligible is not None else self.ready
        selected=False
        def one_prompt(request):
            nonlocal selected
            if not base(request):
                return False
            if request.phase.value in PREFILL:
                if selected:
                    return False
                selected=True
            return True
        return self.previous_take(phases,eligible=one_prompt)

    def close(self):
        self.engine.scheduler.execution_ready=self.previous_ready
        self.engine.scheduler.take_phases=self.previous_take
        self.engine=None;self.actions=None;self.previous_take=None;self.waiting.clear()
