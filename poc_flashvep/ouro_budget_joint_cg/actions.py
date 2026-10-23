"""Opt-in per-request immutable chunk spans. Native GPU ordering is unchanged."""
from dataclasses import dataclass,replace
import time

@dataclass(frozen=True)
class Choice:
    C:int
    G:int
    reason:str='registered fixed choice'

class ChunkActions:
    def __init__(self,engine,selector):
        self.engine=engine;self.selector=selector;self.spans={};self.decisions=[]
        self.original=engine.scheduler._make_scheduled_request
        engine.scheduler._make_scheduled_request=self.make

    def snapshot(self,request,phase,token_budget):
        template=self.original(request,phase,token_budget)
        name=phase.value
        if name not in ('prefill','prefilling_prelude','prefilling_recurrent'):
            return template,None
        scheduler=self.engine.scheduler
        if name=='prefilling_recurrent':
            key=(id(request),request.prefill_chunk_start)
            assert key in self.spans,'A continuation needs its own previously staged span'
            begin,end=self.spans[key]
            assert request.num_preluded_tokens==end and template.token_start==begin
            if token_budget<end-begin:
                return replace(template,token_count=0),None
            state=dict(kind='CONTINUE_CHUNK',id=request.request_id,C=end-begin,depth=template.depth,
                start=begin,prompt_remaining=len(request.prompt_token_ids)-begin,token_budget=token_budget)
        else:
            if name=='prefilling_prelude':
                assert scheduler._prelude_window_open(request),'Cannot resize an unfinished staged chunk'
            state=dict(kind='NEW_CHUNK',id=request.request_id,C=0,depth=0,start=template.token_start,
                prompt_remaining=len(request.prompt_token_ids)-template.token_start,token_budget=token_budget)
        state['context']=state['start']+max(1,state['C'])
        state['atomic']=name=='prefill'
        state['phase']=name
        state['issued_decode']=[dict(id=r.request_id,phase=r.phase.value,R=r.loops_done,outputs=len(r.generated_token_ids))
            for r in scheduler.requests.values() if r.phase.value in ('prelude','recurrent','coda')]
        runner=self.engine.model_runner
        state['request_event_ready']=runner.events.get(request.request_id) is None or runner.events[request.request_id].query()
        state['graph_entries']=len(runner.graphs.entries) if runner.graphs else 0
        return template,state

    def make(self,request,phase,token_budget):
        template,state=self.snapshot(request,phase,token_budget)
        if state is None:
            return template
        name=phase.value
        started=time.perf_counter_ns();choice=self.selector(state);elapsed=time.perf_counter_ns()-started
        assert choice.C>0 and choice.G in (1,2,3,4)
        if state['kind']=='NEW_CHUNK':
            count=min(template.token_count,choice.C)
            if name=='prefilling_prelude':
                self.spans[(id(request),template.token_start)]=(template.token_start,template.token_start+count)
            else:
                assert choice.G==4,'Atomic prefill always performs R4'
            result=replace(template,token_count=count)
        else:
            assert choice.C==state['C'],'Cannot resize a staged token span'
            group=min(choice.G,4-template.depth)
            result=replace(template,token_count=state['C'],loop_count=group,
                completes_prompt=template.depth+group==4 and template.token_start+state['C']==len(request.prompt_token_ids))
        self.decisions.append(dict(state=state,chosen_C=result.token_count,
            chosen_G=result.loop_count if name=='prefilling_recurrent' else 4 if name=='prefill' else 0,planned_G=choice.G,
            reason=choice.reason,selector_ns=elapsed,host_s=time.perf_counter(),phase=name))
        return result

    def close(self):
        self.engine.scheduler._make_scheduled_request=self.original
        self.engine=None;self.original=None;self.spans.clear()

def fixed(C,G):
    return lambda state:Choice(state['C'] if state['kind']=='CONTINUE_CHUNK' else C,G)

def scripted(state):
    # Correctness fixture only. No measured result follows from this schedule.
    if state['kind']=='CONTINUE_CHUNK':
        return Choice(state['C'],2 if state['start']==0 else 4,'scripted legal continuation')
    return Choice(512 if state['start']==0 else 1024,2 if state['start']==0 else 4,'scripted C512/G2 then NEW C1024/G4')

def cycling(state):
    # Covers G3 from depth one and G2 from depth one as legal remainders.
    start=state['start'];depth=state['depth']
    if start==0:
        C,G=128,(1 if depth==0 else 3)
    elif start<384:
        C,G=256,2
    elif start<896:
        C,G=512,(3 if depth==0 else 1)
    elif start<1664:
        C,G=768,(1 if depth in (0,3) else 2)
    else:
        C,G=128,4
    return Choice(state['C'] if state['kind']=='CONTINUE_CHUNK' else C,G,'registered legal frontier cycle')
