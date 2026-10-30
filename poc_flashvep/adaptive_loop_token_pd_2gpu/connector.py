"""Exact token-prefix migration and admission-checked snapshot loop migration."""
import os
import time
import torch
from vllm_rlt.request import Request,RequestPhase
from ..adaptive_recurrence_cut.connector import CutConnector
from ..adaptive_recurrence_cut.common import load_config
from .common import ROOT,append

class PartitionConnector(CutConnector):
    def command(self,command):
        kind=command['kind']
        if kind in ('reserve','prefill'):
            self._next_prefix=command.get('prefix_k',len(command['tokens']))
        if kind=='frontier_decision':
            t=self.transfers[command['tid']]
            if not getattr(t,'awaiting_decision',False):raise RuntimeError('Unexpected frontier decision')
            t.awaiting_decision=False
            if command['accept']:
                self.export_snapshot(t)
            else:
                # No remote writes occurred. All existing state remains P-owned.
                if self.cfg['axis']=='loop':
                    t.cut=self.cache.max_loops;t.request.arc_export_cut=t.cut
                else:
                    t.prefix=len(t.request.prompt_token_ids);t.request.ltp_prefix=t.prefix
                    scheduler=self.engine.scheduler
                    for queue in scheduler.queues.values():
                        while t.request.request_id in queue:queue.remove(t.request.request_id)
                    scheduler.enqueue(t.request,RequestPhase.PREFILLING_PRELUDE)
                t.fallback=True
                append(ROOT/'raw'/os.environ['ARC_RUN']/f'fallback_{os.getpid()}.jsonl',
                    dict(tid=t.tid,seed=t.request.sampling_params.seed,time=time.monotonic(),
                         action='CONTINUE_P_TO_FULL',reason=command['reason']))
            return True
        if kind=='commit':
            # Final plan is authoritative. No D reads or source reuse precede commit.
            t=self.transfers[command['tid']]
            t.cut=command.get('arc_cut',t.cut)
            t.prefix=command.get('prefix_k',t.prefix)
        return super().command(command)

    def reserve(self,tid,tokens,*args,**kwargs):
        super().reserve(tid,tokens,*args,**kwargs)
        if tid not in self.transfers:return
        t=self.transfers[tid];t.prefix=self._next_prefix
        if not 0<=t.prefix<=len(tokens):raise ValueError('Illegal token prefix')
        if t.outgoing and self.cfg['axis']=='token':
            if not t.prefix:raise ValueError('Zero token prefix must use D-local endpoint')
            t.request.ltp_prefix=t.prefix

    def can_schedule(self,request):
        t=self.transfers.get(request.request_id)
        if t and getattr(t,'awaiting_decision',False):return False
        return super().can_schedule(request)

    def prefill_submitted(self,request,*,phase,start,progress,event,stream):
        t=self.transfers[request.request_id]
        if t.compute_done or getattr(t,'awaiting_decision',False):return
        t.last_event=event
        if self.cfg['axis']=='token':
            ready=request.num_prefilled_tokens>=t.prefix
            if request.num_preluded_tokens>t.prefix:raise RuntimeError('P crossed token-prefix frontier')
        else:
            ready=request.prefill_depth_ranges[t.cut-1][0]>=len(request.prompt_token_ids)
            if any(left for left,right in request.prefill_depth_ranges[t.cut:]):
                raise RuntimeError('P crossed recurrent-cut frontier')
        if not ready:return
        total=len(request.prompt_token_ids)
        partial=(t.prefix<total if self.cfg['axis']=='token' else t.cut<self.cache.max_loops)
        if self.cfg.get('dynamic') and partial and not getattr(t,'fallback',False):
            t.awaiting_decision=True
            self.notify('frontier_ready',tid=t.tid,arc_cut=t.cut,prefix_k=t.prefix)
        else:self.export_snapshot(t)

    def export_snapshot(self,t):
        request=t.request;n=len(request.prompt_token_ids)
        runner=self.engine.model_runner;slot=runner._ensure_request_slot(request)
        stream=runner.core_stream or torch.cuda.current_stream(self.cache.device)
        with torch.cuda.stream(stream):
            if t.last_event is not None:stream.wait_event(t.last_event)
            if self.cfg['axis']=='token':
                state=request.hidden_state if t.prefix==n else runner.prefill_states[slot,t.prefix-1]
            else:
                state=request.hidden_state if t.cut==self.cache.max_loops else runner.prefill_states[slot,:n]
            self.hidden[t.slot,:state.numel()].copy_(state.flatten())
            if self.cfg['axis']=='token' and t.prefix<n and self.cfg['model']=='huginn':
                # The qualified Huginn path initializes random states for the entire
                # prompt once. A suffix must consume those same unscaled states.
                generator=runner.rng.peek(request.request_id)
                initial=getattr(self.engine.model,'abc_prompt_initial_states',{}).get(id(generator))
                if initial is None:raise RuntimeError('Missing canonical Huginn suffix initialization state')
                remaining=initial[t.prefix:n].flatten()
                begin=self.engine.model.config.hidden_size
                self.hidden[t.slot,begin:begin+remaining.numel()].copy_(remaining)
            event=torch.cuda.Event();event.record(stream)
        g=runner.rng.peek(request.request_id)
        if g is not None:t.rng_state=g.get_state().tolist()
        t.last_event=event
        self.queue_snapshot(t,n,event)
        self.engine.suspend_transfer(request);t.compute_done=True
        t.transfer_boundary_first=t.transfer_boundary_last=(0,t.prefix if self.cfg['axis']=='token' else n)
        t.transfer_boundary_count=1;self.prefill_tokens+=t.prefix if self.cfg['axis']=='token' else n

    def layout(self,t):
        if self.cfg['axis']!='token':yield from super().layout(t);return
        n=len(t.request.prompt_token_ids)
        recurrent=set(getattr(self.engine.model,'recurrent_kv_layers',range(self.cache.num_layers)))
        coda=set(getattr(self.engine.model,'coda_kv_layers',()))
        for depth in range(self.cache.max_loops):
            for layer in range(self.cache.num_layers):
                if depth and layer not in recurrent:continue
                end=min(t.prefix,n-1) if layer in coda else t.prefix
                yield depth,layer,end

    def hidden_elements(self,t):
        if self.cfg['axis']=='token':
            width=self.engine.model.config.hidden_size
            if self.cfg['model']=='huginn' and t.prefix<len(t.request.prompt_token_ids):
                width+=(len(t.request.prompt_token_ids)-t.prefix)*self.engine.model.config.n_embd
            return width
        return super().hidden_elements(t)

    def notify_plan(self):
        original=self.notify
        def notify(kind,**fields):
            if kind=='commit':fields['prefix_k']=self.transfers[fields['tid']].prefix
            return original(kind,**fields)
        self.notify=notify

    @classmethod
    def create(cls,*args,**kwargs):
        self=super().create(*args,**kwargs);self.notify_plan();return self

    def accept_cut(self,request,hidden,*,cached_tokens=0):
        if self.cfg['axis']!='token':return super().accept_cut(request,hidden,cached_tokens=cached_tokens)
        if cached_tokens:raise ValueError('Prefix cache reuse is outside qualification')
        t=self.transfers[request.request_id];k=t.prefix;n=len(request.prompt_token_ids)
        allocation=self.cache._get_allocation(request.request_id)
        if k==n:
            accepted=self.original_accept(request,hidden[:self.engine.model.config.hidden_size],cached_tokens=0)
        else:
            for depth,layer,end in self.layout(t):allocation.written[depth][layer].prefix=end
        if self.cfg.get('verify_transfer'):
            actual=self.fingerprint(t)
            row=dict(seed=request.sampling_params.seed,cut=t.cut,prefix_k=k,actual=actual,
                     expected=t.expected_fingerprint,passed=actual==t.expected_fingerprint)
            append(ROOT/'raw'/os.environ['ARC_RUN']/f'KV_checks_{os.getpid()}.jsonl',row)
            if not row['passed']:raise RuntimeError('Token KV or hidden transport identity failed')
        if k==n:return accepted
        if self.cfg['model']=='huginn':
            runner=self.engine.model_runner;model=self.engine.model
            generator=runner.rng.acquire(request.request_id,request.sampling_params,self.cache.device)
            width=model.config.hidden_size;embd=model.config.n_embd
            initial=hidden.new_zeros((n,embd))
            initial[k:].copy_(hidden[width:width+(n-k)*embd].view(n-k,embd))
            ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(self.cache.device))
            for stream in (runner.core_stream,runner.boundary_stream):
                if stream is not None:stream.wait_event(ready)
            states=getattr(model,'abc_prompt_initial_states',{})
            model.abc_prompt_initial_states=states;states[id(generator)]=initial
            initialized=getattr(runner,'abc_huginn_initialized',set())
            runner.abc_huginn_initialized=initialized;initialized.add(request.request_id)
        request.num_prefilled_tokens=request.num_preluded_tokens=k
        request.prefill_window_origin=request.prefill_reclaimable_tokens=k
        request.prefill_depth_ranges[:]=Request.prefill_boundaries(n,self.cache.max_loops,start=k)
        request.loops_done=0
        request.import_resume_phase=RequestPhase.PREFILLING_PRELUDE
        self.engine.scheduler.queues[RequestPhase.WAITING].append(request.request_id)
        return True

def install_token_barrier():
    from vllm_rlt.core.scheduler import Scheduler
    original=Scheduler._make_wavefront_task
    window=Scheduler._prelude_window_open
    def prefix_window(self,request):
        return request.num_preluded_tokens<getattr(request,'ltp_prefix',len(request.prompt_token_ids)) and window(self,request)
    Scheduler._prelude_window_open=prefix_window
    def task(self,request,phase,token_budget,*,start,depth=None):
        prefix=getattr(request,'ltp_prefix',len(request.prompt_token_ids))
        if start>=prefix:return None
        return original(self,request,phase,min(token_budget,prefix-start),start=start,depth=depth)
    Scheduler._make_wavefront_task=task
