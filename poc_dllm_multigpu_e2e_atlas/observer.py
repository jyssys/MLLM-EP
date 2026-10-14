"""All-occurrence outer intervals. No last-replay multiplication or hot sync."""
import contextlib
import functools
import time
import torch


class AtlasObserver:
    def __init__(self,runner,dllm,mode):
        self.runner=runner;self.dllm=dllm;self.mode=mode
        self.records=[];self.pending=[];self.updates=[];self.patches=[];self.stack=[]
        self.iteration=None;self.index=0;self.block_steps={};self.serial=0
        self.pool=[];self.origin=None;self.clock=None
        if mode in ['coarse','profile']:
            self.pool=[torch.cuda.Event(enable_timing=True) for _ in range(2048)]
            for event in self.pool:event.record()
            torch.cuda.synchronize()
            # A conservative host bracket for the rank-local GPU origin.
            before=time.perf_counter_ns()
            self.origin=torch.cuda.Event(enable_timing=True);self.origin.record();self.origin.synchronize()
            after=time.perf_counter_ns()
            self.clock=dict(host_before_ns=before,host_after_ns=after,
                uncertainty_us=(after-before)/1000,method='CUDA origin event completion lies inside this host bracket. No cross-rank event subtraction.')

    def patch(self,obj,name,fn):
        self.patches.append((obj,name,getattr(obj,name)));setattr(obj,name,fn)

    def event(self):
        assert self.index<len(self.pool),'Event queue exceeded frozen capacity'
        result=self.pool[self.index];self.index+=1;result.record();return result

    @contextlib.contextmanager
    def scope(self,kind,**metadata):
        ident=self.serial;self.serial+=1
        row=dict(id=ident,parent=self.stack[-1] if self.stack else None,kind=kind,
            iteration=None if self.iteration is None else self.iteration.copy(),**metadata)
        self.stack.append(ident)
        row['host_start_ns']=time.perf_counter_ns()
        if self.mode=='profile':torch.cuda.nvtx.range_push('ATLAS|'+kind+'|'+str(ident)+'|'+str(row['iteration']))
        begin=self.event()
        try:yield row
        finally:
            end=self.event()
            if self.mode=='profile':torch.cuda.nvtx.range_pop()
            row['host_end_ns']=time.perf_counter_ns()
            self.pending.append((row,begin,end));self.stack.pop()

    def timed(self,fn,kind,metadata=None):
        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            extra={} if metadata is None else metadata(args,kwargs)
            with self.scope(kind,**extra):return fn(*args,**kwargs)
        return wrapped

    def __enter__(self):
        from dinfer.decoding.utils import TokenArray
        original=TokenArray.__setitem__
        def update(obj,idx,values,_fn=original):
            result=_fn(obj,idx,values)
            if self.iteration is not None and isinstance(idx,tuple) and isinstance(idx[1],slice):
                self.updates.append((self.iteration.copy(),idx[1].start,idx[1].stop,values))
            return result
        self.patch(TokenArray,'__setitem__',update)
        fn=self.dllm.diff_iteration.forward
        def iteration(*args,_fn=fn,**kwargs):
            block=int(args[6]);step=self.block_steps.get(block,0);self.block_steps[block]=step+1
            loc=args[5]
            self.iteration=dict(block=block,step=step,cross=bool(kwargs.get('is_cross_block',False)),start=loc.start,end=loc.end)
            try:
                if self.mode=='parity':return _fn(*args,**kwargs)
                with self.scope('commit_refresh' if self.iteration['cross'] else 'refinement'):
                    return _fn(*args,**kwargs)
            finally:self.iteration=None
        self.patch(self.dllm.diff_iteration,'forward',iteration)
        if self.mode=='parity':return self
        if self.mode=='profile':
            # A distinct int16 fill anchors CUPTI time to CUDA-event time.
            # It runs before the measured request and does not change model state.
            with self.scope('profiler_clock_anchor'):
                self.anchor_tensor=torch.empty((1,),device=torch.cuda.current_device(),dtype=torch.int16)
                self.anchor_tensor.fill_(187)
        for name,kind in [('prefill','prefill'),('decode','block')]:
            self.patch(self.dllm.block_runner,name,self.timed(getattr(self.dllm.block_runner,name),kind))
        def model_meta(args,kw):
            ids=kw.get('input_ids',args[0] if args else None);kv=kw.get('past_key_values')
            if hasattr(kv,'_data'):kv=kv._data
            return dict(query_rows=ids.shape[1],physical_keys=0 if kv is None else kv.shape[4])
        self.patch(self.runner,'forward',self.timed(self.runner.forward,'model_forward',model_meta))
        self.patch(self.runner.graph_runner,'replay_prepare',self.timed(self.runner.graph_runner.replay_prepare,'graph_input_staging'))
        self.patch(torch.cuda.CUDAGraph,'replay',self.timed(torch.cuda.CUDAGraph.replay,'graph_replay'))
        self.patch(self.dllm.decoder,'decode',self.timed(self.dllm.decoder.decode,'sampling_acceptance'))
        from dinfer.decoding.utils import DiffusionKVCacheManager,BlockDiffusionPrefixCacheManager
        for obj,name in [(DiffusionKVCacheManager,'range_update'),(BlockDiffusionPrefixCacheManager,'extend_cache')]:
            self.patch(obj,name,self.timed(getattr(obj,name),'KV_'+name))
        self.patch(TokenArray,'get_generated_tokens',self.timed(TokenArray.get_generated_tokens,'output_finalization'))
        return self

    def __exit__(self,*unused):
        for obj,name,fn in reversed(self.patches):setattr(obj,name,fn)

    def finish(self):
        # Caller completed the request and synchronized once. Resolve all events now.
        spans=[]
        for row,a,b in self.pending:
            spans.append(dict(**row,GPU_start_ms=self.origin.elapsed_time(a),GPU_end_ms=self.origin.elapsed_time(b),
                GPU_interval_ms=a.elapsed_time(b),host_interval_ms=(row['host_end_ns']-row['host_start_ns'])/1e6))
        return dict(spans=sorted(spans,key=lambda r:r['id']),
            updates=[dict(iteration=i,start=start,end=end,ids=t.cpu().tolist()) for i,start,end,t in self.updates],
            event_count=self.index,clock=self.clock,
            scope='Actual outer intervals for every occurrence. GPU intervals can contain CPU launch gaps. No operator-service claim.')
