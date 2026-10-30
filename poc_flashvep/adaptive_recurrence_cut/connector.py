"""NIXL cut handoff for complete prompt state, with depth-aware valid KV."""
import hashlib
import os
import time
import torch
from vllm_rlt.pd.connector import PDStageConnector
from vllm_rlt.pd.transport import NixlConnector,partition_segments
from vllm_rlt.request import Request,RequestPhase
from ..looped_pd_abc.segments import coalesce
from .common import ROOT,load_config,write,append


class CutConnector(PDStageConnector):
    @classmethod
    def create(cls,engine,*,config,notify):
        cfg=load_config()
        if engine.scheduler.config.mode!='refill' or not engine.scheduler.config.wavefront_chunked_prefill:
            raise ValueError('Cut migration requires separate-phase staged G1')
        parameter=next(engine.model.parameters())
        limit=engine.scheduler.config.max_num_seqs+(config.max_receiving_requests if engine.stage_config.produces_tokens else config.max_draining_requests)
        hidden=torch.zeros((limit,cfg['state_capacity_tokens']*engine.model.config.hidden_size),
                           device=parameter.device,dtype=parameter.dtype)
        self=cls(engine,hidden,NixlConnector(engine.stage_config.stage_id,engine.cache_manager,hidden,config),config,notify=notify)
        self.cfg=cfg
        self.original_accept=engine.accept_import
        engine.accept_import=self.accept_cut
        previous=self.notify

        def notify_cut(kind,**fields):
            if kind=='commit':
                t=self.transfers[fields['tid']]
                fields['rng_state']=getattr(t,'rng_state',None)
                fields['arc_cut']=t.cut
                if cfg.get('verify_transfer'):fields['fingerprint']=self.fingerprint(t)
            previous(kind,**fields)

        self.notify=notify_cut
        return self

    def command(self,command):
        kind=command['kind']
        if kind in ('reserve','prefill'):
            self._next_cut=command.get('arc_cut',self.cfg['cut'])
        if kind=='commit':
            t=self.transfers[command['tid']]
            if command.get('arc_cut',t.cut)!=t.cut:raise RuntimeError('Cut changed after reservation')
            t.expected_fingerprint=command.get('fingerprint')
            if command.get('rng_state') is not None:
                g=self.engine.model_runner.rng.acquire(t.tid,t.request.sampling_params,self.cache.device)
                g.set_state(torch.tensor(command['rng_state'],dtype=torch.uint8))
        return super().command(command)

    def reserve(self,tid,tokens,*args,**kwargs):
        if len(tokens)>self.cfg['state_capacity_tokens']:raise ValueError('Prompt state capacity exceeded')
        super().reserve(tid,tokens,*args,**kwargs)
        if tid not in self.transfers:return
        t=self.transfers[tid]
        t.cut=self._next_cut
        if not t.outgoing:t.request.arc_received_cut=t.cut
        if not 0<=t.cut<=self.cache.max_loops:raise ValueError('Invalid recurrent cut')
        if t.cut==0:
            if t.outgoing:raise ValueError('Cut zero must never create P-side work')
            t.ready=True
            t.request.import_resume_phase=RequestPhase.PREFILLING_PRELUDE
            self.engine.scheduler.queues[RequestPhase.WAITING].append(tid)
        if t.outgoing:t.request.arc_export_cut=t.cut

    def prefill_submitted(self,request,*,phase,start,progress,event,stream):
        t=self.transfers[request.request_id]
        if t.compute_done:return
        n=len(request.prompt_token_ids)
        t.last_event=event
        if request.prefill_depth_ranges[t.cut-1][0]<n:return
        if any(left for left,right in request.prefill_depth_ranges[t.cut:]):
            raise RuntimeError('P executed work beyond the committed recurrent cut')
        runner=self.engine.model_runner
        slot=runner._ensure_request_slot(request)
        stream=stream or torch.cuda.current_stream(self.cache.device)
        with torch.cuda.stream(stream):
            if event is not None:stream.wait_event(event)
            state=request.hidden_state if t.cut==self.cache.max_loops else runner.prefill_states[slot,:n]
            self.hidden[t.slot,:state.numel()].copy_(state.flatten())
            ready=torch.cuda.Event();ready.record(stream)
        g=runner.rng.peek(request.request_id)
        if g is not None:t.rng_state=g.get_state().tolist()
        t.last_event=ready
        self.queue_snapshot(t,n,ready)
        self.engine.suspend_transfer(request)
        t.compute_done=True
        t.transfer_boundary_first=t.transfer_boundary_last=(0,n)
        t.transfer_boundary_count=1
        self.prefill_tokens+=n

    def layout(self,t):
        n=len(t.request.prompt_token_ids)
        recurrent=set(getattr(self.engine.model,'recurrent_kv_layers',range(self.cache.num_layers)))
        coda=set(getattr(self.engine.model,'coda_kv_layers',()))
        for depth in range(t.cut):
            for layer in range(self.cache.num_layers):
                if depth and layer not in recurrent:continue
                if layer in coda and t.cut<self.cache.max_loops:continue
                end=n-1 if layer in coda else n
                yield depth,layer,end

    def segments(self,t,info,tables):
        for depth,layer,end in self.layout(t):
            table=tables[depth]
            for position in range(0,end,self.cache.block_size):
                size=min(self.cache.block_size,end-position)*info['token_bytes']
                for base in (info['key_ptr'],info['value_ptr']):
                    yield base+table[position//self.cache.block_size]*info['block_bytes']+layer*info['layer_bytes'],size,info['device']

    def hidden_elements(self,t):
        return self.engine.model.config.hidden_size*(len(t.request.prompt_token_ids) if t.cut<self.cache.max_loops else 1)

    def queue_snapshot(self,t,n,event):
        peer=self.transport.peers[t.peer]
        source=list(self.segments(t,self.transport.info,self.cache.plane_block_tables(t.tid)))
        target=list(self.segments(t,peer,t.target_tables))
        kv_bytes=sum(segment[1] for segment in source)
        size=self.hidden_elements(t)*self.hidden.element_size()
        registered=self.transport.info['hidden_bytes']
        source.append((self.transport.info['hidden_ptr']+t.slot*registered,size,self.transport.info['device']))
        target.append((peer['hidden_ptr']+t.target_slot*registered,size,peer['device']))
        source,target=coalesce(source,target)
        descriptors=0
        for local,remote,amount in partition_segments(source,target,self.config.transfer_chunk_bytes,self.config.max_transfer_descriptors):
            t.chunks.append((event,t.next_sequence,local,remote,amount));t.next_sequence+=1
            descriptors+=len(local)
        append(ROOT/'raw'/os.environ['ARC_RUN']/f'migration_{os.getpid()}.jsonl',
               dict(time=time.monotonic(),seed=t.request.sampling_params.seed,cut=t.cut,
                    prompt_tokens=n,KV_bytes=kv_bytes,hidden_bytes=size,
                    fragments=t.next_sequence,descriptors=descriptors,
                    evidence='MEASURED_TRANSPORT_PAYLOAD_ACCOUNTING'))

    def fingerprint(self,t):
        allocation=self.cache._get_allocation(t.tid)
        h=hashlib.sha256();size=0
        for depth,layer,end in self.layout(t):
            written=allocation.written[depth][layer].prefix
            if written<end:raise RuntimeError(f'KV frontier is not valid: {depth}/{layer} {written}<{end}')
            blocks=torch.tensor(allocation.block_tables[depth][:(end+self.cache.block_size-1)//self.cache.block_size],device=self.cache.device)
            for tensor in (self.cache.key_cache,self.cache.value_cache):
                value=tensor.index_select(0,blocks)[:,layer].flatten(0,1)[:end]
                raw=value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                h.update(raw);size+=len(raw)
        raw=self.hidden[t.slot,:self.hidden_elements(t)].contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return dict(KV_sha256=h.hexdigest(),KV_bytes=size,hidden_sha256=hashlib.sha256(raw).hexdigest(),hidden_bytes=len(raw))

    def accept_cut(self,request,hidden,*,cached_tokens=0):
        if cached_tokens:raise ValueError('Prefix-cache migration is not qualified')
        if self.engine.scheduler.requests.get(request.request_id) is not request:return False
        t=self.transfers[request.request_id]
        n=len(request.prompt_token_ids);width=self.engine.model.config.hidden_size
        allocation=self.cache._get_allocation(request.request_id)
        if t.cut==self.cache.max_loops:
            accepted=self.original_accept(request,hidden[:width],cached_tokens=0)
        else:
            for depth,layer,end in self.layout(t):allocation.written[depth][layer].prefix=end
        if self.cfg.get('verify_transfer'):
            actual=self.fingerprint(t)
            row=dict(seed=request.sampling_params.seed,cut=t.cut,actual=actual,
                     expected=t.expected_fingerprint,passed=actual==t.expected_fingerprint,
                     evidence='MEASURED_HEAVY_BYTE_IDENTITY')
            append(ROOT/'raw'/os.environ['ARC_RUN']/f'KV_checks_{os.getpid()}.jsonl',row)
            if not row['passed']:raise RuntimeError('KV or hidden transport identity failed')
        if t.cut==self.cache.max_loops:
            return accepted
        request.num_preluded_tokens=n
        request.num_prefilled_tokens=0
        request.prefill_depth_ranges[:]=Request.prefill_boundaries(n,self.cache.max_loops)
        for depth in range(t.cut):
            request.prefill_depth_ranges[depth][0]=n
            if depth+1<self.cache.max_loops:request.prefill_depth_ranges[depth+1][1]=n
        request.prefill_window_origin=request.prefill_reclaimable_tokens=0
        request.loops_done=t.cut
        request.import_prompt_state=hidden[:n*width].view(n,width)
        request.import_resume_phase=RequestPhase.PREFILLING_RECURRENT
        self.engine.scheduler.queues[RequestPhase.WAITING].append(request.request_id)
        return True
