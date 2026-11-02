"""Audit lazy captures and warm native geometries with real, valid requests."""
import os
import sys
import time
from .common import ROOT,write,append


def worker_stats(engine):
    if hasattr(engine,'peers'):
        return {name:peer.client.stats() for name,peer in engine.peers.items()}
    return engine.stats()


def install(cfg):
    from vllm_rlt.worker import cuda_graph
    if not getattr(cuda_graph,'_recovery_capture_audit',False):
        cuda_graph._recovery_capture_audit=True
        capture=cuda_graph._capture
        def observed(hidden,stream,pool,run,**kwargs):
            begin=time.monotonic()
            frame=sys._getframe(1)
            owner=frame.f_locals.get('self')
            result=capture(hidden,stream,pool,run,**kwargs)
            append(ROOT/'raw'/cfg['name']/f'graph_captures_{os.getpid()}.jsonl',dict(
                time=begin,end=time.monotonic(),host_seconds=time.monotonic()-begin,
                owner=type(owner).__name__,rows=len(hidden),startup=kwargs.get('startup',False),
                measured_phase=(ROOT/'raw'/cfg['name']/'measurement_begin.json').exists(),
                evidence='MEASURED_NATIVE_CAPTURE_HOST_SPAN',scope='Native capture includes synchronization, two warm executions and GC. Not pure GPU kernel time.'))
            return result
        cuda_graph._capture=observed
    if not cfg.get('prime_all_graph_shapes'):return
    from ..looped_pd_atomic_prefill import rlt_worker
    original=rlt_worker.serve
    from vllm_rlt.pd.connector import PDStageConnector
    available=PDStageConnector.can_schedule
    folder=ROOT/'raw'/cfg['name'];marker=folder/'native_graph_warmup_phase.json'
    def can_schedule(self,request):
        if getattr(self,'_recovery_all_warmup_done',False):return available(self,request)
        if marker.exists():
            phase=__import__('json').loads(marker.read_text())
            if phase['state']=='ALL_COMPLETE':
                self._recovery_all_warmup_done=True
                return available(self,request)
            if phase['state']=='RUNNING' and request.sampling_params.seed in phase['seeds']:
                key=tuple(phase['seeds'])
                if getattr(self,'_recovery_released_warmup',None)!=key:
                    selected=[t for t in self.transfers.values() if t.request.sampling_params.seed in phase['seeds'] and not t.cancelled]
                    wanted=phase['count']
                    outgoing=self.engine.stage_config.role=='prefill'
                    # A short D-geometry cohort can exceed P compute slots.
                    # P must continue importing these requests one by one.
                    # Only D waits until all valid imports are ready.
                    if outgoing and phase['kind']=='short_D_geometry':return available(self,request)
                    ready=len(selected)==wanted and (outgoing or all(t.ready for t in selected))
                    if not ready:return False
                    self._recovery_released_warmup=key
        return available(self,request)
    PDStageConnector.can_schedule=can_schedule

    def drain(engine,deadline):
        while getattr(engine,'transfers',{}):
            if time.monotonic()>deadline:raise TimeoutError('Native warm-up transfer drain')
            assert not engine.step(),'Warm-up left an unconsumed output'

    def phase(engine,inputs,kind,rows,short):
        from vllm_rlt import SamplingParams
        tokens=[list(inputs[i%len(inputs)][:512] if short else inputs[i%len(inputs)]) for i in range(rows)]
        seeds=[cfg['seed']-10000-(100 if short else 0)+i for i in range(rows)]
        replicas=len(engine.clients) if hasattr(engine,'clients') else 1
        caps=[2+(i//replicas)%cfg.get('graph_rows',cfg['capacity']) if short else 1 for i in range(rows)]
        write(marker,dict(state='RUNNING',kind=kind,seeds=seeds,count=rows,time=time.monotonic()))
        before=worker_stats(engine);active={};results=[];begin=time.monotonic();deadline=begin+300
        for i,t in enumerate(tokens):
            rid=f'recovery_warm_{kind}_{i}';active[rid]=i
            engine.add_request(rid,t,SamplingParams(max_tokens=caps[i],temperature=0.,seed=seeds[i],
                min_loops=cfg['depth'],max_loops=cfg['depth'],exit_threshold=1.,ignore_eos=True))
        while active:
            assert time.monotonic()<deadline,'Graph warm-up watchdog'
            for out in engine.step():
                if not out.finished:continue
                i=active.pop(out.request_id)
                assert len(out.token_ids)==caps[i] and set(out.exit_depths)=={cfg['depth']}
                row=dict(warmup=True,name=cfg['name'],model=cfg['model'],sample_id=i,seed=seeds[i],
                    input_ids=tokens[i],output_ids=list(out.token_ids),exit_depths=list(out.exit_depths),
                    input_tokens=len(tokens[i]),output_tokens=len(out.token_ids),finish_reason=str(out.finish_reason),
                    evidence='MEASURED_NATIVE_GEOMETRY_WARMUP',scope='Additional startup requests only. Not in measured arrivals, throughput, SLO, or model accuracy.')
                append(folder/'graph_warmup_requests.jsonl',row);results.append(row)
        drain(engine,deadline)
        after=worker_stats(engine)
        observed=sum(v['issued_loop_rows']-before[k]['issued_loop_rows'] for k,v in after.items())
        expected=sum((len(t)+cap-1)*cfg['depth'] for t,cap in zip(tokens,caps))
        assert observed==expected,'Warm-up changed active fixed-depth accounting'
        write(marker,dict(state='COMPLETE',kind=kind,time=time.monotonic()))
        append(folder/'graph_warmup_accounting.jsonl',dict(kind=kind,count=rows,expected=expected,observed=observed,
            passed=True,host_seconds=time.monotonic()-begin,scope='Native active request rows. Internal native graph-capture warm executions are not request work counters.'))
        return results

    def serve(engine,inputs,args,run,warmup=False,contract=None):
        if not warmup:return original(engine,inputs,args,run,False,contract)
        # Preserve the original full-length warm-up first. Do not make the
        # timed long-prompt kernel specialization cold by using only short text.
        result=original(engine,inputs,args,run,True,contract)
        drain(engine,time.monotonic()+300)
        if cfg['arm'].startswith('DP'):
            count=2*cfg.get('graph_rows',cfg['capacity'])
        else:
            phase(engine,inputs,'full_P_batch',min(cfg['P_slots'],len(inputs)),False)
            count=min(cfg.get('graph_rows',cfg['capacity']),cfg['D_slots'])
        phase(engine,inputs,'short_D_geometry',count,True)
        write(marker,dict(state='ALL_COMPLETE',time=time.monotonic(),
            scope='Extra short warm-up prefixes can cycle supplied inputs with independent request IDs. Measured prompts never cycle.'))
        return result
    rlt_worker.serve=serve
