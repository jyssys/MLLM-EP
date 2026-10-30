"""Host API and admission diagnostics. These are not exclusive GPU DMA times."""
import os
import time
from collections import Counter


def queued_compute_ready(connector,request):
    """Allow immutable fixed-depth KV to remain queued. Retain event and byte limits."""
    t=connector.transfers.get(request.request_id)
    if t is None or not t.outgoing:
        return True
    if t.request is not request or t.cancelled:
        return False
    params=request.sampling_params
    if params.min_loops!=params.max_loops or params.max_loops!=connector.cache.max_loops:
        raise ValueError('Queued-compute experiment requires full fixed depth')
    return ((t.last_event is None or t.last_event.query()) and
            connector.transport.inflight_bytes<connector.config.max_inflight_bytes)


def install():
    if os.environ.get('B_NUMERICAL_QUEUED_COMPUTE_GENERIC')=='1':
        from vllm_rlt.pd.connector import PDStageConnector
        PDStageConnector.can_schedule=queued_compute_ready
    if os.environ.get('B_NUMERICAL_QUEUED_COMPUTE')=='1':
        from ..looped_pd_abc.frontier import FrontierConnector
        FrontierConnector.can_schedule=queued_compute_ready
    if os.environ.get('B_NUMERICAL_TRANSFER_PROBE')!='1': return
    from vllm_rlt.pd.connector import PDStageConnector
    from vllm_rlt.pd.transport import NixlConnector
    if getattr(PDStageConnector,'_numerical_transfer_probe',False): return
    PDStageConnector._numerical_transfer_probe=True
    totals=Counter()
    intervals=[]
    blocked={}
    poll=NixlConnector.poll
    def poll_timed(self):
        begin=time.perf_counter()
        try: return poll(self)
        finally:
            totals['poll_calls']+=1
            totals['poll_host_seconds']+=time.perf_counter()-begin
    NixlConnector.poll=poll_timed
    submit=NixlConnector.submit
    def submit_timed(self,tid,sequence,peer,local,remote,size):
        begin=time.perf_counter()
        value=submit(self,tid,sequence,peer,local,remote,size)
        totals['submit_calls']+=1
        totals['submit_host_seconds']+=time.perf_counter()-begin
        totals['accepted_submits' if value else 'rejected_submits']+=1
        if value:
            totals['accepted_descriptors']+=len(local)
            totals['accepted_bytes']+=size
        return value
    NixlConnector.submit=submit_timed
    can=PDStageConnector.can_schedule
    def can_schedule(self,request):
        value=can(self,request)
        t=self.transfers.get(request.request_id)
        if t is not None and t.outgoing:
            now=time.monotonic()
            totals['prefill_can_schedule_calls']+=1
            if not value:
                reason='chunks' if t.chunks else 'event' if t.last_event and not t.last_event.query() else 'inflight_limit'
                totals['blocked_'+reason+'_observations']+=1
                blocked.setdefault(request.request_id,(now,reason))
            elif request.request_id in blocked:
                start,reason=blocked.pop(request.request_id)
                intervals.append(dict(tid=request.request_id,start=start,end=now,seconds=now-start,first_reason=reason))
        return value
    PDStageConnector.can_schedule=can_schedule
    if os.environ.get('B_NUMERICAL_QUEUED_COMPUTE')=='1':
        from ..looped_pd_abc.frontier import FrontierConnector
        # Retain probe accounting for the independently overridden B readiness method.
        def queued_observed(self,request):
            value=queued_compute_ready(self,request)
            totals['queued_compute_checks']+=1
            totals['queued_compute_ready_with_pending_chunks']+=int(value and bool(getattr(self.transfers.get(request.request_id),'chunks',())))
            return value
        FrontierConnector.can_schedule=queued_observed
    from multiprocessing.util import Finalize
    def finish():
        from .common import ROOT,write
        write(ROOT/'raw'/os.environ['B_NUMERICAL_RUN']/f'transfer_host_probe_{os.getpid()}.json',
              dict(totals=dict(totals),blocked_intervals=intervals,unfinished_blocked=blocked,
                   evidence='MEASURED_INSTRUMENTED_HOST_OBSERVATION',
                   scope='Process lifetime includes warmup. Host API durations can overlap GPU work. Admission intervals are polling observations, not exclusive DMA or additive E2E.'))
    Finalize(None,finish,exitpriority=14)
