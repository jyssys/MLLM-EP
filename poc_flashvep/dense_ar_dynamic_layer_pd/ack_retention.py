"""Keep source KV until Native ACK, or fail the bounded experiment safely."""
import atexit,time
from .common import *


class LeaseGuard:
    def __init__(self,maximum_hold_s):self.maximum=maximum_hold_s;self.seen={};self.renewals=[]
    def retain(self,pending,now):
        for rid in list(self.seen):
            if rid not in pending:del self.seen[rid]
        for rid,expires in list(pending.items()):
            self.seen.setdefault(rid,now)
            assert now-self.seen[rid]<self.maximum,'Native ACK did not arrive within the bounded hold. Do not free live source KV.'
            if expires<now+2:
                pending[rid]=now+30
                self.renewals.append(dict(request_id=rid,time=now,old_expiry=expires,new_expiry=pending[rid]))


def install(cfg,role):
    if role!='prefill' or not cfg.get('native_ACK_hold'):return
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    original=NixlBaseConnectorWorker.get_finished
    guards={}
    def finished(self,*args,**kwargs):
        guard=guards.setdefault(id(self),LeaseGuard(cfg.get('native_ACK_max_hold_s',cfg.get('timeout',600))))
        guard.retain(self._reqs_to_send,time.perf_counter())
        result=original(self,*args,**kwargs)
        guard.retain(self._reqs_to_send,time.perf_counter())
        return result
    NixlBaseConnectorWorker.get_finished=finished
    def export():
        write(ROOT/'raw'/cfg['name']/'native_ACK_retention_prefill.json',dict(
            renewals=[r for g in guards.values() for r in g.renewals],
            pending_seen=[rid for g in guards.values() for rid in g.seen],
            scope='Process-local optional safety guard. Native completion notifications still release KV. A missing ACK causes a bounded experiment failure, not silent Source reuse. This is generic transport safety, not a Layer speedup.'))
    atexit.register(export)
