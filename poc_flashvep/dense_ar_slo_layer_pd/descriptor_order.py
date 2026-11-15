"""Observe paired descriptor order and optionally sort exact transfer pairs."""
import atexit,time
from .common import *


def runs(local,remote):
    if not len(local):return 0
    return 1+sum(int(a1)!=int(a0)+1 or int(b1)!=int(b0)+1
        for a0,a1,b0,b1 in zip(local,local[1:],remote,remote[1:]))


def ordered_pairs(local,remote):
    assert len(local)==len(remote)
    order=sorted(range(len(local)),key=lambda i:int(local[i]))
    return [int(local[i]) for i in order],[int(remote[i]) for i in order]


def install(cfg,role):
    if role!='decode' or not cfg.get('descriptor_order_probe'):return
    from nixl._api import nixl_agent
    original=nixl_agent.make_prepped_xfer;rows=[]
    def observe(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw):
        start=time.monotonic();local,remote=ordered_pairs(local_indices,remote_indices)
        row=dict(time=start,count=len(local),original_paired_ascending_runs=runs(local_indices,remote_indices),
            sorted_paired_ascending_runs=runs(local,remote),sorted=bool(cfg.get('sort_descriptor_pairs')),
            ordering_host_ms=1000*(time.monotonic()-start),
            pairs_SHA256=hashlib.sha256(json.dumps(list(zip(local,remote))).encode()).hexdigest())
        call_start=time.monotonic()
        if cfg.get('sort_descriptor_pairs'):
            value=original(self,operation,local_side,local,remote_side,remote,*args,**kw)
        else:value=original(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw)
        row['make_prepped_xfer_host_ms']=1000*(time.monotonic()-call_start);rows.append(row);return value
    nixl_agent.make_prepped_xfer=observe
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'descriptor_order_probe.json',dict(rows=rows,
        evidence='STRUCTURAL_ACCOUNTING',scope='Paired descriptor-index runs are not wire throughput or merged NIXL descriptor counts. Sorting preserves source/destination pairs and changes no model work. Probe costs perturb diagnostic runs.')))
