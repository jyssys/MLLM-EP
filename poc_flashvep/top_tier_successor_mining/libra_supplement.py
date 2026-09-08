"""Thin safety/provenance wrapper around the supplied, unchanged Cython code."""
import importlib.util
from pathlib import Path
import time
import numpy as np


class SupplementPlanner:
    def __init__(self,extension,replicas=8,local=4):
        path=Path(extension)
        spec=importlib.util.spec_from_file_location("prefetch_rebalance_cython",path)
        self.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.replicas,self.local=replicas,local

    def prefetch(self,ids):
        ids=np.ascontiguousarray(ids,dtype=np.int64)
        g,m,k=ids.shape
        # The supplied boundscheck=False Cython assumes valid, equal-size
        # source arrays. Never pass -1 padding or an out-of-bounds expert.
        assert g==4 and m>0 and k==8 and ids.min()>=0 and ids.max()<128
        send=np.zeros((g,g,self.replicas),dtype=np.int64)
        send_len=np.zeros((g,g),dtype=np.int64)
        recv_len=np.zeros_like(send_len)
        t0=time.perf_counter()
        mapping,remote,recv=self.module.prefetch_impl(ids,128,m,g,0,send,send_len,recv_len,
            self.replicas,self.local,7,False)
        elapsed=(time.perf_counter()-t0)*1000
        mapping,remote=np.array(mapping,copy=True),np.array(remote,copy=True)
        assert mapping.shape==(128,4) and mapping.dtype==np.bool_
        assert mapping[np.arange(128),np.arange(128)//32].all()
        assert (mapping.sum(0)-32<=self.replicas).all()
        assert (send_len.sum(0)==recv_len.sum(1)).all()
        assert (send_len<=self.replicas).all()
        return {"placement":mapping,"remote":remote,"send":send,"send_len":send_len,
                "recv_len":recv_len,"prefetch_cpu_ms":elapsed}

    def rebalance(self,ids,plan):
        ids=np.ascontiguousarray(ids,dtype=np.int64)
        g,m,k=ids.shape
        assert g==4 and m>0 and k==8 and ids.min()>=0 and ids.max()<128
        masks=np.zeros((g,g,m,k),dtype=np.int64)
        times=[]
        for rank in range(g):
            remote=plan["remote"].copy()
            t0=time.perf_counter()
            self.module.rebalance_impl(ids,plan["placement"],remote,128,m,g,rank,masks[rank],0)
            times.append((time.perf_counter()-t0)*1000)
        local=np.stack([plan["placement"][ids[r],r] for r in range(g)])
        assert np.array_equal(masks.sum(0)+local,np.ones_like(ids)),"Assignment conservation"
        for dest in range(g):
            assigned=ids[masks[dest].astype(bool)]
            assert plan["placement"][assigned,dest].all(),"Replica ownership"
            assert not masks[dest,dest].any(),"Remote stage must exclude local source"
        return {"remote_masks":masks.astype(bool),"local_mask":local,
                "loads":masks.sum((1,2,3))+local.sum((1,2)),
                "rebalance_cpu_ms_per_rank":times}
