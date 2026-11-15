"""Sort exact NIXL source/destination pairs without Python descriptor loops."""
import atexit,time
from .common import *


def install(cfg,role):
    if role!='decode' or not cfg.get('nixl_descriptor_sort'):return
    import numpy as np
    from nixl._api import nixl_agent
    original=nixl_agent.make_prepped_xfer;rows=[]
    def sorted_read(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw):
        begin=time.monotonic();local=np.asarray(local_indices);remote=np.asarray(remote_indices)
        assert local.ndim==remote.ndim==1 and len(local)==len(remote)
        permutation=np.argsort(local,kind='stable')
        ordered_local=local[permutation];ordered_remote=remote[permutation]
        # Aliased destinations can make ordering semantic. Keep the original
        # order for that case. Prefix caching is disabled in this campaign.
        unique=len(local)==0 or bool(np.all(ordered_local[1:]!=ordered_local[:-1]))
        sort_end=time.monotonic()
        if unique:value=original(self,operation,local_side,ordered_local,remote_side,ordered_remote,*args,**kw)
        else:value=original(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw)
        rows.append(dict(time=begin,input_descriptors=len(local),sorted=unique,
            sort_preparation_ms=1000*(sort_end-begin),make_prepped_host_ms=1000*(time.monotonic()-sort_end)))
        return value
    nixl_agent.make_prepped_xfer=sorted_read
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'nixl_paired_sort.json',dict(rows=rows,
        evidence='MEASURED_HOST_BOUNDARIES',scope='Only descriptor pair order changes. Original NIXL READ, completion, notifications, and ACK remain intact. These Host intervals are not exclusive wire times.')))
