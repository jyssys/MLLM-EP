"""Coalesce exact byte ranges inside registered KV layer regions."""
import atexit,time
from .common import *


def coalesce(local,remote,local_regions,remote_regions):
    import numpy as np
    assert local.shape==remote.shape and local.ndim==2 and local.shape[1]==3
    order=np.argsort(local[:,0],kind='stable');a=local[order];b=remote[order]
    lr=np.asarray(local_regions)[order];rr=np.asarray(remote_regions)[order]
    if not len(a):return a,b
    assert np.array_equal(a[:,1],b[:,1]),'Source and destination byte counts differ'
    assert np.all(a[1:,0]>=a[:-1,0]+a[:-1,1]),'Overlapping destinations need original transfer order'
    contiguous=(a[1:,0]==a[:-1,0]+a[:-1,1])&(b[1:,0]==b[:-1,0]+b[:-1,1])&(
        a[1:,2]==a[:-1,2])&(b[1:,2]==b[:-1,2])&(lr[1:]==lr[:-1])&(rr[1:]==rr[:-1])
    starts=np.r_[0,np.flatnonzero(~contiguous)+1]
    sizes=np.add.reduceat(a[:,1],starts)
    x=a[starts].copy();y=b[starts].copy();x[:,1]=sizes;y[:,1]=sizes
    assert int(x[:,1].sum())==int(local[:,1].sum())
    return x,y


def install(cfg,role):
    if role!='decode' or not cfg.get('nixl_manual_coalesce'):return
    import numpy as np
    from nixl._api import nixl_agent
    prep=nixl_agent.prep_xfer_dlist;make=nixl_agent.make_prepped_xfer
    get_desc=nixl_agent.get_xfer_descs;tables={};rows=[];descriptor_arrays={}
    def described(self,xfer_list,*args,**kwargs):
        value=get_desc(self,xfer_list,*args,**kwargs)
        # vLLM converts the array to a pybind descriptor object before prep.
        # Retain that object's identity until its prepared handle is recorded.
        if isinstance(xfer_list,np.ndarray) and xfer_list.ndim==2 and xfer_list.shape[1]==3 and len(xfer_list)>100000:
            descriptor_arrays[id(value)]=(value,np.array(xfer_list,copy=True))
        return value
    def prepared(self,agent_name,xfer_list,mem_type=None,backends=[]):
        handle=prep(self,agent_name,xfer_list,mem_type,backends)
        if isinstance(xfer_list,np.ndarray) and xfer_list.ndim==2 and xfer_list.shape[1]==3:
            tables[(id(self),handle._handle)]=(np.array(xfer_list,copy=True),agent_name)
        else:
            entry=descriptor_arrays.get(id(xfer_list))
            if entry is not None and entry[0] is xfer_list:tables[(id(self),handle._handle)]=(entry[1],agent_name)
        return handle
    def merged(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw):
        lt=tables.get((id(self),local_side._handle));rt=tables.get((id(self),remote_side._handle))
        if operation!='READ' or lt is None or rt is None or len(lt[0])%36 or len(rt[0])%36:
            return make(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw)
        start=time.monotonic();li=np.asarray(local_indices);ri=np.asarray(remote_indices)
        a=lt[0][li];b=rt[0][ri]
        try:x,y=coalesce(a,b,li//(len(lt[0])//36),ri//(len(rt[0])//36))
        except AssertionError:
            return make(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw)
        prepared_at=time.monotonic()
        if len(x)>=len(a):
            value=make(self,operation,local_side,local_indices,remote_side,remote_indices,*args,**kw)
        else:
            # Keep the original completion notification and native handle type.
            # Original memory registration still owns every source byte.
            notif=args[0] if args else kw.get('notif_msg',b'')
            backend=args[1] if len(args)>1 else kw.get('backends',[])
            local=self.get_xfer_descs(x,'VRAM');remote=self.get_xfer_descs(y,'VRAM')
            value=self.initialize_xfer(operation,local,remote,rt[1],notif_msg=notif,backends=backend)
        rows.append(dict(time=start,input_descriptors=len(a),submitted_descriptors=len(x),
            bytes=int(a[:,1].sum()),coalescing_host_ms=1000*(prepared_at-start),
            request_setup_host_ms=1000*(time.monotonic()-prepared_at)))
        return value
    nixl_agent.get_xfer_descs=described
    nixl_agent.prep_xfer_dlist=prepared;nixl_agent.make_prepped_xfer=merged
    atexit.register(lambda:write(ROOT/'raw'/cfg['name']/'manual_KV_coalescing.json',dict(rows=rows,
        evidence='MEASURED_HOST_BOUNDARIES',scope='Merge only physically contiguous source/destination ranges inside the same registered layer regions. Keep native READ completion and ACK. No byte, layer, or token is omitted.')))
