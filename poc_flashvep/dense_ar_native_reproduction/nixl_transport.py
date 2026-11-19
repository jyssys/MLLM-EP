"""Observe standard NIXL setup and optionally reuse exact physical coalescing."""
import atexit
import time
from .common import ROOT,write


def coalesce(local,remote,local_regions,remote_regions):
    import numpy as np
    assert local.shape==remote.shape and local.ndim==2 and local.shape[1]==3
    order=np.argsort(local[:,0],kind='stable');a=local[order];b=remote[order]
    lr=np.asarray(local_regions)[order];rr=np.asarray(remote_regions)[order]
    if not len(a):return a,b
    assert np.array_equal(a[:,1],b[:,1])
    assert np.all(a[1:,0]>=a[:-1,0]+a[:-1,1])
    contiguous=(a[1:,0]==a[:-1,0]+a[:-1,1])&(b[1:,0]==b[:-1,0]+b[:-1,1])&(
        a[1:,2]==a[:-1,2])&(b[1:,2]==b[:-1,2])&(lr[1:]==lr[:-1])&(rr[1:]==rr[:-1])
    starts=np.r_[0,np.flatnonzero(~contiguous)+1];sizes=np.add.reduceat(a[:,1],starts)
    x=a[starts].copy();y=b[starts].copy();x[:,1]=sizes;y[:,1]=sizes
    assert int(x[:,1].sum())==int(local[:,1].sum())
    return x,y


def install(cfg,role):
    if role not in ('prefill','decode'):return
    import numpy as np
    from nixl._api import nixl_agent
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import NixlPullConnectorWorker
    rows=[];tables={};descriptor_arrays={};active={};folder=ROOT/'raw'/cfg['name']
    get_desc=nixl_agent.get_xfer_descs;prep=nixl_agent.prep_xfer_dlist
    make=nixl_agent.make_prepped_xfer;transfer=nixl_agent.transfer
    get_telemetry=nixl_agent.get_xfer_telemetry;requests_by_handle={}
    original_read=NixlPullConnectorWorker._read_blocks
    def read(self,*a,**kw):
        begin=time.monotonic();active['request_id']=kw.get('request_id')
        try:return original_read(self,*a,**kw)
        finally:
            rows.append(dict(kind='read_blocks_host',request_id=active.get('request_id'),start=begin,end=time.monotonic()))
            active.clear()
    NixlPullConnectorWorker._read_blocks=read
    def described(self,xfer_list,*a,**kw):
        value=get_desc(self,xfer_list,*a,**kw)
        if cfg.get('nixl_manual_coalesce') and isinstance(xfer_list,np.ndarray) and xfer_list.ndim==2 and xfer_list.shape[1]==3 and len(xfer_list)>100000:
            descriptor_arrays[id(value)]=(value,np.array(xfer_list,copy=True))
        return value
    def prepared(self,agent_name,xfer_list,mem_type=None,backends=[]):
        handle=prep(self,agent_name,xfer_list,mem_type,backends)
        if cfg.get('nixl_manual_coalesce'):
            if isinstance(xfer_list,np.ndarray) and xfer_list.ndim==2 and xfer_list.shape[1]==3:
                tables[(id(self),handle._handle)]=(np.array(xfer_list,copy=True),agent_name)
            elif id(xfer_list) in descriptor_arrays:
                entry=descriptor_arrays[id(xfer_list)]
                if entry[0] is xfer_list:tables[(id(self),handle._handle)]=(entry[1],agent_name)
        return handle
    def merged(self,operation,local_side,local_indices,remote_side,remote_indices,*a,**kw):
        begin=time.monotonic();before=len(local_indices);after=before;byte_count=None
        lt=tables.get((id(self),local_side._handle));rt=tables.get((id(self),remote_side._handle))
        x=y=None
        if cfg.get('nixl_manual_coalesce') and operation=='READ' and lt is not None and rt is not None and len(lt[0])%36==0 and len(rt[0])%36==0:
            li=np.asarray(local_indices);ri=np.asarray(remote_indices);source=lt[0][li];dest=rt[0][ri]
            try:x,y=coalesce(source,dest,li//(len(lt[0])//36),ri//(len(rt[0])//36))
            except AssertionError:pass
            if x is not None:after=len(x);byte_count=int(x[:,1].sum())
        if x is None or after>=before:
            result=make(self,operation,local_side,local_indices,remote_side,remote_indices,*a,**kw)
        else:
            local=self.get_xfer_descs(x,'VRAM');remote=self.get_xfer_descs(y,'VRAM')
            notif=a[0] if a else kw.get('notif_msg',b'');backend=a[1] if len(a)>1 else kw.get('backends',[])
            result=self.initialize_xfer(operation,local,remote,rt[1],notif_msg=notif,backends=backend)
        rows.append(dict(kind='make_transfer_host',request_id=active.get('request_id'),start=begin,end=time.monotonic(),
            input_descriptors=before,submitted_descriptors=after,bytes=byte_count))
        return result
    def transferred(self,*a,**kw):
        if a:requests_by_handle[id(a[0])]=active.get('request_id')
        begin=time.monotonic();cpu_begin=time.thread_time();result=transfer(self,*a,**kw)
        rows.append(dict(kind='submit_transfer_host',request_id=active.get('request_id'),start=begin,end=time.monotonic(),
            calling_thread_CPU_ms=1000*(time.thread_time()-cpu_begin),returned_state=result))
        return result
    def telemetry(self,handle,*a,**kw):
        result=get_telemetry(self,handle,*a,**kw)
        rows.append(dict(kind='NIXL_native_telemetry',request_id=requests_by_handle.pop(id(handle),None),
            time=time.monotonic(),data={key:getattr(result,key,None) for key in
                ('startTime','postDuration','xferDuration','totalBytes','descCount')},
            duration_unit='microseconds, per installed NIXL API. Not pure DMA or CPU-exclusive time.'))
        return result
    nixl_agent.get_xfer_descs=described;nixl_agent.prep_xfer_dlist=prepared
    nixl_agent.make_prepped_xfer=merged;nixl_agent.transfer=transferred;nixl_agent.get_xfer_telemetry=telemetry
    def close():
        write(folder/f'NIXL_host_setup_{role}.json',dict(rows=rows,coalescing=cfg.get('nixl_manual_coalesce',False),
            evidence='MEASURED_HOST_BOUNDARIES',scope='Nested host enclosures, not pure DMA time. Coalescing preserves every byte and native READ completion/ACK. No model, sampling, scheduler-priority or cohort change.'))
    atexit.register(close)
