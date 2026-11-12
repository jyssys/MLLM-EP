"""Persistent GPU buffers and explicit ACK for exact Hidden/Residual READs."""
import json,os,time
from pathlib import Path
from .common import write,digest


class Slots:
    def __init__(self,count):self.owners=[None]*count
    def free_count(self):return sum(owner is None for owner in self.owners)
    def take(self,request):
        assert request not in self.owners,'A request cannot own duplicate slots'
        for i,owner in enumerate(self.owners):
            if owner is None:self.owners[i]=request;return i
        raise RuntimeError('HIDDEN_BUFFER_CREDIT_EXHAUSTED')
    def release(self,index,request):
        assert self.owners[index]==request,'Stale Hidden ACK or wrong owner'
        self.owners[index]=None


class GPUHiddenTransport:
    def __init__(self,cfg,role,folder,event):
        import torch
        from nixl._api import nixl_agent,nixl_agent_config
        self.cfg=cfg;self.role=role;self.folder=folder;self.event=event
        self.directory=folder/'state_transport';self.directory.mkdir(exist_ok=True)
        count=cfg.get('hidden_source_slots',32) if role=='prefill' else cfg.get('hidden_destination_slots',4)
        self.pool=torch.empty((count,2,cfg['length'],4096),dtype=torch.bfloat16,device='cuda')
        self.slots=Slots(count);self.received={};self.retirement={};self.remotes={};self.closed=False
        self.agent=nixl_agent('dense-hidden-'+role+'-'+str(os.getpid()),nixl_agent_config(backends=['UCX'],capture_telemetry=True))
        self.registration=self.agent.register_memory(self.pool)
        meta=self.directory/f'hidden_agent_{role}.bin';meta.write_bytes(self.agent.get_agent_metadata())
        self.metadata=meta;self.slot_bytes=self.pool[0].numel()*self.pool.element_size()
        event('HIDDEN_GPU_POOL',slots=count,bytes=self.pool.numel()*self.pool.element_size(),
            slot_bytes=self.slot_bytes,metadata_SHA256=digest(meta),
            source_UUID=str(torch.cuda.get_device_properties(0).uuid),backend='NIXL_UCX')

    def reap(self):
        if self.role=='prefill':
            for i,rid in enumerate(list(self.slots.owners)):
                if rid is None:continue
                p=self.directory/(rid+'.hidden_ack.json')
                if not p.exists():continue
                value=json.loads(p.read_text());assert value['request_id']==rid and value['slot']==i
                assert value['copied_bytes']==self.slot_bytes and value['version']==1
                self.slots.release(i,rid);self.event('HIDDEN_SOURCE_RELEASED',request_id=rid,slot=i)
        else:
            for rid,(slot,done) in list(self.retirement.items()):
                if done.query():
                    self.slots.release(slot,rid);del self.retirement[rid]

    def put(self,rid,hidden,residual,cut):
        import torch
        assert self.role=='prefill' and Path(rid).name==rid
        assert hidden.shape==residual.shape==(self.cfg['length'],4096)
        self.reap();slot=self.slots.take(rid);start=time.monotonic()
        self.pool[slot,0].copy_(hidden);self.pool[slot,1].copy_(residual)
        value=dict(version=1,request_id=rid,slot=slot,cut=cut,tokens=hidden.shape[0],
            pointer=self.pool[slot].data_ptr(),bytes=self.slot_bytes,device=0,
            source_UUID=str(torch.cuda.get_device_properties(0).uuid),metadata_file=str(self.metadata),
            metadata_SHA256=digest(self.metadata),dtype='bfloat16',hidden_shape=list(hidden.shape))
        if self.cfg.get('hidden_verify_hash'):
            from .extension import tensor_hash
            value.update(hidden_SHA256=tensor_hash(self.pool[slot,0]),residual_SHA256=tensor_hash(self.pool[slot,1]))
        write(self.directory/(rid+'.gpu.json'),value)
        self.event('HIDDEN_GPU_EXPORTED',request_id=rid,slot=slot,bytes=self.slot_bytes,
            enqueue_host_ms=1000*(time.monotonic()-start),
            readiness='Copy precedes the Native producer sample on its current CUDA stream. The coordinator advertises this descriptor only after that CPU sample completes.')

    def get(self,rid,cut):
        import torch
        assert self.role=='decode' and Path(rid).name==rid
        value=json.loads((self.directory/(rid+'.gpu.json')).read_text())
        assert value['version']==1 and value['request_id']==rid and value['cut']==cut
        assert value['tokens']==self.cfg['length'] and value['dtype']=='bfloat16'
        assert value['bytes']==self.slot_bytes
        assert value['source_UUID']!=str(torch.cuda.get_device_properties(0).uuid)
        self.reap();slot=self.slots.take(rid);start=time.monotonic()
        meta=Path(value['metadata_file']);assert digest(meta)==value['metadata_SHA256']
        if str(meta) not in self.remotes:self.remotes[str(meta)]=self.agent.add_remote_agent(meta.read_bytes())
        local=self.agent.get_xfer_descs([(self.pool[slot].data_ptr(),self.slot_bytes,0)],'VRAM')
        remote=self.agent.get_xfer_descs([(value['pointer'],self.slot_bytes,value['device'])],'VRAM')
        handle=self.agent.initialize_xfer('READ',local,remote,self.remotes[str(meta)],backends=['UCX'])
        submitted=time.monotonic();status=self.agent.transfer(handle)
        try:
            while status!='DONE':
                assert status!='ERR','Hidden NIXL READ failed'
                assert time.monotonic()-start<30,'Hidden NIXL READ timed out'
                status=self.agent.check_xfer_state(handle)
                if status!='DONE':time.sleep(.00005)
            completed=time.monotonic()
            telemetry=self.agent.get_xfer_telemetry(handle)
            measured=dict(NIXL_xfer_duration_us=telemetry.xferDuration,NIXL_post_duration_us=telemetry.postDuration,
                NIXL_xfer_duration_ms=telemetry.xferDuration/1e3,
                NIXL_post_duration_ms=telemetry.postDuration/1e3,NIXL_total_bytes=telemetry.totalBytes,
                telemetry_unit_contract='NIXL 1.4.1 Python API documents time fields in microseconds.')
            assert telemetry.totalBytes==self.slot_bytes
            if self.cfg.get('hidden_verify_hash'):
                from .extension import tensor_hash
                assert tensor_hash(self.pool[slot,0])==value['hidden_SHA256']
                assert tensor_hash(self.pool[slot,1])==value['residual_SHA256']
            write(self.directory/(rid+'.hidden_ack.json'),dict(version=1,request_id=rid,slot=value['slot'],
                copied_bytes=self.slot_bytes,copy_complete=time.monotonic(),destination_slot=slot))
            self.received[rid]=slot
            self.event('HIDDEN_GPU_IMPORTED',request_id=rid,slot=slot,bytes=self.slot_bytes,
                preparation_host_ms=1000*(submitted-start),transfer_request_host_ms=1000*(completed-submitted),
                **measured,
                scope='NIXL submit-to-DONE host enclosure. It is not exclusive wire time or a CUDA kernel duration.')
            return self.pool[slot,0],self.pool[slot,1]
        finally:self.agent.release_xfer_handle(handle)

    def retire(self,source_ids):
        import torch
        for rid in source_ids:
            slot=self.received.pop(rid);done=torch.cuda.Event();done.record()
            self.retirement[rid]=(slot,done)

    def close(self):
        import torch
        if self.closed:return
        torch.cuda.synchronize();self.reap()
        assert not self.received and not self.retirement,'Destination Hidden buffers are still live'
        pending=[r for r in self.slots.owners if r is not None]
        self.event('HIDDEN_GPU_POOL_CLOSE',pending_requests=pending,all_ACKed=not pending)
        assert not pending,'Source Hidden ACK is missing; do not silently reuse memory'
        for name in self.remotes.values():self.agent.remove_remote_agent(name)
        self.agent.deregister_memory(self.registration);self.closed=True
