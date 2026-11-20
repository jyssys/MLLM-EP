"""Transfer variable-length raw hidden/residual with explicit ownership."""
import json
import os
import time
from .common import write,digest


class HiddenTransport:
    def __init__(self,cfg,role,folder):
        import torch
        from nixl._api import nixl_agent,nixl_agent_config
        self.cfg,self.role,self.folder=cfg,role,folder
        self.directory=folder/'hidden_transport';self.directory.mkdir(exist_ok=True)
        count=cfg.get('hidden_source_slots',2) if role=='prefill' else cfg.get('hidden_destination_slots',1)
        self.pool=torch.empty((count,2,38912,4096),device='cuda',dtype=torch.bfloat16)
        self.owners=[None]*count;self.parts={};self.received={};self.retired={};self.remotes={};self.rows=[]
        self.agent=nixl_agent('BI0-hidden-'+role+'-'+str(os.getpid()),nixl_agent_config(backends=['UCX'],capture_telemetry=True))
        self.registration=self.agent.register_memory(self.pool)
        self.metadata=self.directory/f'{role}_agent.bin';self.metadata.write_bytes(self.agent.get_agent_metadata())

    def reap(self):
        if self.role=='prefill':
            for index,rid in enumerate(self.owners):
                if rid is None:continue
                path=self.directory/(rid+'.ACK.json')
                if not path.exists():continue
                value=json.loads(path.read_text())
                assert value['slot']==index and value['request_id']==rid and value['bytes']==self.parts[rid]['length']*16384
                self.owners[index]=None;del self.parts[rid]
        else:
            for rid,(index,event) in list(self.retired.items()):
                if event.query():self.owners[index]=None;del self.retired[rid]

    def take(self,rid):
        self.reap();assert rid not in self.owners
        index=next((i for i,value in enumerate(self.owners) if value is None),None)
        assert index is not None,'HIDDEN_CREDIT_EXHAUSTED'
        self.owners[index]=rid;return index

    def put_part(self,rid,start,length,hidden,residual,cut):
        import torch
        assert self.role=='prefill' and hidden.shape==residual.shape and hidden.shape[1]==4096
        if rid not in self.parts:
            assert start==0
            self.parts[rid]=dict(slot=self.take(rid),cursor=0,length=length)
        part=self.parts[rid];assert part['cursor']==start
        count=len(hidden);slot=part['slot'];assert start+count<=length<=38912
        self.pool[slot,0,start:start+count].copy_(hidden)
        self.pool[slot,1,start:start+count].copy_(residual)
        part['cursor']+=count
        if part['cursor']==length:
            nbytes=length*4096*2
            write(self.directory/(rid+'.json'),dict(version=1,request_id=rid,slot=slot,length=length,cut=cut,
                bytes=2*nbytes,pointers=[self.pool[slot,i].data_ptr() for i in (0,1)],
                segment_bytes=nbytes,metadata=str(self.metadata),metadata_SHA256=digest(self.metadata),
                source_UUID=str(torch.cuda.get_device_properties(0).uuid)))
        self.rows.append(dict(event='P_PART_COPY_SUBMITTED',request_id=rid,start=start,count=count,length=length,
            time=time.monotonic(),scope='The producer CPU sample completes before this metadata is advertised.'))

    def get(self,rid,cut):
        import torch
        if rid in self.received:
            slot,length=self.received[rid]
            return self.pool[slot,0,:length],self.pool[slot,1,:length]
        assert self.role=='decode'
        value=json.loads((self.directory/(rid+'.json')).read_text())
        assert value['version']==1 and value['request_id']==rid and value['cut']==cut
        assert value['source_UUID']!=str(torch.cuda.get_device_properties(0).uuid)
        slot=self.take(rid);length=value['length'];assert value['bytes']==length*16384
        from pathlib import Path
        metadata=Path(value['metadata']);assert digest(metadata)==value['metadata_SHA256']
        if str(metadata) not in self.remotes:self.remotes[str(metadata)]=self.agent.add_remote_agent(metadata.read_bytes())
        local=self.agent.get_xfer_descs([(self.pool[slot,i].data_ptr(),value['segment_bytes'],0) for i in (0,1)],'VRAM')
        remote=self.agent.get_xfer_descs([(pointer,value['segment_bytes'],0) for pointer in value['pointers']],'VRAM')
        started=time.monotonic();handle=self.agent.initialize_xfer('READ',local,remote,self.remotes[str(metadata)],backends=['UCX'])
        try:
            status=self.agent.transfer(handle)
            while status!='DONE':
                assert status!='ERR' and time.monotonic()-started<30
                status=self.agent.check_xfer_state(handle)
                if status!='DONE':time.sleep(.00005)
            telemetry=self.agent.get_xfer_telemetry(handle);assert telemetry.totalBytes==value['bytes']
            self.received[rid]=(slot,length)
            write(self.directory/(rid+'.ACK.json'),dict(request_id=rid,slot=value['slot'],bytes=value['bytes'],time=time.monotonic()))
            self.rows.append(dict(event='D_HIDDEN_READ_DONE',request_id=rid,bytes=value['bytes'],
                host_ms=1000*(time.monotonic()-started),NIXL_post_us=telemetry.postDuration,
                NIXL_xfer_us=telemetry.xferDuration,time=time.monotonic()))
            return self.pool[slot,0,:length],self.pool[slot,1,:length]
        finally:self.agent.release_xfer_handle(handle)

    def retire(self,rid):
        import torch
        slot,_=self.received.pop(rid);done=torch.cuda.Event();done.record();self.retired[rid]=(slot,done)

    def close(self):
        import torch
        torch.cuda.synchronize();self.reap()
        passed=not self.parts and not self.received and not self.retired and all(x is None for x in self.owners)
        write(self.folder/f'hidden_lifetime_{self.role}.json',dict(passed=passed,owners=self.owners,rows=self.rows,
            scope='Source reuse follows completed READ ACK. Destination reuse follows a completed suffix CUDA event. Normal completion only.'))
        assert passed,'Do not free a live hidden slot'
        for remote in self.remotes.values():self.agent.remove_remote_agent(remote)
        self.agent.deregister_memory(self.registration)
