"""Measure the native NIXL transport without model computation or fake KV."""
import uuid
from collections import deque
from .common import *


def main():
    cfg=config();folder=ROOT/'raw'/cfg['name'];write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
    import torch
    from nixl._api import nixl_agent,nixl_agent_config
    agents=[];registrations=[];pending=deque();max_bytes=max(cfg['payload_bytes']);source=None;target=None
    try:
        source=torch.full((max_bytes//2,),1.25,dtype=torch.bfloat16,device='cuda:0')
        target=torch.zeros((max_bytes//2,),dtype=torch.bfloat16,device='cuda:1')
        for device,tensor in enumerate((source,target)):
            torch.cuda.set_device(device)
            agent=nixl_agent('atomic_transport_'+uuid.uuid4().hex,nixl_agent_config(backends=['UCX']))
            agents.append(agent);registrations.append(agent.register_memory(tensor))
        torch.cuda.set_device(0);peer=agents[0].add_remote_agent(agents[1].get_agent_metadata())
        if isinstance(peer,bytes):peer=peer.decode()
        for gpu in (0,1):torch.cuda.synchronize(gpu)
        for size in cfg['payload_bytes']:
            for chunk,inflight in [(64*1024**2,256*1024**2),(64*1024**2,1024*1024**2),(256*1024**2,1024*1024**2)]:
                for repeat in range(3):
                    target[:size//2].zero_();torch.cuda.synchronize(1);torch.cuda.set_device(0)
                    offset=0;live_bytes=0;fragments=0;start=time.perf_counter();deadline=start+120
                    while offset<size or pending:
                        while offset<size and live_bytes<inflight:
                            amount=min(chunk,size-offset,inflight-live_bytes)
                            # Match the order of descriptors from full physical
                            # blocks. This is not a Looped-model inference run.
                            segments=[];position=0;block=1572864
                            while position<amount:
                                n=min(block,amount-position);segments.append((position,n));position+=n
                            local=[(source.data_ptr()+offset+p,n,0) for p,n in segments]
                            remote=[(target.data_ptr()+offset+p,n,1) for p,n in segments]
                            handle=agents[0].initialize_xfer('WRITE',agents[0].get_xfer_descs(local,'VRAM'),
                                agents[0].get_xfer_descs(remote,'VRAM'),peer,backends=['UCX'])
                            pending.append((handle,amount));live_bytes+=amount;offset+=amount;fragments+=1
                            assert agents[0].transfer(handle)!='ERR'
                        rest=deque()
                        for handle,amount in pending:
                            state=agents[0].check_xfer_state(handle);assert state!='ERR'
                            if state=='DONE':agents[0].release_xfer_handle(handle);live_bytes-=amount
                            else:rest.append((handle,amount))
                        pending=rest
                        assert time.perf_counter()<deadline,'Standalone transport watchdog'
                    elapsed=time.perf_counter()-start;torch.cuda.synchronize(1)
                    assert bool(torch.all(target[:size//2]==1.25)),'Transport BF16 payload mismatch'
                    append(folder/'transport_samples.jsonl',dict(payload_bytes=size,fragment_bytes=chunk,max_inflight_bytes=inflight,
                        repeat=repeat,fragments=fragments,host_completion_s=elapsed,effective_bytes_per_s=size/elapsed,
                        exact_payload_check=True,evidence='MEASURED_STANDALONE_NIXL_HOST_COMPLETION',
                        scope='Original NIXL UCX backend, no model execution. Includes metadata, submission, polling, and transfer. Not pure GPU DMA or an E2E speedup.'))
        write(folder/'hardware.json',dict(gpus=cfg['gpus'],UUIDs=[str(torch.cuda.get_device_properties(i).uuid) for i in (0,1)],
            model_computation=False,buffer_dtype='torch.bfloat16',physical_block_descriptor_bytes=1572864))
        write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        if not pending:
            if len(agents)==2:
                try:agents[0].remove_remote_agent(peer)
                except (UnboundLocalError,RuntimeError):pass
            for agent,registration in zip(agents,registrations):agent.deregister_memory(registration)


if __name__=='__main__':main()
