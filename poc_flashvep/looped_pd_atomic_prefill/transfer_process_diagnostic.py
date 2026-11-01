"""Check native two-process NIXL capability without a model computation."""
import argparse
import subprocess
import tempfile
import uuid
from collections import deque
from multiprocessing.connection import Listener,Client
from .common import *


def stage(cfg,role,address):
    import torch
    from nixl._api import nixl_agent,nixl_agent_config
    torch.set_num_threads(1);torch.cuda.set_device(0);channel=Client(address,family='AF_UNIX')
    tensor=torch.full((max(cfg['payload_bytes'])//2,),1.25 if role=='source' else 0.,dtype=torch.bfloat16,device='cuda:0')
    agent=nixl_agent('atomic_two_process_'+uuid.uuid4().hex,nixl_agent_config(backends=['UCX']))
    registration=agent.register_memory(tensor);peer=None;pending=deque()
    channel.send(dict(metadata=agent.get_agent_metadata(),pointer=tensor.data_ptr(),uuid=str(torch.cuda.get_device_properties(0).uuid),pid=os.getpid()))
    running=True
    try:
        while running:
            # Native D also calls get_new_notifs while it owns incoming work.
            agent.get_new_notifs()
            if not channel.poll():time.sleep(.0001);continue
            command=channel.recv();kind=command['kind']
            if kind=='connect':
                remote=command['remote'];peer=agent.add_remote_agent(remote['metadata'])
                if isinstance(peer,bytes):peer=peer.decode()
                channel.send(dict(kind='connected'));continue
            if kind=='stop':running=False;continue
            size=command['size']
            if kind=='prepare':tensor[:size//2].zero_();torch.cuda.synchronize();channel.send(dict(kind='prepared'));continue
            if kind=='verify':
                torch.cuda.synchronize();channel.send(dict(kind='verified',exact=bool(torch.all(tensor[:size//2]==1.25))));continue
            assert kind=='run' and role=='source'
            chunk=command['chunk'];inflight=command['inflight'];offset=0;live_bytes=0;fragments=0
            start=time.perf_counter();deadline=start+120
            while offset<size or pending:
                while offset<size and live_bytes<inflight:
                    amount=min(chunk,size-offset,inflight-live_bytes);segments=[];position=0
                    while position<amount:
                        length=min(1572864,amount-position);segments.append((position,length));position+=length
                    local=[(tensor.data_ptr()+offset+p,n,0) for p,n in segments]
                    target=[(remote['pointer']+offset+p,n,0) for p,n in segments]
                    handle=agent.initialize_xfer('WRITE',agent.get_xfer_descs(local,'VRAM'),agent.get_xfer_descs(target,'VRAM'),peer,backends=['UCX'])
                    pending.append((handle,amount));live_bytes+=amount;offset+=amount;fragments+=1
                    assert agent.transfer(handle)!='ERR'
                remaining=deque()
                for handle,amount in pending:
                    state=agent.check_xfer_state(handle);assert state!='ERR'
                    if state=='DONE':agent.release_xfer_handle(handle);live_bytes-=amount
                    else:remaining.append((handle,amount))
                pending=remaining;agent.get_new_notifs()
                assert time.perf_counter()<deadline,'Two-process transport watchdog'
            channel.send(dict(kind='finished',host_completion_s=time.perf_counter()-start,fragments=fragments))
    finally:
        if not pending:
            if peer is not None:agent.remove_remote_agent(peer)
            agent.deregister_memory(registration)
        channel.close()


def coordinator(cfg):
    folder=ROOT/'raw'/cfg['name'];write(folder/'status.json',dict(status='RUNNING',time=time.time(),pid=os.getpid()))
    children=[];channels={};logs=[]
    try:
        with tempfile.TemporaryDirectory(prefix='atomic_native_transport_') as temporary:
            ready={}
            for role,gpu in zip(('source','target'),cfg['gpus']):
                address=str(Path(temporary)/role);listener=Listener(address,family='AF_UNIX');listener._listener._socket.settimeout(120)
                log=(folder/(role+'.log')).open('w');logs.append(log)
                children.append(subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.transfer_process_diagnostic',
                    os.environ['ARC_CONFIG'],'--role',role,'--address',address],env=environment([gpu]),cwd=REPO,stdout=log,stderr=subprocess.STDOUT))
                channels[role]=listener.accept();listener.close();assert channels[role].poll(120)
                ready[role]=channels[role].recv()
                assert ready[role]['uuid'].removeprefix('GPU-')==UUIDS[gpu].removeprefix('GPU-')
            write(folder/'hardware.json',dict(gpus=cfg['gpus'],workers={r:{k:v for k,v in value.items() if k!='metadata'} for r,value in ready.items()},
                model_execution=False,process_geometry='Independent physical-GPU workers, each with local cuda:0'))
            for role,other in [('source','target'),('target','source')]:
                channels[role].send(dict(kind='connect',remote=ready[other]));assert channels[role].poll(30)
                assert channels[role].recv()['kind']=='connected'
            for size in cfg['payload_bytes']:
                for chunk,inflight in [(64*1024**2,256*1024**2),(64*1024**2,1024*1024**2),(256*1024**2,1024*1024**2)]:
                    for repeat in range(3):
                        channels['target'].send(dict(kind='prepare',size=size));assert channels['target'].poll(30)
                        assert channels['target'].recv()['kind']=='prepared'
                        channels['source'].send(dict(kind='run',size=size,chunk=chunk,inflight=inflight))
                        assert channels['source'].poll(150),'Source transfer response watchdog'
                        result=channels['source'].recv();assert result['kind']=='finished'
                        channels['target'].send(dict(kind='verify',size=size));assert channels['target'].poll(30)
                        check=channels['target'].recv();assert check['exact'],'Actual BF16 payload mismatch'
                        append(folder/'transport_samples.jsonl',dict(payload_bytes=size,fragment_bytes=chunk,max_inflight_bytes=inflight,
                            repeat=repeat,fragments=result['fragments'],host_completion_s=result['host_completion_s'],
                            effective_bytes_per_s=size/result['host_completion_s'],exact_payload_check=True,
                            evidence='MEASURED_TWO_PROCESS_NATIVE_NIXL_HOST_COMPLETION',
                            scope='Independent GPU contexts and active receiver progress. No model execution. Includes CPU metadata, submission, polling, and transfer. Not pure DMA or E2E speedup.'))
            write(folder/'status.json',dict(status='COMPLETE',time=time.time()))
    except BaseException as error:
        write(folder/'status.json',dict(status='FAILED',error=repr(error),time=time.time()));raise
    finally:
        for channel in channels.values():
            try:channel.send(dict(kind='stop'))
            except (BrokenPipeError,EOFError):pass
        for child in children:
            try:child.wait(30)
            except subprocess.TimeoutExpired:child.terminate();child.wait(10)
        for log in logs:log.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--role');parser.add_argument('--address');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.role:stage(cfg,args.role,args.address)
    else:coordinator(cfg)


if __name__=='__main__':main()
