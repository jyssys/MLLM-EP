"""Test same-node CUDA IPC before considering any serving connector change."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import time

from .common import ROOT,REPO,PYTHON,resources,write,digest,preserve


def wait_file(path,seconds=150):
    deadline=time.time()+seconds
    while not path.exists():
        if time.time()>deadline:raise TimeoutError('IPC peer observation expired; inspect both owned processes')
        time.sleep(.01)
    return json.loads(path.read_text())


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def cases(feature):
    import torch
    torch.manual_seed(2026112406)
    base=torch.randn((4096,3584),dtype=torch.bfloat16)
    values=[('synthetic_128_view',base[17:145]),('synthetic_512',base[:512].clone()),
            ('synthetic_2048_view',base[63:2111])]
    if feature:
        from safetensors.torch import load_file
        value=load_file(str(feature))['embedding']
        assert value.dtype==torch.bfloat16 and value.ndim==2 and value.shape[1]==3584
        values.append(('replayed_real_encoder_embedding',value))
    return values


def case_path(folder,index,peer,count,suffix):
    stem=f'case_{index}' if count==1 else f'case_{index}_peer{peer}'
    return folder/(stem+suffix)


def producer(folder,feature,consumer_count=1,async_ready=False):
    import torch
    from torch.multiprocessing.reductions import reduce_tensor
    rows=cases(feature)
    write(folder/'case_count.json',dict(count=len(rows),source_gpu=0,consumer_count=consumer_count,
        feature=str(feature) if feature else None,feature_SHA256=digest(feature) if feature else None,
        scope='Synthetic protocol fixtures and optional replayed actual E output. Not online image encoding.'))
    if async_ready:
        for peer in range(consumer_count):wait_file(folder/f'consumer_{peer}_ready.json')
    for index,(name,cpu) in enumerate(rows):
        expected=tensor_sha(cpu) if async_ready else None
        # Retain a nonzero storage offset in the IPC view cases.
        if name.endswith('_view'):
            base=torch.zeros((len(cpu)+31,3584),dtype=cpu.dtype,device='cuda')
            tensor=base[19:19+len(cpu)]
        else:base=cpu.to('cuda');tensor=base
        if async_ready:
            pinned=cpu.pin_memory()
            torch.cuda._sleep(500_000_000)
            tensor.copy_(pinned,non_blocking=True)
            ready=torch.cuda.Event();ready.record()
        else:
            if name.endswith('_view'):tensor.copy_(cpu)
            expected=tensor_sha(tensor)
        for peer in range(consumer_count):
            begin=time.monotonic()
            # A fresh export creates its own receiver reference-count contract.
            # Never send the same reduction descriptor to two receivers.
            function,args=reduce_tensor(tensor.detach())
            assert function.__name__=='rebuild_cuda_tensor' and len(args)==15
            payload=pickle.dumps(args,protocol=5)
            target=case_path(folder,index,peer,consumer_count,'.ipc_args.pickle')
            with target.open('xb') as stream:stream.write(payload)
            write(case_path(folder,index,peer,consumer_count,'.json'),dict(name=name,peer=peer,
                shape=list(tensor.shape),stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),dtype=str(tensor.dtype),expected_SHA256=expected,
                payload_SHA256=digest(target),source_current_device=torch.cuda.current_device(),
                source_device_name=torch.cuda.get_device_name(),source_pid=os.getpid(),producer_retains_until_ack=True,
                source_ready_at_export=bool(ready.query()) if async_ready else True,
                export_host_ms=(time.monotonic()-begin)*1000,event_sync_required=bool(args[-1]),
                async_readiness_probe=async_ready))
        for peer in range(consumer_count):
            acknowledgement=wait_file(case_path(folder,index,peer,consumer_count,'_ack.json'))
            assert acknowledgement['passed'],acknowledgement
        del tensor,base,args
        gc.collect();torch.cuda.ipc_collect()
    write(folder/'producer_complete.json',dict(time=time.time(),passed=True,count=len(rows)))


def consumer(folder,peer=0):
    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    torch.cuda.init()
    torch.empty(1,device='cuda');torch.cuda.synchronize()
    write(folder/f'consumer_{peer}_ready.json',dict(pid=os.getpid(),ready=True,peer=peer))
    info=wait_file(folder/'case_count.json');results=[];count=info.get('consumer_count',1)
    assert 0<=peer<count
    for index in range(info['count']):
        description=wait_file(case_path(folder,index,peer,count,'.json'))
        path=case_path(folder,index,peer,count,'.ipc_args.pickle')
        assert path.stat().st_uid==os.getuid() and not path.is_symlink()
        assert digest(path)==description['payload_SHA256']
        # The pickle contains only the trusted producer's Torch reduction arguments.
        # It is not an external dataset or a remotely supplied model artifact.
        args=pickle.loads(path.read_bytes())
        begin=time.monotonic()
        received=rebuild_cuda_tensor(*args)
        opened=time.monotonic()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();local=received.clone();end.record();end.synchronize()
        actual=tensor_sha(local)
        proof=dict(name=description['name'],peer=peer,shape=list(local.shape),dtype=str(local.dtype),
            source_storage_offset=description['storage_offset'],consumer_device=str(received.device),
            consumer_current_device=torch.cuda.current_device(),consumer_device_name=torch.cuda.get_device_name(),
            expected_SHA256=description['expected_SHA256'],actual_SHA256=actual,
            open_host_ms=(opened-begin)*1000,clone_GPU_enclosure_ms=start.elapsed_time(end),
            source_ready_at_export=description.get('source_ready_at_export',True),
            producer_event_sync_required=description.get('event_sync_required'),
            export_host_ms=description.get('export_host_ms'),
            passed=actual==description['expected_SHA256'],
            scope='Isolated CUDA IPC protocol. Clone event enclosure is not exclusive kernel time or online speedup.')
        del received,args
        gc.collect();torch.cuda.synchronize();torch.cuda.ipc_collect()
        write(case_path(folder,index,peer,count,'_ack.json'),proof);results.append(proof)
        assert proof['passed'],proof
    target=folder/'consumer_complete.json' if count==1 else folder/f'consumer_{peer}_complete.json'
    write(target,dict(time=time.time(),passed=True,rows=results))


def guardian(name,feature,consumer_gpus=(1,),async_ready=False):
    preserve();folder=ROOT/'raw'/name
    assert not folder.exists(),'Do not overwrite a GPU protocol probe'
    folder.mkdir()
    gpus=[0,*consumer_gpus]
    assert len(gpus)==len(set(gpus)) and set(gpus)<=set(range(8))
    owner=resources();children=[];success=False;started=time.time()
    from .source_snapshot import snapshot
    archive=snapshot(name)
    write(ROOT/'configs'/f'{name}.json',dict(time=started,GPUs=gpus,feature=str(feature) if feature else None,
        asynchronous_producer_readiness=async_ready,independent_descriptor_per_consumer=True,
        GPU_info=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,gpus)),
            '--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True),
        source_archive_SHA256=archive['archive_SHA256'],evidence='CORRECTNESS_ONLY',
        hypothesis='A P2P-accessible CUDA allocation may preserve exact E embeddings without synchronous CPU staging.',
        mandatory='Keep the producer tensor alive until consumer clone, synchronization and IPC-view release finish. '
            'This does not yet implement an EC connector or encoder-safe extra prefill.'))
    try:
        owner.stop(gpus)
        for role,gpu,peer in [('producer',0,0)]+[('consumer',gpu,peer) for peer,gpu in enumerate(consumer_gpus)]:
            command=[str(PYTHON),'-m',__package__+'.gpu_ipc_protocol_probe','--role',role,'--name',name]
            command+=['--consumer-gpus',','.join(map(str,consumer_gpus)),'--consumer-index',str(peer)]
            if async_ready:command+=['--async-ready']
            if feature:command+=['--feature',str(feature)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            label=role if role=='producer' or len(consumer_gpus)==1 else f'consumer{peer}'
            with (ROOT/'logs'/f'{name}_{label}.log').open('xb') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(dict(role=label,gpu=gpu,pid=child.pid,sid=child.pid,process=child))
        write(ROOT/'state'/f'{name}_processes.json',dict(time=time.time(),children=[{k:v for k,v in r.items() if k!='process'} for r in children]))
        while any(r['process'].poll() is None for r in children):
            if any(r['process'].poll() not in (None,0) for r in children):break
            if time.time()-started>240:raise TimeoutError('Inspect both owned IPC processes; no unrelated process is a target')
            time.sleep(.5)
        success=all(r['process'].poll()==0 for r in children)
        assert success,{r['role']:r['process'].poll() for r in children}
        actual=[]
        for peer in range(len(consumer_gpus)):
            target=folder/'consumer_complete.json' if len(consumer_gpus)==1 else folder/f'consumer_{peer}_complete.json'
            value=json.loads(target.read_text());assert value['passed'];actual.extend(value['rows'])
        write(ROOT/'analysis'/f'{name}.json',dict(time=time.time(),evidence='CORRECTNESS_ONLY',status='PASS',
            rows=actual,GPUs=gpus,scope='Same-node cross-process CUDA IPC. Synthetic and replayed real E tensors. '
                'No new token split, online serving or SLO-safe insertion was tested.'))
    finally:
        for row in children:
            child=row['process']
            if child.poll() is None:
                proc=Path('/proc',str(child.pid))
                assert proc.stat().st_uid==os.getuid() and os.getsid(child.pid)==row['sid']
                assert b'gpu_ipc_protocol_probe' in (proc/'cmdline').read_bytes()
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        until=time.time()+45
        while any(r['gpu'] in gpus for r in owner.processes()) and time.time()<until:time.sleep(.5)
        owner.restore([gpu for gpu in gpus if not any(r['gpu']==gpu for r in owner.processes())])
        write(ROOT/'state'/f'{name}_complete.json',dict(time=time.time(),success=success,
            child_returncodes={r['role']:r['process'].poll() for r in children},restoration_attempted=True))
        preserve()


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['guardian','producer','consumer'],default='guardian')
    p.add_argument('--name',required=True);p.add_argument('--feature',type=Path)
    p.add_argument('--consumer-gpus',default='1');p.add_argument('--consumer-index',type=int,default=0)
    p.add_argument('--async-ready',action='store_true');args=p.parse_args()
    assert args.name.replace('_','').isalnum()
    folder=ROOT/'raw'/args.name
    gpus=tuple(map(int,args.consumer_gpus.split(',')))
    if args.role=='guardian':guardian(args.name,args.feature,gpus,args.async_ready)
    elif args.role=='producer':producer(folder,args.feature,len(gpus),args.async_ready)
    else:consumer(folder,args.consumer_index)


if __name__=='__main__':main()
