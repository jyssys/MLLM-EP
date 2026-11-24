"""Exact local EC transport with producer ownership and clone-completion ACKs."""
import json
import os
from pathlib import Path
import re
import time
import urllib.request
from urllib.parse import urlparse
import uuid

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnector,ECExampleConnectorMetadata

from .common import ROOT,append,write
from .cuda_ipc_codec import encode_reduction_args,decode_reduction_args

REGISTRY=None
CONSUMER_CONNECTOR=None


def directory(root,key):
    assert isinstance(key,str) and re.fullmatch('[A-Za-z0-9_-]+',key),'Invalid EC identifier'
    path=Path(root)/key
    assert not path.is_symlink()
    return path


class ProducerRegistry:
    def __init__(self,root,max_storage_bytes=12*1024**3):
        self.root=Path(root);self.held={};self.ready={};self.exports={};self.max_storage_bytes=max_storage_bytes
        self.export_stream=None
        self.inline_role=None
        self.peak_storage_bytes=0;self.saved_count=0;self.released_count=0

    def storage_bytes(self):
        return sum({value.untyped_storage().data_ptr():value.untyped_storage().nbytes()
                    for value in self.held.values()}.values())

    def save(self,key,tensor):
        import torch
        assert key not in self.held,'Never overwrite a live E embedding'
        assert tensor.device.type=='cuda' and tensor.dtype==torch.bfloat16
        assert tensor.ndim==2 and tensor.shape[1]==3584
        path=directory(self.root,key);path.mkdir(parents=True,exist_ok=True)
        assert not (path/'cuda_ipc_ready.json').exists(),'Do not reuse an E export identifier'
        self.held[key]=tensor.detach()
        self.ready[key]=torch.cuda.Event();self.ready[key].record()
        size=self.storage_bytes();self.peak_storage_bytes=max(self.peak_storage_bytes,size)
        if size>self.max_storage_bytes:
            raise MemoryError('Live EC backing storage exceeds the frozen12GiB research bound. Do not free in-flight tensors.')
        self.saved_count+=1
        write(path/'cuda_ipc_ready.json',dict(key=key,producer_pid=os.getpid(),
            shape=list(tensor.shape),dtype=str(tensor.dtype),transport='cuda_ipc',
            source_original_E=True,scope='Marker only. Original GPU tensor remains owned by E until safe request completion.'))
        plan_file=path/'cuda_ipc_consumers.json'
        if plan_file.exists():
            plan=json.loads(plan_file.read_text())
            assert plan['key']==key and 0<plan['prefix_tokens']<plan['full_prompt_tokens']
            roles=plan['roles']
            assert roles and len(roles)==len(set(roles)) and set(roles)<={'prefill','decode'}
        else:
            roles=[] if self.inline_role is None else [self.inline_role]
        for role in roles:
            assert role in ('prefill','decode','pd')
            descriptor=self.export([key],role)
            write(path/f'cuda_ipc_inline_{role}.json',descriptor)

    def export(self,keys,consumer):
        import torch
        from torch.multiprocessing.reductions import reduce_tensor
        assert keys and len(keys)==len(set(keys))
        assert isinstance(consumer,str) and re.fullmatch('[A-Za-z0-9_-]+',consumer)
        assert all(key in self.held for key in keys),'Required live original E embedding is absent'
        token=uuid.uuid4().hex;self.exports[token]=dict(keys=set(keys),consumer=consumer)
        rows=[]
        if self.export_stream is None:self.export_stream=torch.cuda.Stream()
        # An export must wait for its own encoder result, not for unrelated
        # later encoder work already queued on the compute stream.
        with torch.cuda.stream(self.export_stream):
            for key in keys:
                torch.cuda.current_stream().wait_event(self.ready[key])
                function,args=reduce_tensor(self.held[key])
                assert function.__name__=='rebuild_cuda_tensor'
                rows.append(dict(key=key,args=encode_reduction_args(args)))
        return dict(producer_pid=os.getpid(),token=token,consumer=consumer,items=rows)

    def acknowledge(self,token):
        keys=self.exports.pop(token)['keys']
        return dict(acknowledged=True,images=len(keys),producer_pid=os.getpid())

    def drain_file_acks(self):
        for token,value in list(self.exports.items()):
            path=self.root/'_ipc_acks'/(token+'.json')
            if not path.exists():continue
            assert not path.is_symlink() and path.stat().st_uid==os.getuid()
            ack=json.loads(path.read_text())
            assert ack['token']==token and ack['producer_pid']==os.getpid()
            assert ack['consumer']==value['consumer'] and ack['clone_complete'] is True
            self.exports.pop(token)

    def release(self,keys):
        keys=set(keys)
        self.drain_file_acks()
        assert not any(keys & pending['keys'] for pending in self.exports.values()),'GPU clone completion ACK is still pending'
        for key in keys:
            path=directory(self.root,key)
            if key not in self.held:
                assert (path/'cuda_ipc_released.json').exists(),'Unknown E export cannot be released'
                continue
            self.held.pop(key);self.ready.pop(key);self.released_count+=1
            write(path/'cuda_ipc_released.json',dict(key=key,producer_pid=os.getpid(),time=time.time(),
                reason='Downstream request completed and every issued clone descriptor was acknowledged.'))
        return self.state()

    def state(self):
        self.drain_file_acks()
        return dict(producer_pid=os.getpid(),live_images=len(self.held),live_storage_bytes=self.storage_bytes(),
            peak_storage_bytes=self.peak_storage_bytes,outstanding_exports=len(self.exports),
            saved_count=self.saved_count,released_count=self.released_count,max_storage_bytes=self.max_storage_bytes)


class CUDAIPCEncoderConnector(ECExampleConnector):
    def __init__(self,vllm_config,role):
        super().__init__(vllm_config,role)
        global REGISTRY,CONSUMER_CONNECTOR
        extra=vllm_config.ec_transfer_config.ec_connector_extra_config
        self.producer_api=extra['producer_api']
        parsed=urlparse(self.producer_api)
        assert parsed.scheme=='http' and parsed.hostname=='127.0.0.1' and not parsed.path
        if self.is_producer and role==ECConnectorRole.WORKER:
            assert REGISTRY is None
            assert not vllm_config.compilation_config.cudagraph_mm_encoder,'Untested mutable encoder graph outputs are forbidden'
            REGISTRY=ProducerRegistry(self._storage_path)
        if self.is_consumer and role==ECConnectorRole.WORKER:
            assert CONSUMER_CONNECTOR is None
            self.ack_mode='rpc';CONSUMER_CONNECTOR=self

    def has_cache_item(self,identifier):
        path=directory(self._storage_path,identifier)
        found=(path/'cuda_ipc_ready.json').is_file() and not (path/'cuda_ipc_released.json').exists()
        if self.is_consumer and not found:
            raise FileNotFoundError('Missing live original E export. Local vision fallback is forbidden: '+identifier)
        return found

    def save_caches(self,encoder_cache,mm_hash,**kwargs):
        if not self.is_producer:return
        assert REGISTRY is not None
        REGISTRY.save(mm_hash,encoder_cache[mm_hash])

    def _rpc(self,operation,body):
        request=urllib.request.Request(self.producer_api+'/campaign/ec_ipc/'+operation,
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:
            result=json.load(response)
        assert len(result)==1,'This bounded transport expects TP1 E'
        return result[0]

    def start_load_caches(self,encoder_cache,**kwargs):
        import torch
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        metadata=self._get_connector_metadata()
        assert isinstance(metadata,ECExampleConnectorMetadata)
        keys=[item.mm_hash for item in metadata.mm_datas if item.mm_hash not in encoder_cache]
        if not keys:return
        records=[json.loads((directory(self._storage_path,key)/'cuda_ipc_ready.json').read_text()) for key in keys]
        assert len({r['producer_pid'] for r in records})==1
        pid=records[0]['producer_pid'];proc=Path('/proc',str(pid))
        assert proc.exists() and proc.stat().st_uid==os.getuid(),'The exact producer process must remain alive'
        begin=time.monotonic()
        role=os.environ['MLLM_EPD_ROLE'];groups=[];remaining=[];inline_images=0;rpc_ms=0.
        for key in keys:
            folder=directory(self._storage_path,key)
            inline=folder/f'cuda_ipc_inline_{role}.json'
            taken=folder/f'cuda_ipc_inline_{role}_taken.json'
            if inline.exists() and not taken.exists():
                data=json.loads(inline.read_text())
                assert data['consumer']==role and [r['key'] for r in data['items']]==[key]
                try:
                    with taken.open('x') as stream:json.dump(dict(consumer_pid=os.getpid(),time=time.time()),stream)
                except FileExistsError:remaining.append(key)
                else:groups.append(data);inline_images+=1
            else:remaining.append(key)
        if remaining:
            sent=time.monotonic()
            groups.append(self._rpc('export',dict(keys=remaining,consumer=f'{role}_{os.getpid()}')))
            rpc_ms=(time.monotonic()-sent)*1000
        exported=time.monotonic()
        assert all(group['producer_pid']==pid for group in groups)
        items={item['key']:item for group in groups for item in group['items']}
        assert set(items)==set(keys) and sum(len(group['items']) for group in groups)==len(keys)
        views=[]
        for key,record in zip(keys,records):
            item=items[key]
            received=rebuild_cuda_tensor(*decode_reduction_args(item['args']))
            assert list(received.shape)==record['shape'] and str(received.dtype)==record['dtype']
            views.append(received)
            encoder_cache[item['key']]=received.clone()
        complete=torch.cuda.Event();complete.record();complete.synchronize()
        del received
        views.clear()
        cloned=time.monotonic();ack_mode=getattr(self,'ack_mode','rpc')
        for data in groups:
            if ack_mode=='rpc':
                acknowledged=self._rpc('ack',dict(token=data['token']))
                assert acknowledged['acknowledged'] and acknowledged['producer_pid']==pid
            else:
                assert ack_mode=='file'
                path=Path(self._storage_path)/'_ipc_acks'/(data['token']+'.json')
                assert re.fullmatch('[a-f0-9]{32}',data['token']) and not path.exists()
                write(path,dict(token=data['token'],producer_pid=pid,consumer=data['consumer'],
                    clone_complete=True,time=time.time()))
        ended=time.monotonic()
        append(ROOT/'raw'/f'ec_cuda_ipc_{os.environ["MLLM_EPD_ROLE"]}_gpu{os.environ["CUDA_VISIBLE_DEVICES"]}.jsonl',
            dict(begin=begin,end=ended,images=len(keys),keys=keys,producer_pid=pid,consumer_pid=os.getpid(),
                export_RPC_ms=rpc_ms,descriptor_prepare_host_ms=(exported-begin)*1000,
                inline_images=inline_images,import_clone_sync_host_ms=(cloned-exported)*1000,
                completion_ACK_host_ms=(ended-cloned)*1000,ack_mode=ack_mode,
                logical_bytes=sum(encoder_cache[key].numel()*encoder_cache[key].element_size() for key in keys),
                clone_complete_before_ACK=True,scope='Original E tensor transport host enclosure, not exclusive GPU cost.'))
