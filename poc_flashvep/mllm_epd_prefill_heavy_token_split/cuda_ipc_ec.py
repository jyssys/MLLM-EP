"""Exact local EC transport with producer ownership and clone-completion ACKs."""
from dataclasses import dataclass
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

from ..mllm_epd_h200_dynamic_correctness_slice_aware.embedding_rows import (
    SLICE_KEY, SlicedEncoderRows, suffix_rows,
)
from .common import ROOT,append,write
from .crossing_ipc import (
    CROSSING_PLAN, CROSSING_ROLE, PREISSUE_ENV, TORCH_CUDA_IPC_REVISION,
    cancellation_marker, claim_crossing_revoke, claim_inline, decode_inline_roles,
    inline_paths, try_mark,
)
from .cuda_ipc_codec import encode_reduction_args,decode_reduction_args
from .token_split import split_plan

REGISTRY=None
CONSUMER_CONNECTOR=None


@dataclass
class SliceAwareECMetadata(ECExampleConnectorMetadata):
    rows_by_key: dict[str, tuple[int, int]]


def directory(root,key):
    assert isinstance(key,str) and re.fullmatch('[A-Za-z0-9_-]+',key),'Invalid EC identifier'
    path=Path(root)/key
    assert not path.is_symlink()
    return path


def release_unused_cuda_reference(value):
    # A fresh reduce_tensor export has one ref-count slot. This is the very
    # same decrement used by rebuild_cuda_tensor for an unneeded duplicate
    # descriptor, without opening its CUDA allocation or IPC event.
    args = decode_reduction_args(value)
    args[4]._release_ipc_counter(args[11], args[12], device=args[6])


class ProducerRegistry:
    def __init__(self,root,max_storage_bytes=12*1024**3):
        self.root=Path(root);self.held={};self.ready={};self.exports={};self.max_storage_bytes=max_storage_bytes
        self.export_stream=None
        self.inline_role=None
        self.capture_correctness_hashes=os.environ.get('MLLM_EPD_LATE_VERIFY_E_SERIAL')=='1'
        self.crossing_preissue=os.environ.get(PREISSUE_ENV)=='1'
        if self.crossing_preissue:
            import torch

            if torch.version.git_version != TORCH_CUDA_IPC_REVISION:
                raise RuntimeError('Crossing descriptor retirement requires the audited Torch CUDA IPC revision')
        self.crossing_issued=0;self.crossing_acknowledged=0;self.crossing_revoked=0
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
        ready=dict(key=key,producer_pid=os.getpid(),
            shape=list(tensor.shape),dtype=str(tensor.dtype),transport='cuda_ipc',
            source_original_E=True,scope='Marker only. Original GPU tensor remains owned by E until safe request completion.')
        if self.capture_correctness_hashes:
            from .batch_prefix_probe import tensor_signature
            ready['original_embedding_signature']=tensor_signature(tensor)
        write(path/'cuda_ipc_ready.json',ready)
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
        if self.crossing_preissue and roles == ['prefill'] and not plan_file.exists():
            self.preissue_crossing(key, path)

    def preissue_crossing(self, key, folder):
        candidate = folder / CROSSING_PLAN
        if not candidate.is_file():
            return
        plan = json.loads(candidate.read_text())
        if plan["key"] != key or not 0 <= plan["start"] < plan["end"] <= plan["prompt_tokens"]:
            raise ValueError("Unexpected crossing candidate ownership")
        request_key = plan["request_key"]
        if cancellation_marker(self.root, request_key).exists():
            return
        descriptor = self.export([key], CROSSING_ROLE)
        token = descriptor["token"]
        entry = self.exports[token]
        entry.update(crossing_request=request_key, crossing_args=descriptor["items"][0]["args"])
        self.crossing_issued += 1
        try:
            write(inline_paths(folder, CROSSING_ROLE)[0], descriptor)
        except BaseException:
            self.revoke_crossing_descriptor(token, entry)
            raise
        if cancellation_marker(self.root, request_key).exists():
            self.revoke_crossing_descriptor(token, entry)

    def revoke_crossing_descriptor(self, token, entry):
        assert entry["consumer"] == CROSSING_ROLE and len(entry["keys"]) == 1
        folder = directory(self.root, next(iter(entry["keys"])))
        if not claim_crossing_revoke(folder, token):
            return False
        # The marker is permanent even on failure. A native destructor or a
        # late consumer must never decrement this descriptor a second time.
        release_unused_cuda_reference(entry["crossing_args"])
        self.exports.pop(token)
        self.crossing_revoked += 1
        return True

    def cancel_crossing(self, request_key):
        if not self.crossing_preissue:
            raise RuntimeError("Crossing descriptor preissue is disabled")
        marker = cancellation_marker(self.root, request_key)
        marker.parent.mkdir(mode=0o700, exist_ok=True)
        try_mark(marker, dict(request_key=request_key, producer_pid=os.getpid(), time=time.time()))
        self.drain_file_acks()
        candidates = [(token, entry) for token, entry in self.exports.items()
                      if entry.get("crossing_request") == request_key]
        revoked = sum(self.revoke_crossing_descriptor(token, entry) for token, entry in candidates)
        return dict(request_key=request_key, revoked=revoked, still_claimed=len(candidates) - revoked)

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
        entry = self.exports[token]
        if "crossing_request" in entry:
            folder = directory(self.root, next(iter(entry["keys"])))
            taken = inline_paths(folder, CROSSING_ROLE)[1]
            witness = json.loads(taken.read_text())
            if (witness.get("disposition") != "claimed" or witness.get("token") != token
                    or witness.get("consumer_role") != "decode"):
                raise ValueError("Crossing ACK does not match a live D claim")
            self.crossing_acknowledged += 1
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
            self.acknowledge(token)

    def release(self,keys):
        keys=set(keys)
        self.drain_file_acks()
        if self.crossing_preissue:
            for token, entry in list(self.exports.items()):
                if "crossing_request" in entry and keys & entry["keys"]:
                    self.revoke_crossing_descriptor(token, entry)
        assert not any(keys & pending['keys'] for pending in self.exports.values()),'GPU clone completion ACK is still pending'
        for key in keys:
            path=directory(self.root,key)
            if key not in self.held:
                assert (path/'cuda_ipc_released.json').exists(),'Unknown E export cannot be released'
                continue
            self.held.pop(key);self.ready.pop(key);self.released_count+=1
            write(path/'cuda_ipc_released.json',dict(key=key,producer_pid=os.getpid(),time=time.time(),
                reason='Downstream request completed. Descriptors were acknowledged or safely revoked unused.'))
        return self.state()

    def state(self):
        self.drain_file_acks()
        if self.crossing_preissue and (self.root / '_crossing_cancelled').is_dir():
            for token, entry in list(self.exports.items()):
                request_key = entry.get("crossing_request")
                if request_key is not None and cancellation_marker(self.root, request_key).exists():
                    self.revoke_crossing_descriptor(token, entry)
        return dict(producer_pid=os.getpid(),live_images=len(self.held),live_storage_bytes=self.storage_bytes(),
            peak_storage_bytes=self.peak_storage_bytes,outstanding_exports=len(self.exports),
            crossing_preissue=self.crossing_preissue,crossing_issued=self.crossing_issued,
            crossing_acknowledged=self.crossing_acknowledged,crossing_revoked=self.crossing_revoked,
            saved_count=self.saved_count,released_count=self.released_count,max_storage_bytes=self.max_storage_bytes)


class CUDAIPCEncoderConnector(ECExampleConnector):
    def __init__(self,vllm_config,role):
        super().__init__(vllm_config,role)
        global REGISTRY,CONSUMER_CONNECTOR
        extra=vllm_config.ec_transfer_config.ec_connector_extra_config
        self.slice_aware = (
            os.environ.get('MLLM_EPD_ROLE') == 'decode'
            and os.environ.get('MLLM_EPD_SLICE_AWARE_EC') == '1'
        )
        self._slice_rows_by_key = {}
        if self.slice_aware:
            if vllm_config.cache_config.enable_prefix_caching or vllm_config.speculative_config:
                raise ValueError('Slice-aware EC requires uncached prompts and non-speculative decode')
            if vllm_config.kv_transfer_config.kv_load_failure_policy != 'fail':
                raise ValueError('Slice-aware EC requires an exact imported prefix or failed request')
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

    def update_state_after_alloc(self, request, index):
        super().update_state_after_alloc(request, index)
        if not self.slice_aware:
            return
        feature = request.mm_features[index]
        key = feature.identifier
        if key not in self._mm_datas_need_loads:
            return
        params = request.kv_transfer_params or {}
        plan = split_plan(params)
        position = feature.mm_position
        rows = (0, position.get_num_embeds())
        use_slice = params.get(SLICE_KEY, True)
        if type(use_slice) is not bool:
            raise ValueError('Slice-aware EC selection must be a boolean')
        if use_slice and plan is not None and params.get('remote_block_ids'):
            prefix, length = plan
            if params.get('remote_num_tokens') != prefix or request.num_prompt_tokens != length:
                raise ValueError('Slice-aware EC requires the exact remote token ownership')
            # NIXL clears do_remote_prefill at allocation. The computed count
            # must still prove that this image's omitted rows lie in the import.
            if request.num_computed_tokens >= prefix:
                rows = suffix_rows(position, prefix)
        previous = self._slice_rows_by_key.get(key)
        if previous is not None:
            if previous[1] != rows[1]:
                raise ValueError('Shared encoder identifier has a different row count')
            rows = (min(previous[0], rows[0]), rows[1])
        self._slice_rows_by_key[key] = rows

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        rows = {key: bounds for key, bounds in self._slice_rows_by_key.items() if bounds[0] != 0}
        self._slice_rows_by_key.clear()
        return SliceAwareECMetadata(metadata.mm_datas, rows) if rows else metadata

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
        rows_by_key = metadata.rows_by_key if isinstance(metadata, SliceAwareECMetadata) else {}
        keys=[item.mm_hash for item in metadata.mm_datas if item.mm_hash not in encoder_cache]
        if not keys:return
        records=[json.loads((directory(self._storage_path,key)/'cuda_ipc_ready.json').read_text()) for key in keys]
        assert len({r['producer_pid'] for r in records})==1
        pid=records[0]['producer_pid'];proc=Path('/proc',str(pid))
        assert proc.exists() and proc.stat().st_uid==os.getuid(),'The exact producer process must remain alive'
        begin=time.monotonic()
        role=os.environ['MLLM_EPD_ROLE'];groups=[];remaining=[];inline_images=0;redirected_inline_images=0;rpc_ms=0.
        preissued_crossing_images=0
        allow_redirect=role=='decode' and os.environ.get('MLLM_EPD_LATE_INLINE_REDIRECT')=='1'
        for key in keys:
            folder=directory(self._storage_path,key)
            candidates = decode_inline_roles(folder, allow_redirect) if role == 'decode' else (role,)
            # Suffix-only images still take the existing P-inline fast path.
            # A crossing image may use its own spare only after that fails.
            for issued_role in candidates:
                data = claim_inline(folder,key,issued_role,role)
                if data is not None:
                    break
            if data is None:
                remaining.append(key)
            else:
                groups.append(data)
                inline_images+=1
                redirected_inline_images+=int(issued_role=='prefill' and role=='decode')
                preissued_crossing_images+=int(issued_role==CROSSING_ROLE)
        if remaining:
            sent=time.monotonic()
            groups.append(self._rpc('export',dict(keys=remaining,consumer=f'{role}_{os.getpid()}')))
            rpc_ms=(time.monotonic()-sent)*1000
        exported=time.monotonic()
        assert all(group['producer_pid']==pid for group in groups)
        items={item['key']:item for group in groups for item in group['items']}
        assert set(items)==set(keys) and sum(len(group['items']) for group in groups)==len(keys)
        views=[]
        clone_shapes=[]
        for key,record in zip(keys,records):
            item=items[key]
            received=rebuild_cuda_tensor(*decode_reduction_args(item['args']))
            assert list(received.shape)==record['shape'] and str(received.dtype)==record['dtype']
            views.append(received)
            start, end = rows_by_key.get(key, (0, received.shape[0]))
            if not 0 <= start < end == received.shape[0]:
                raise ValueError(f'Invalid encoder clone row interval for {key}: {(start, end)}')
            clone = (received[start:end] if start else received).clone()
            encoder_cache[key] = SlicedEncoderRows(clone, start) if start else clone
            clone_shapes.append(dict(key=key, first_source_row=start, source_shape=list(received.shape),
                                     source_stride=list(received.stride()),
                                     source_storage_offset_elements=received.storage_offset(),
                                     clone_shape=list(clone.shape), clone_stride=list(clone.stride()),
                                     logical_bytes=clone.numel() * clone.element_size(),
                                     resident_storage_bytes=clone.untyped_storage().nbytes()))
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
                inline_images=inline_images,redirected_inline_images=redirected_inline_images,
                preissued_crossing_images=preissued_crossing_images,
                import_clone_sync_host_ms=(cloned-exported)*1000,
                completion_ACK_host_ms=(ended-cloned)*1000,ack_mode=ack_mode,
                logical_bytes=sum(item['logical_bytes'] for item in clone_shapes), clone_shapes=clone_shapes,
                clone_complete_before_ACK=True,scope='Original E tensor transport host enclosure, not exclusive GPU cost.'))
