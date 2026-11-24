"""Check JSON geometry and exact E ownership before any serving trial."""
import json
from contextlib import nullcontext
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def reduction(shape=(3,3584)):
    import torch
    return (torch.Tensor,torch.Size(shape),(3584,1),0,torch.storage.TypedStorage,torch.bfloat16,0,
            b'm'*64,shape[0]*shape[1]*2,0,False,b'counter',0,b'e'*64,True)


class FakeTensor:
    def __init__(self,pointer=100):
        import torch
        self.dtype=torch.bfloat16;self.shape=(3,3584);self.ndim=2
        self.device=SimpleNamespace(type='cuda');self.pointer=pointer
    def detach(self):return self
    def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.pointer,nbytes=lambda:21504)
    def clone(self):return FakeTensor(self.pointer+1)
    def numel(self):return 3*3584
    def element_size(self):return 2


class CodecTest(unittest.TestCase):
    def test_versioned_opaque_memory_handle_is_not_truncated(self):
        from .cuda_ipc_codec import encode_reduction_args,decode_reduction_args
        original=list(reduction());original[7]=b'v1'+b'm'*64
        original=tuple(original)
        self.assertEqual(decode_reduction_args(encode_reduction_args(original)),original)
    def test_json_round_trip_preserves_only_known_arguments(self):
        from .cuda_ipc_codec import encode_reduction_args,decode_reduction_args
        original=reduction()
        self.assertEqual(decode_reduction_args(json.loads(json.dumps(encode_reduction_args(original)))),original)
    def test_out_of_bounds_or_wrong_dtype_is_rejected(self):
        import torch
        from .cuda_ipc_codec import encode_reduction_args,decode_reduction_args
        value=encode_reduction_args(reduction());value['tensor_offset']=1
        with self.assertRaises(AssertionError):decode_reduction_args(value)
        wrong=list(reduction());wrong[5]=torch.float32
        with self.assertRaises(AssertionError):encode_reduction_args(wrong)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.event=patch('torch.cuda.Event',lambda:SimpleNamespace(record=lambda:None))
        self.stream=patch('torch.cuda.current_stream',lambda:SimpleNamespace(wait_event=lambda event:None))
        self.new_stream=patch('torch.cuda.Stream',lambda:object())
        self.context=patch('torch.cuda.stream',lambda stream:nullcontext())
        for item in [self.event,self.stream,self.new_stream,self.context]:
            item.start();self.addCleanup(item.stop)
    def test_cannot_release_before_every_issued_clone_ack(self):
        from .cuda_ipc_ec import ProducerRegistry
        with TemporaryDirectory() as folder:
            registry=ProducerRegistry(folder);registry.save('image-A',FakeTensor())
            fake=lambda tensor:(SimpleNamespace(__name__='rebuild_cuda_tensor'),reduction())
            with patch('torch.multiprocessing.reductions.reduce_tensor',side_effect=fake) as exported:
                a=registry.export(['image-A'],'prefill_1');b=registry.export(['image-A'],'decode_2')
                self.assertEqual(exported.call_count,2)
            with self.assertRaises(AssertionError):registry.release(['image-A'])
            registry.acknowledge(a['token'])
            with self.assertRaises(AssertionError):registry.release(['image-A'])
            registry.acknowledge(b['token']);result=registry.release(['image-A'])
            self.assertEqual(result['live_images'],0)
            self.assertEqual(registry.release(['image-A'])['live_images'],0)
            with self.assertRaises(AssertionError):registry.export(['image-A'],'decode_2')
            self.assertTrue((Path(folder)/'image-A/cuda_ipc_released.json').exists())
    def test_shared_storage_is_not_double_counted(self):
        from .cuda_ipc_ec import ProducerRegistry
        with TemporaryDirectory() as folder:
            registry=ProducerRegistry(folder)
            registry.save('image-A',FakeTensor(100));registry.save('image-B',FakeTensor(100))
            self.assertEqual(registry.storage_bytes(),21504)
            with self.assertRaises(AssertionError):registry.save('image-A',FakeTensor(100))
    def test_file_ack_keeps_the_same_release_gate(self):
        from .cuda_ipc_ec import ProducerRegistry
        from .common import write
        with TemporaryDirectory() as folder:
            registry=ProducerRegistry(folder);registry.save('image-A',FakeTensor())
            fake=lambda tensor:(SimpleNamespace(__name__='rebuild_cuda_tensor'),reduction())
            with patch('torch.multiprocessing.reductions.reduce_tensor',side_effect=fake):
                exported=registry.export(['image-A'],'prefill_1')
            with self.assertRaises(AssertionError):registry.release(['image-A'])
            write(Path(folder)/'_ipc_acks'/(exported['token']+'.json'),dict(token=exported['token'],
                producer_pid=os.getpid(),consumer='prefill_1',clone_complete=True))
            self.assertEqual(registry.release(['image-A'])['live_images'],0)
    def test_native_inline_export_still_requires_a_clone_ack(self):
        from .cuda_ipc_ec import ProducerRegistry
        with TemporaryDirectory() as folder:
            registry=ProducerRegistry(folder);registry.inline_role='prefill'
            fake=lambda tensor:(SimpleNamespace(__name__='rebuild_cuda_tensor'),reduction())
            with patch('torch.multiprocessing.reductions.reduce_tensor',side_effect=fake):
                registry.save('image-A',FakeTensor())
            descriptor=json.loads((Path(folder)/'image-A/cuda_ipc_inline_prefill.json').read_text())
            self.assertEqual(descriptor['consumer'],'prefill')
            self.assertEqual([r['key'] for r in descriptor['items']],['image-A'])
            with self.assertRaises(AssertionError):registry.release(['image-A'])
            registry.acknowledge(descriptor['token'])
            self.assertEqual(registry.release(['image-A'])['live_images'],0)
    def test_memory_bound_never_frees_an_inflight_tensor(self):
        from .cuda_ipc_ec import ProducerRegistry
        with TemporaryDirectory() as folder:
            registry=ProducerRegistry(folder,max_storage_bytes=21504)
            registry.save('image-A',FakeTensor(100))
            with self.assertRaises(MemoryError):registry.save('image-B',FakeTensor(200))
            self.assertEqual(set(registry.held),{'image-A','image-B'})


class ConsumerTest(unittest.TestCase):
    def test_inline_export_is_consumed_once_and_never_waits_for_export_rpc(self):
        from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnectorMetadata,MMMeta
        from .cuda_ipc_codec import encode_reduction_args
        from .cuda_ipc_ec import CUDAIPCEncoderConnector
        with TemporaryDirectory() as folder:
            path=Path(folder)/'image-A';path.mkdir();token='a'*32
            (path/'cuda_ipc_ready.json').write_text(json.dumps(dict(producer_pid=os.getpid(),shape=[3,3584],dtype='torch.bfloat16')))
            (path/'cuda_ipc_inline_prefill.json').write_text(json.dumps(dict(producer_pid=os.getpid(),token=token,
                consumer='prefill',items=[dict(key='image-A',args=encode_reduction_args(reduction()))])))
            connector=object.__new__(CUDAIPCEncoderConnector);connector._storage_path=folder;connector.ack_mode='file'
            connector._connector_metadata=ECExampleConnectorMetadata();connector._connector_metadata.mm_datas=[MMMeta('image-A',3)]
            connector._rpc=lambda *args:(_ for _ in ()).throw(AssertionError('Inline does not require a live export RPC'))
            class Event:
                def record(self):pass
                def synchronize(self):pass
            with patch('torch.multiprocessing.reductions.rebuild_cuda_tensor',return_value=FakeTensor()),\
                 patch('torch.cuda.Event',Event),patch.dict(os.environ,MLLM_EPD_ROLE='prefill',CUDA_VISIBLE_DEVICES='5'),\
                 patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.append'):
                connector.start_load_caches({})
                self.assertTrue((path/'cuda_ipc_inline_prefill_taken.json').exists())
                self.assertTrue((Path(folder)/'_ipc_acks'/(token+'.json')).exists())
                # A later cache reload must request a fresh descriptor, not reuse a refcounter.
                with self.assertRaisesRegex(AssertionError,'export RPC'):connector.start_load_caches({})
    def test_file_ack_follows_sync_without_waiting_for_a_producer_rpc(self):
        from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnectorMetadata,MMMeta
        from .cuda_ipc_codec import encode_reduction_args
        from .cuda_ipc_ec import CUDAIPCEncoderConnector
        from .common import write as original_write
        with TemporaryDirectory() as folder:
            path=Path(folder)/'image-A';path.mkdir()
            (path/'cuda_ipc_ready.json').write_text(json.dumps(dict(producer_pid=os.getpid(),shape=[3,3584],dtype='torch.bfloat16')))
            connector=object.__new__(CUDAIPCEncoderConnector);connector._storage_path=folder;connector.ack_mode='file'
            connector._connector_metadata=ECExampleConnectorMetadata();connector._connector_metadata.mm_datas=[MMMeta('image-A',3)]
            calls=[];token='a'*32
            class Event:
                def record(self):calls.append('record')
                def synchronize(self):calls.append('synchronize')
            def rpc(operation,body):
                self.assertEqual(operation,'export','A file ACK must not block on an E ACK RPC')
                calls.append(operation)
                return dict(producer_pid=os.getpid(),token=token,consumer='prefill_1',
                    items=[dict(key='image-A',args=encode_reduction_args(reduction()))])
            def written(path,value):
                self.assertEqual(calls[-1],'synchronize')
                self.assertTrue(value['clone_complete']);calls.append('file_ack');original_write(path,value)
            connector._rpc=rpc
            with patch('torch.multiprocessing.reductions.rebuild_cuda_tensor',return_value=FakeTensor()),\
                 patch('torch.cuda.Event',Event),patch.dict(os.environ,MLLM_EPD_ROLE='prefill',CUDA_VISIBLE_DEVICES='5'),\
                 patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.write',side_effect=written),\
                 patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.append'):
                cache={};connector.start_load_caches(cache)
            self.assertEqual(calls,['export','record','synchronize','file_ack'])
            self.assertTrue((Path(folder)/'_ipc_acks'/(token+'.json')).exists())
    def test_clone_sync_precedes_ack(self):
        from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnectorMetadata,MMMeta
        from .cuda_ipc_codec import encode_reduction_args
        from .cuda_ipc_ec import CUDAIPCEncoderConnector
        with TemporaryDirectory() as folder:
            path=Path(folder)/'image-A';path.mkdir()
            (path/'cuda_ipc_ready.json').write_text(json.dumps(dict(producer_pid=os.getpid(),shape=[3,3584],dtype='torch.bfloat16')))
            connector=object.__new__(CUDAIPCEncoderConnector);connector._storage_path=folder
            connector._connector_metadata=ECExampleConnectorMetadata()
            connector._connector_metadata.mm_datas=[MMMeta('image-A',3)]
            calls=[]
            class Event:
                def record(self):calls.append('record')
                def synchronize(self):calls.append('synchronize')
            def rpc(operation,body):
                calls.append(operation)
                if operation=='export':return dict(producer_pid=os.getpid(),token='token',
                    items=[dict(key='image-A',args=encode_reduction_args(reduction()))])
                self.assertEqual(calls[-2],'synchronize')
                return dict(acknowledged=True,producer_pid=os.getpid())
            connector._rpc=rpc
            with patch('torch.multiprocessing.reductions.rebuild_cuda_tensor',return_value=FakeTensor()),\
                 patch('torch.cuda.Event',Event),patch.dict(os.environ,MLLM_EPD_ROLE='prefill',CUDA_VISIBLE_DEVICES='5'),\
                 patch('poc_flashvep.mllm_epd_prefill_heavy_token_split.cuda_ipc_ec.append'):
                cache={};connector.start_load_caches(cache)
            self.assertEqual(calls,['export','record','synchronize','ack'])
            self.assertIn('image-A',cache)


if __name__=='__main__':unittest.main()
