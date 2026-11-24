"""Request identity and exact multimodal metadata handoff contracts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


def request(messages=None,**changes):
    fields=dict(model='Qwen/Qwen2.5-VL-7B-Instruct',messages=messages or [{'role':'user','content':'original'}],
        temperature=0,max_tokens=16,stream=True,stream_options={'include_usage':True},kv_transfer_params=None)
    fields.update(changes)
    return SimpleNamespace(model_dump=lambda **kwargs:fields)


def rendered():
    import torch
    from vllm.multimodal.inputs import MultiModalFieldConfig,MultiModalFieldElem,MultiModalKwargsItem,MultiModalKwargsItems,PlaceholderRange
    item=MultiModalKwargsItem(dict(
        pixel_values=MultiModalFieldElem(data=torch.arange(24,dtype=torch.float32).reshape(4,6),
            field=MultiModalFieldConfig.flat('image',[slice(0,4)]).field),
        image_grid_thw=MultiModalFieldElem(data=torch.tensor([1,2,2]),
            field=MultiModalFieldConfig.batched('image',keep_on_cpu=True).field)))
    return ([{'role':'user','content':'original'}],[dict(type='multimodal',prompt_token_ids=[7,8,9,10,11,12],
        arrival_time=1.,prompt='original prompt',mm_kwargs=MultiModalKwargsItems(image=[item]),
        mm_hashes={'image':['original-image']},mm_placeholders={'image':[PlaceholderRange(2,1)]})])


class MetadataHandoffTest(unittest.TestCase):
    def test_real_api_generated_request_ids_do_not_change_prompt_identity(self):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from .processor_handoff import input_fingerprint
        base=dict(model='Qwen/Qwen2.5-VL-7B-Instruct',messages=[{'role':'user','content':'original'}])
        encoder=ChatCompletionRequest(**base,max_tokens=1,stream=False)
        decoder=ChatCompletionRequest(**base,max_tokens=16,stream=True,stream_options={'include_usage':True})
        self.assertNotEqual(encoder.request_id,decoder.request_id)
        self.assertEqual(input_fingerprint(encoder),input_fingerprint(decoder))

    def test_full_roundtrip_keeps_exact_tokens_fields_and_tensor_values(self):
        import torch
        from .processor_handoff import RenderStore,input_fingerprint
        original=rendered()
        with TemporaryDirectory() as folder:
            store=RenderStore(Path(folder));key='a'*64
            store.save(key,request(),original,'full')
            result=store.load(key,request(max_tokens=1,stream=False,stream_options=None,
                kv_transfer_params={'do_remote_decode':True}),arrival_time=123.)
            self.assertEqual(input_fingerprint(request()),input_fingerprint(request(max_tokens=1,stream=False,stream_options=None)))
            self.assertEqual(result[0],original[0])
            self.assertEqual(result[1][0]['prompt_token_ids'],original[1][0]['prompt_token_ids'])
            self.assertEqual(result[1][0]['mm_placeholders'],original[1][0]['mm_placeholders'])
            self.assertEqual(result[1][0]['arrival_time'],123.)
            for field in ['pixel_values','image_grid_thw']:
                self.assertTrue(torch.equal(result[1][0]['mm_kwargs']['image'][0][field].data,
                    original[1][0]['mm_kwargs']['image'][0][field].data))

    def test_grid_only_does_not_modify_the_real_encoder_input(self):
        import torch
        from .processor_handoff import RenderStore
        original=rendered();pixels=original[1][0]['mm_kwargs']['image'][0]['pixel_values'].data.clone()
        with TemporaryDirectory() as folder:
            store=RenderStore(Path(folder));key='b'*64
            store.save(key,request(),original,'grid_only')
            result=store.load(key,request(),arrival_time=5.)
            self.assertNotIn('pixel_values',result[1][0]['mm_kwargs']['image'][0])
            self.assertTrue(torch.equal(original[1][0]['mm_kwargs']['image'][0]['pixel_values'].data,pixels))
            self.assertTrue(torch.equal(result[1][0]['mm_kwargs']['image'][0]['image_grid_thw'].data,
                original[1][0]['mm_kwargs']['image'][0]['image_grid_thw'].data))
            self.assertEqual(result[1][0]['prompt_token_ids'],original[1][0]['prompt_token_ids'])

    def test_placeholder_embedding_mask_is_preserved(self):
        import torch
        from vllm.multimodal.inputs import PlaceholderRange
        from .processor_handoff import RenderStore
        original=rendered();mask=torch.tensor([True,False,True,False])
        original[1][0]['mm_placeholders']['image']=[PlaceholderRange(1,4,is_embed=mask)]
        with TemporaryDirectory() as folder:
            store=RenderStore(Path(folder));key='f'*64
            store.save(key,request(),original,'grid_only')
            result=store.load(key,request(),arrival_time=7.)
            placeholder=result[1][0]['mm_placeholders']['image'][0]
            self.assertEqual((placeholder.offset,placeholder.length),(1,4))
            self.assertTrue(torch.equal(placeholder.is_embed,mask))

    def test_identity_mismatch_and_corrupt_buffer_do_not_fall_back(self):
        from .processor_handoff import RenderStore
        with TemporaryDirectory() as folder:
            store=RenderStore(Path(folder));key='c'*64
            store.save(key,request(),rendered(),'grid_only')
            with self.assertRaises(AssertionError):store.load(key,request(messages=[{'role':'user','content':'different'}]),arrival_time=2.)
            data=Path(folder)/key/'buffer_0.bin'
            data.write_bytes(data.read_bytes()+b'corruption')
            with self.assertRaises(AssertionError):store.load(key,request(),arrival_time=2.)

    def test_missing_and_unsafe_keys_fail_without_processor_fallback(self):
        from .processor_handoff import RenderStore
        with TemporaryDirectory() as folder:
            store=RenderStore(Path(folder))
            with self.assertRaises(FileNotFoundError):store.load('d'*64,request(),arrival_time=2.)
            with self.assertRaises(AssertionError):store.load('../escape',request(),arrival_time=2.)


class MetadataPathTest(unittest.IsolatedAsyncioTestCase):
    async def test_consumer_never_calls_the_processor_or_falls_back(self):
        import os
        from unittest.mock import AsyncMock,patch
        from vllm.renderers.online_renderer import OnlineRenderer
        from . import processor_handoff as module
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'evidence/raw').mkdir(parents=True)
            env=dict(MLLM_EPD_METADATA_PATH=str(root/'metadata'),MLLM_EPD_EC_PATH=str(root/'EC'),
                     MLLM_EPD_ROLE='encoder',CUDA_VISIBLE_DEVICES='4')
            original=rendered();key='e'*64
            produce=AsyncMock(return_value=original)
            with patch.dict(os.environ,env),patch.object(module,'ROOT',root/'evidence'),\
                    patch.object(OnlineRenderer,'render_chat',produce):
                module.install();token=module.HANDOFF_CONTEXT.set((key,'grid_only'))
                try:self.assertIs(await OnlineRenderer.render_chat(None,request()),original)
                finally:module.HANDOFF_CONTEXT.reset(token)
            self.assertEqual(produce.await_count,1)
            cache=root/'EC/original-image';cache.mkdir(parents=True)
            marker=cache/'encoder_cache.safetensors';marker.touch()
            env['MLLM_EPD_ROLE']='prefill';env['CUDA_VISIBLE_DEVICES']='5'
            consumer=AsyncMock(side_effect=AssertionError('The P consumer must not run the original processor'))
            with patch.dict(os.environ,env),patch.object(module,'ROOT',root/'evidence'),\
                    patch.object(OnlineRenderer,'render_chat',consumer):
                module.install();token=module.HANDOFF_CONTEXT.set((key,'grid_only'))
                try:
                    result=await OnlineRenderer.render_chat(None,request())
                    self.assertEqual(result[1][0]['prompt_token_ids'],original[1][0]['prompt_token_ids'])
                    self.assertEqual(consumer.await_count,0)
                    marker.unlink()
                    with self.assertRaises(AssertionError):await OnlineRenderer.render_chat(None,request())
                    self.assertEqual(consumer.await_count,0)
                finally:module.HANDOFF_CONTEXT.reset(token)


if __name__=='__main__':unittest.main()
