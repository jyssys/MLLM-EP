"""Exercise the actual ASGI proxy with a CPU-only E/P/D protocol fixture."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from . import proxy
from .token_split import PLAN_KEY, POLICY_KEY


class Reply:
    status=200
    def __init__(self,value=None,content=None):
        self.value=value;self.content=content;self.released=False
    async def json(self):return self.value
    async def text(self):return json.dumps(self.value)
    def release(self):self.released=True


class Post:
    def __init__(self,callback):self.callback=callback
    def __await__(self):return self.callback().__await__()
    async def __aenter__(self):
        self.reply=await self.callback();return self.reply
    async def __aexit__(self,*args):self.reply.release()


class Stream:
    def __init__(self):
        self.role_sent=asyncio.Event();self.allow_first=asyncio.Event()
        self.first_sent=asyncio.Event();self.allow_finish=asyncio.Event()
    async def iter_any(self):
        self.role_sent.set()
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        await self.allow_first.wait()
        self.first_sent.set()
        yield b'data: {"choices":[{"token_ids":[42],"delta":{"content":"answer"}}]}\n\n'
        await self.allow_finish.wait()
        yield b'data: [DONE]\n\n'


class Session:
    def __init__(self,stream):self.stream=stream;self.calls=[];self.headers_ready=asyncio.Event();self.headers_block=None
    def post(self,url,json,headers=None):
        self.calls.append((url,json,headers))
        async def result():
            if url.startswith('http://E/campaign/ec_ipc/release'):
                return Reply([dict(released=True)])
            if url.startswith('http://E/'):
                return Reply(dict(usage=dict(prompt_tokens=100)))
            if url.startswith('http://P/'):
                plan=json['kv_transfer_params'].get(PLAN_KEY)
                length=plan['prefix_tokens'] if plan else 100
                return Reply(dict(kv_transfer_params=dict(remote_block_ids=[[1]],remote_num_tokens=length,
                    do_remote_prefill=True,do_remote_decode=False,remote_engine_id='real-protocol-fixture')))
            assert url.startswith('http://D/')
            self.headers_ready.set()
            if self.headers_block is not None:await self.headers_block.wait()
            return Reply(content=self.stream)
        return Post(result)
    async def close(self):pass


def body():
    return dict(model='original-model',messages=[dict(role='user',content=[
        dict(type='image_url',image_url=dict(url='file:///original_image')),dict(type='text',text='original question')])],
        max_tokens=2048,stream=True,return_token_ids=True,temperature=0)


class IdleProxyTest(unittest.IsolatedAsyncioTestCase):
    async def test_batch_one_preserves_upstream_concurrency_and_fifo_d_completion(self):
        streams = {name: Stream() for name in ('first', 'suffix', 'third')}
        for stream in streams.values():
            stream.allow_first.set()

        class BatchOneSession(Session):
            def post(self, url, json, headers=None):
                if not url.startswith('http://D/'):
                    return super().post(url, json, headers)
                self.calls.append((url, json, headers))
                name = headers['x-request-id'].removesuffix(':D')
                async def result():
                    return Reply(content=streams[name])
                return Post(result)

        for token in (False, True):
            for stream in streams.values():
                stream.role_sent.clear()
                stream.first_sent.clear()
                stream.allow_finish.clear()
            session = BatchOneSession(None)
            with TemporaryDirectory() as directory,patch.object(proxy,'ROOT',Path(directory)),\
                 patch.object(proxy.aiohttp,'ClientSession',return_value=session):
                (Path(directory)/'raw').mkdir()
                app = proxy.application('epd','http://E','http://P',['http://D'],'cpu_batch_one',
                    processor_handoff='grid_only',release_encoder_ipc=True,
                    decode_batch_one=True,token_suffix_fraction=.1 if token else 0.,
                    token_admission='decode_idle' if token else 'none')
                async with app.router.lifespan_context(app),httpx.AsyncClient(
                    transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                    first = asyncio.create_task(client.post('/v1/chat/completions',json=body(),
                        headers={'x-request-id':'first'}))
                    await streams['first'].first_sent.wait()
                    queued = []
                    for name in ('suffix','third'):
                        queued.append(asyncio.create_task(client.post('/v1/chat/completions',json=body(),
                            headers={'x-request-id':name})))
                        await asyncio.sleep(0)
                    state = (await client.get('/campaign_state')).json()['continuation_admission']
                    self.assertEqual((state['active_http_requests'],state['admission_waiting']),(1,2))
                    self.assertEqual(sum(url.startswith('http://P/') for url,_,_ in session.calls),3)
                    self.assertFalse(streams['suffix'].role_sent.is_set())
                    streams['first'].allow_finish.set()
                    self.assertEqual((await first).status_code,200)
                    await streams['suffix'].first_sent.wait()
                    self.assertFalse(streams['third'].role_sent.is_set())
                    streams['suffix'].allow_finish.set()
                    self.assertEqual((await queued[0]).status_code,200)
                    await streams['third'].first_sent.wait()
                    streams['third'].allow_finish.set()
                    self.assertEqual((await queued[1]).status_code,200)
                records = [json.loads(line) for line in
                    (Path(directory)/'raw/proxy_cpu_batch_one.jsonl').read_text().splitlines()]
                self.assertEqual([row['runtime_request_id'] for row in records],['first','suffix','third'])
                self.assertEqual(records[1]['D_at_P_ready']['active_http_requests'],1)
                self.assertTrue(all(row['D_admission']['after']['active_http_requests']==1 for row in records))

    async def test_idle_policy_falls_back_to_unmodified_native_before_encode(self):
        stream=Stream();stream.allow_first.set();stream.allow_finish.set();session=Session(stream)
        with TemporaryDirectory() as directory,patch.object(proxy,'ROOT',Path(directory)),\
             patch.object(proxy.aiohttp,'ClientSession',return_value=session):
            (Path(directory)/'raw').mkdir()
            app=proxy.application('epd','http://E','http://P',['http://D'],'cpu_fallback',
                processor_handoff='grid_only',release_encoder_ipc=True,token_suffix_fraction=.2,
                token_admission='decode_idle',token_policy='idle_fallback')
            async with app.router.lifespan_context(app),httpx.AsyncClient(
                transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                reply=await client.post('/v1/chat/completions',json=body(),headers={'x-request-id':'fallback'})
                self.assertEqual(reply.status_code,200)
            record=json.loads((Path(directory)/'raw/proxy_cpu_fallback.jsonl').read_text())
            self.assertFalse(record['token_enabled'])
            self.assertFalse(record['D_admission']['extra_prefill'])
            for url,request,_ in session.calls:
                if '/v1/' in url:
                    self.assertNotIn(POLICY_KEY,request.get('kv_transfer_params',{}))
                    self.assertNotIn(PLAN_KEY,request.get('kv_transfer_params',{}))

    async def test_role_only_sse_does_not_release_prefix_owner_and_completion_closes_it(self):
        stream=Stream();session=Session(stream)
        with TemporaryDirectory() as directory,patch.object(proxy,'ROOT',Path(directory)),\
             patch.object(proxy.aiohttp,'ClientSession',return_value=session):
            (Path(directory)/'raw').mkdir()
            app=proxy.application('epd','http://E','http://P',['http://D'],'cpu_idle',
                processor_handoff='grid_only',release_encoder_ipc=True,
                token_suffix_fraction=.2,token_admission='decode_idle')
            async with app.router.lifespan_context(app),httpx.AsyncClient(
                transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                task=asyncio.create_task(client.post('/v1/chat/completions',json=body(),headers={'x-request-id':'actual_id'}))
                await stream.role_sent.wait()
                state=(await client.get('/campaign_state')).json()
                self.assertEqual(state['continuation_admission']['prefill_owner'],'actual_id')
                stream.allow_first.set();await stream.first_sent.wait();await asyncio.sleep(0)
                state=(await client.get('/campaign_state')).json()
                self.assertIsNone(state['continuation_admission']['prefill_owner'])
                self.assertEqual(state['continuation_admission']['active_http_requests'],1)
                stream.allow_finish.set()
                response=await task
                self.assertEqual(response.status_code,200)
                state=(await client.get('/campaign_state')).json()
                self.assertEqual(state['continuation_admission']['active_http_requests'],0)
                self.assertEqual(state['decode_pending'],0)
            record=json.loads((Path(directory)/'raw/proxy_cpu_idle.jsonl').read_text())
            self.assertTrue(record['token_enabled'])
            self.assertGreaterEqual(record['D_first_generated_token'],record['D_first_SSE_bytes'])
            calls={url:request for url,request,_ in session.calls}
            self.assertIn(POLICY_KEY,calls['http://E/v1/chat/completions']['kv_transfer_params'])
            self.assertEqual(calls['http://D/v1/chat/completions']['kv_transfer_params'][PLAN_KEY],
                dict(prefix_tokens=80,full_prompt_tokens=100))
            self.assertTrue(any('/release' in url for url in calls))

    async def test_cancellation_before_d_headers_does_not_leak_a_gate_lease(self):
        stream=Stream();session=Session(stream);session.headers_block=asyncio.Event()
        with TemporaryDirectory() as directory,patch.object(proxy,'ROOT',Path(directory)),\
             patch.object(proxy.aiohttp,'ClientSession',return_value=session):
            (Path(directory)/'raw').mkdir()
            app=proxy.application('epd','http://E','http://P',['http://D'],'cpu_cancel',
                processor_handoff='grid_only',token_suffix_fraction=.2,token_admission='decode_idle')
            async with app.router.lifespan_context(app),httpx.AsyncClient(
                transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                task=asyncio.create_task(client.post('/v1/chat/completions',json=body(),headers={'x-request-id':'cancel_id'}))
                await session.headers_ready.wait();task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
                state=(await client.get('/campaign_state')).json()
                self.assertIsNone(state['continuation_admission']['prefill_owner'])
                self.assertEqual(state['continuation_admission']['active_http_requests'],0)
                self.assertEqual(state['decode_pending'],0)
                self.assertEqual(state['failed'],1)
                self.assertTrue(state['continuation_admission']['failed_closed'])
                retry=await client.post('/v1/chat/completions',json=body(),headers={'x-request-id':'new_after_failure'})
                self.assertEqual(retry.status_code,502)
                self.assertEqual(sum(url.startswith('http://D/') for url,_,_ in session.calls),1)


if __name__=='__main__':unittest.main()
