"""Run the actual full-request ASGI path without a GPU or fabricated split metadata."""
import asyncio
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from . import whole_deflection_proxy as module
from .test_idle_proxy import Session,Stream,body
from .token_split import PLAN_KEY,POLICY_KEY


class WholeDeflectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_direct_full_request_has_no_p_forward_or_prefix_kv_import(self):
        stream=Stream();stream.allow_first.set();stream.allow_finish.set();session=Session(stream)
        original=body();before=copy.deepcopy(original)
        with TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)),\
                patch.object(module.aiohttp,'ClientSession',return_value=session):
            (Path(folder)/'raw').mkdir()
            app=module.application('full_correctness','http://E','http://P','http://D',
                policy='always_correctness',correctness_only=True)
            async with app.router.lifespan_context(app),httpx.AsyncClient(
                    transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                self.assertEqual((await client.get('/health')).json()['name'],'full_correctness')
                response=await client.post('/v1/chat/completions',json=original,headers={'x-request-id':'full'})
                self.assertEqual(response.status_code,200)
            self.assertEqual(original,before)
            self.assertFalse(any(url.startswith('http://P/') for url,_,_ in session.calls))
            destination=next(value for url,value,_ in session.calls if url.startswith('http://D/'))
            self.assertNotIn('kv_transfer_params',destination)
            self.assertEqual(destination['messages'],original['messages'])
            self.assertEqual(destination['max_tokens'],2048)
            trace=json.loads((Path(folder)/'raw/proxy_full_correctness.jsonl').read_text())
            self.assertTrue(trace['deflected']);self.assertEqual(trace['evidence'],'CORRECTNESS_ONLY')
            self.assertNotIn('P_submit',trace);self.assertNotIn('P_ready',trace)

    async def test_no_pressure_rule_and_never_control_preserve_native_full_prefill(self):
        for policy in ['never','idle_after_encode']:
            stream=Stream();stream.allow_first.set();stream.allow_finish.set();session=Session(stream)
            with TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)),\
                    patch.object(module.aiohttp,'ClientSession',return_value=session):
                (Path(folder)/'raw').mkdir()
                app=module.application(policy,'http://E','http://P','http://D',policy=policy)
                async with app.router.lifespan_context(app),httpx.AsyncClient(
                        transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                    reply=await client.post('/v1/chat/completions',json=body(),headers={'x-request-id':'native'})
                    self.assertEqual(reply.status_code,200)
                source=next(value for url,value,_ in session.calls if url.startswith('http://P/'))
                destination=next(value for url,value,_ in session.calls if url.startswith('http://D/'))
                self.assertTrue(source['kv_transfer_params']['do_remote_decode'])
                self.assertTrue(destination['kv_transfer_params']['do_remote_prefill'])
                for request in [source,destination]:
                    self.assertNotIn(PLAN_KEY,request['kv_transfer_params'])
                    self.assertNotIn(POLICY_KEY,request['kv_transfer_params'])
                trace=json.loads((Path(folder)/f'raw/proxy_{policy}.jsonl').read_text())
                self.assertFalse(trace['deflected']);self.assertTrue(trace['decision']['after_encode'])

    async def test_role_sse_is_not_the_full_prefill_completion_boundary(self):
        stream=Stream();session=Session(stream)
        with TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)),\
                patch.object(module.aiohttp,'ClientSession',return_value=session):
            (Path(folder)/'raw').mkdir()
            app=module.application('role_gate','http://E','http://P','http://D',
                policy='always_correctness',correctness_only=True)
            async with app.router.lifespan_context(app),httpx.AsyncClient(
                    transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                task=asyncio.create_task(client.post('/v1/chat/completions',json=body(),headers={'x-request-id':'full'}))
                await stream.role_sent.wait()
                state=(await client.get('/campaign_state')).json()
                self.assertEqual(state['continuation_admission']['prefill_owner'],'full')
                stream.allow_first.set();await stream.first_sent.wait();await asyncio.sleep(0)
                state=(await client.get('/campaign_state')).json()
                self.assertIsNone(state['continuation_admission']['prefill_owner'])
                self.assertEqual(state['continuation_admission']['active_http_requests'],1)
                stream.allow_finish.set();self.assertEqual((await task).status_code,200)

    def test_force_or_capture_mode_cannot_be_a_performance_policy(self):
        with self.assertRaises(AssertionError):
            module.application('invalid','http://E','http://P','http://D',policy='always_correctness')
        with self.assertRaises(AssertionError):
            module.application('invalid','http://E','http://P','http://D',capture=True)


if __name__=='__main__':unittest.main()
