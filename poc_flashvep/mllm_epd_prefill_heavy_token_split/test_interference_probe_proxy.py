"""Real coroutine gate prevents probes from racing prior decode or pending Native admission."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from . import interference_probe_proxy as module
from .interference_probe_proxy import ProbeAdmission
from .test_idle_proxy import Session,Stream,Post,body


class HeldPrefill(Session):
    def __init__(self,stream):
        super().__init__(stream);self.P_entered=asyncio.Event();self.P_continue=asyncio.Event()

    def post(self,url,json,headers=None):
        original=super().post(url,json,headers)
        if not url.startswith('http://P/'):return original
        async def callback():
            self.P_entered.set();await self.P_continue.wait()
            return await original
        return Post(callback)


class ProbeAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_reject_active_decode_and_resume_native_only_after_real_probe_completion(self):
        gate=ProbeAdmission();await gate.enter('old_native',False)
        self.assertFalse((await gate.try_probe('rejected'))['accepted'])
        await gate.leave('old_native')
        result=await gate.try_probe('probe')
        self.assertTrue(result['accepted']);self.assertEqual(result['admission']['before']['active_http_requests'],0)
        native=asyncio.create_task(gate.enter('new_native',False));await asyncio.sleep(0)
        self.assertFalse(native.done())
        self.assertFalse((await gate.try_probe('next_probe'))['accepted'])
        await gate.first_token('probe');await gate.leave('probe')
        await native;await gate.leave('new_native')
        self.assertEqual(gate.state()['active_http_requests'],0)

    async def test_failed_probe_cannot_reopen_new_native_admission(self):
        gate=ProbeAdmission();self.assertTrue((await gate.try_probe('failed'))['accepted'])
        await gate.leave('failed',failed=True)
        with self.assertRaises(RuntimeError):await gate.try_probe('retry')
        with self.assertRaises(RuntimeError):await gate.enter('new_native',False)

    async def test_native_waiter_has_priority_over_another_probe(self):
        gate=ProbeAdmission();await gate.try_probe('first')
        native=asyncio.create_task(gate.enter('already_waiting_native',False));await asyncio.sleep(0)
        async with gate.condition:
            gate.prefill_owner=None
            self.assertTrue(gate.pending)
        self.assertFalse((await gate.try_probe('must_not_overtake'))['accepted'])
        await gate.leave('first');await native;await gate.leave('already_waiting_native')


class ProbeASGITest(unittest.IsolatedAsyncioTestCase):
    async def test_real_native_handler_cannot_submit_decode_while_probe_owns_it(self):
        stream=Stream();stream.allow_first.set();stream.allow_finish.set();session=HeldPrefill(stream)
        with TemporaryDirectory() as directory,patch.object(module,'ROOT',Path(directory)),\
                patch.object(module.proxy,'ROOT',Path(directory)),\
                patch.object(module.proxy.aiohttp,'ClientSession',return_value=session):
            (Path(directory)/'raw').mkdir()
            original_factory=module.idle_admission.IdleAdmission
            app=module.application('actual_probe_fixture','http://E','http://P','http://D')
            self.assertIs(module.idle_admission.IdleAdmission,original_factory)
            async with app.router.lifespan_context(app),httpx.AsyncClient(
                    transport=httpx.ASGITransport(app),base_url='http://fixture') as client:
                idle=await client.post('/campaign/probe/acquire',json=dict(arm='D'))
                self.assertFalse(idle.json()['accepted'])
                native=asyncio.create_task(client.post('/v1/chat/completions',json=body(),headers={'x-request-id':'native'}))
                await session.P_entered.wait()
                lease=(await client.post('/campaign/probe/acquire',json=dict(arm='D'))).json()
                self.assertTrue(lease['accepted'])
                self.assertEqual(lease['admission']['admission']['before']['active_http_requests'],0)
                session.P_continue.set();await asyncio.sleep(0);await asyncio.sleep(0)
                self.assertFalse(session.headers_ready.is_set())
                with self.assertRaises(AssertionError):
                    await client.post('/campaign/probe/release',json=dict(token=lease['token'],failed=False))
                state=(await client.get('/campaign/probe/state')).json()
                self.assertIn(lease['token'],state['leases'])
                reply=await client.post('/campaign/probe/release',json=dict(token=lease['token'],
                    failed=False,output_token_ids=[42],completion_tokens=1))
                self.assertTrue(reply.json()['released'])
                self.assertEqual((await native).status_code,200)
                state=(await client.get('/campaign_state')).json()
                self.assertEqual(state['completed'],1)
                self.assertEqual(state['continuation_admission']['active_http_requests'],0)


if __name__=='__main__':unittest.main()
