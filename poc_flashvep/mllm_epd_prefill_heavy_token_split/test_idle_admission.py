"""Check exact first-token parsing and admission lifetimes with real coroutines."""
import asyncio
import unittest

from .idle_admission import FirstToken, IdleAdmission


class IdleAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_d_termination_fails_closed_instead_of_admitting_more_work(self):
        gate=IdleAdmission()
        await gate.enter('disconnected',True)
        await gate.leave('disconnected',failed=True)
        self.assertTrue(gate.state()['failed_closed'])
        with self.assertRaises(RuntimeError):await gate.enter('new_suffix',True)
        self.assertEqual(gate.state()['active_http_requests'],0)

    async def test_native_enters_after_first_token_but_extra_prefill_waits_until_done(self):
        gate=IdleAdmission()
        await gate.enter('existing_decode',False)
        suffix=asyncio.create_task(gate.enter('suffix',True))
        await asyncio.sleep(0)
        self.assertFalse(suffix.done())
        await gate.leave('existing_decode')
        admitted=await suffix
        self.assertEqual(admitted['before']['active_http_requests'],0)
        native=asyncio.create_task(gate.enter('new_native',False))
        second=asyncio.create_task(gate.enter('second_suffix',True))
        await asyncio.sleep(0)
        self.assertFalse(native.done());self.assertFalse(second.done())
        await gate.first_token('suffix')
        await native
        self.assertFalse(second.done())
        self.assertEqual(set(gate.active),{'suffix','new_native'})
        await gate.leave('suffix')
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        await gate.leave('new_native')
        await second
        await gate.leave('second_suffix')
        self.assertEqual(gate.state()['active_http_requests'],0)
        self.assertIsNone(gate.prefill_owner)

    async def test_cancelled_waiter_never_holds_a_lease(self):
        gate=IdleAdmission()
        await gate.enter('native',False)
        task=asyncio.create_task(gate.enter('cancelled',True))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(gate.waiting,0)
        self.assertEqual(set(gate.active),{'native'})
        await gate.leave('native')

    async def test_failure_before_first_token_releases_prefill_ownership(self):
        gate=IdleAdmission()
        await gate.enter('failed_suffix',True)
        native=asyncio.create_task(gate.enter('native',False))
        await asyncio.sleep(0)
        self.assertFalse(native.done())
        await gate.leave('failed_suffix')
        await native
        await gate.leave('native')

    async def test_duplicate_waiting_ids_cannot_overwrite_an_active_lease(self):
        gate=IdleAdmission()
        await gate.enter('native',False)
        waiting=asyncio.create_task(gate.enter('same_id',True))
        await asyncio.sleep(0)
        with self.assertRaises(AssertionError):await gate.enter('same_id',True)
        await gate.leave('native')
        await waiting
        await gate.leave('same_id')

    def test_sse_role_headers_fragmentation_and_usage_do_not_release_early(self):
        parser=FirstToken()
        self.assertFalse(parser.feed(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'))
        self.assertFalse(parser.feed(b'data: {"usage":{"prompt_tokens":100},"choices":[]}\n\n'))
        self.assertFalse(parser.feed(b'data: {"choices":[{"token_i'))
        self.assertFalse(parser.feed(b'ds":[42],"delta":{"content":"x"}}]}'))
        self.assertTrue(parser.feed(b'\n\n'))
        self.assertTrue(parser.seen)
        self.assertFalse(parser.feed(b'data: {"choices":[{"token_ids":[43]}]}\n\n'))


if __name__=='__main__':unittest.main()
