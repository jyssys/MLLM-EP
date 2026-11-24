"""Check completed-row boundaries and collector teardown without GPUs."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .profile_completion_roi import collect,completed_requests


class CompletionROITest(unittest.IsolatedAsyncioTestCase):
    async def test_observation_has_two_boundaries_and_reverse_teardown(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'requests.jsonl';finished=asyncio.Event();calls=[]
            def rpc(url,operation,body):calls.append((url,operation));return [{'passed':True}]
            async def produce():
                for index in range(8):
                    with path.open('a') as stream:stream.write('{"request":'+str(index)+'}\n')
                    await asyncio.sleep(.02)
                finished.set()
            writer=asyncio.create_task(produce())
            records,audit=await collect(path,2,6,['E','P','D'],'roi',rpc,finished,poll_seconds=.001)
            await writer
            self.assertGreaterEqual(audit['begin_completed'],2)
            self.assertLess(audit['all_collectors_open_completed'],6)
            self.assertGreaterEqual(audit['end_completed'],6)
            self.assertEqual(calls,[('E','profile/begin'),('P','profile/begin'),('D','profile/begin'),
                ('D','profile/end'),('P','profile/end'),('E','profile/end')])
            self.assertTrue(all('end' in row for row in records.values()))

    async def test_a_partial_last_row_does_not_advance_the_boundary(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'rows';path.write_text('{}\n{"unfinished":')
            self.assertEqual(completed_requests(path),1)

    async def test_finished_or_expired_interval_does_not_start_collectors(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'rows';finished=asyncio.Event();finished.set();calls=[]
            def rpc(*args):calls.append(args)
            with self.assertRaisesRegex(AssertionError,'Serving ended'):
                await collect(path,2,6,['E'],'roi',rpc,finished,poll_seconds=.001)
            path.write_text('{}\n'*8)
            with self.assertRaisesRegex(AssertionError,'interval already ended'):
                await collect(path,2,6,['E'],'roi',rpc,finished,poll_seconds=.001)
            self.assertEqual(calls,[])

    async def test_partial_setup_failure_closes_only_started_collectors(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'rows';path.write_text('{}\n'*2);calls=[]
            def rpc(url,operation,body):
                calls.append((url,operation))
                if url=='D' and operation=='profile/begin':raise RuntimeError('setup failure')
                return [{'passed':True}]
            with self.assertRaisesRegex(RuntimeError,'setup failure'):
                await collect(path,2,6,['E','P','D'],'roi',rpc,asyncio.Event(),poll_seconds=.001)
            self.assertEqual(calls[-2:],[('P','profile/end'),('E','profile/end')])


if __name__=='__main__':unittest.main()
