"""Keep full evidence serialization outside concurrent token receipt."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from . import client


class Content:
    def __init__(self, count, initial_delay):
        self.count=count;self.initial_delay=initial_delay

    async def iter_any(self):
        await asyncio.sleep(self.initial_delay)
        for i in range(self.count):
            value=dict(choices=[dict(token_ids=[i],delta=dict(content='x'))])
            if i==0:value['prompt_token_ids']=list(range(2000))
            yield b'data: '+json.dumps(value).encode()+b'\n\n'
            await asyncio.sleep(.002)
        yield b'data: '+json.dumps(dict(usage=dict(prompt_tokens=2000,
            completion_tokens=self.count),choices=[])).encode()+b'\n\n'


class Response:
    status=200
    def __init__(self, count, delay):self.content=Content(count,delay)
    async def __aenter__(self):return self
    async def __aexit__(self,*args):return False


class Session:
    def post(self,url,*,json,headers):return Response(json['count'],json['delay'])


async def receipt_repro(folder, sink=None):
    row=dict(request_id='a',prompt_SHA256='test',total_input_tokens=2000,
             source='fixture',stratum='fixture',image_count=1)
    other=dict(row,request_id='b')
    arrival=time.monotonic()
    kwargs={} if sink is None else dict(evidence_sink=sink)
    return await asyncio.gather(
        client.one(Session(),'http://fixture',row,dict(count=1,delay=0),arrival,'probe',0,**kwargs),
        client.one(Session(),'http://fixture',other,dict(count=16,delay=.001),arrival,'probe',1,**kwargs))


class ImmediateEvidenceRepro(unittest.TestCase):
    def test_heavy_completed_trace_can_block_another_stream(self):
        # Controlled service time reproduces the observed synchronous raw writer.
        def slow_append(path,row):
            if 'chunks' in row:time.sleep(.08)
        with TemporaryDirectory() as folder,patch.object(client,'ROOT',Path(folder)),\
             patch.object(client,'append',side_effect=slow_append):
            rows=asyncio.run(receipt_repro(folder))
        gap=max(rows[1]['ITL_ms'])
        self.assertGreater(gap,34.,'The receipt loop must reproduce the frozen ITL failure')
        print('IMMEDIATE_RAW_REPRO_MAX_ITL_MS',gap,flush=True)


class DeferredEvidenceTest(unittest.TestCase):
    def test_full_rows_are_identical_and_written_only_after_drain(self):
        from .client_evidence import DeferredEvidence
        with TemporaryDirectory() as folder:
            raw=Path(folder)/'full.jsonl';journal=Path(folder)/'completed.jsonl'
            sink=DeferredEvidence(raw,journal)
            with patch.object(client,'ROOT',Path(folder)),\
                 patch.object(client,'append',side_effect=AssertionError('Full append inside receipt loop')):
                rows=asyncio.run(receipt_repro(folder,sink))
            self.assertFalse(raw.exists())
            records=[json.loads(x) for x in journal.read_text().splitlines()]
            self.assertEqual(len(records),2)
            for record in records:
                self.assertNotIn('chunks',record)
                self.assertNotIn('prompt_token_ids',record)
                self.assertIn('token_times',record)
            proof=sink.flush()
            full=[json.loads(x) for x in raw.read_text().splitlines()]
            self.assertEqual({r['request_id']:r for r in full},{r['request_id']:r for r in rows})
            self.assertEqual(proof['full_rows'],2)

    def test_duplicate_flush_cannot_duplicate_evidence(self):
        from .client_evidence import DeferredEvidence
        with TemporaryDirectory() as folder:
            raw=Path(folder)/'full.jsonl';journal=Path(folder)/'completed.jsonl'
            sink=DeferredEvidence(raw,journal)
            sink.record(dict(request_id='a',runtime_request_id='a',finish=1.,chunks=[]))
            sink.flush()
            with self.assertRaises(AssertionError):sink.flush()
            self.assertEqual(len(raw.read_text().splitlines()),1)


if __name__=='__main__':unittest.main()
