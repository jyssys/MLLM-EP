"""Protocol-level regressions; no model or GPU needed."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from aiohttp import web

from run_request_trace import run_one


class StreamProtocols(unittest.IsolatedAsyncioTestCase):
    async def check_stream(self, backend, wire, expected_text, expected_ids):
        async def endpoint(request):
            body = await request.json()
            if backend == "fastpp":
                self.assertEqual(body["rid"], "protocol_test:test")
            response = web.StreamResponse()
            await response.prepare(request)
            # Deliberately split JSON/SSE framing across network chunks.
            for offset in range(0, len(wire), 7):
                await response.write(wire[offset:offset + 7])
                await asyncio.sleep(0)
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/generate", endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory(prefix="successor-client-test-") as tmp:
                args = SimpleNamespace(backend=backend, run_id="protocol_test",
                                       out=Path(tmp)/"requests.jsonl",
                                       url=f"http://127.0.0.1:{port}")
                row = dict(request_id="test", arrival_s=0, prompt="fixture",
                           max_new_tokens=2)
                async with aiohttp.ClientSession() as session:
                    record = await run_one(session, row, time.perf_counter(), args)
                self.assertEqual(record["status"], "PASS")
                self.assertEqual(record["output_tokens"], 2)
                self.assertEqual(record["output_text"], expected_text)
                self.assertEqual(record["output_ids"], expected_ids)
                self.assertFalse(record["stream_coalesced"])
                self.assertEqual(len(record["stream_events"]), 2)
        finally:
            await runner.cleanup()

    async def test_layered_delta(self):
        wire = b"".join(json.dumps(item).encode()+b"\n" for item in [
            {"generated_text": "4", "output_tokens": [19]},
            {"generated_text": "2", "output_tokens": [17]}])
        await self.check_stream("layered", wire, "42", [19, 17])

    async def test_fastpp_cumulative(self):
        wire = b"".join(b"data: "+json.dumps(item).encode()+b"\n\n" for item in [
            {"text": "4", "meta_info": {"completion_tokens": 1}},
            {"text": "42", "meta_info": {"completion_tokens": 2}},
            {"text": "42", "meta_info": {"completion_tokens": 2}}])
        await self.check_stream("fastpp", wire+b"data: [DONE]\n\n", "42", None)


if __name__ == "__main__":
    unittest.main()
