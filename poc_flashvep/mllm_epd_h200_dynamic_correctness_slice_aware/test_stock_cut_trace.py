import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from .stock_cut_observer import DecodeQueueScheduler, StockCutScheduler, runtime_id


class StockCutTraceTests(TestCase):
    def test_scheduler_hooks_match_pinned_async_interfaces(self):
        for cls, names in (
            (StockCutScheduler, ("_update_after_schedule", "update_from_output", "shutdown")),
            (DecodeQueueScheduler, ("add_request", "schedule", "update_from_output", "shutdown")),
        ):
            for name in names:
                original = inspect.signature(getattr(AsyncScheduler, name))
                wrapped = inspect.signature(getattr(cls, name))
                self.assertEqual(list(original.parameters), list(wrapped.parameters))
        self.assertEqual(runtime_id("chatcmpl-native_common_b8_5_0042:P-67c182ae"),
                         "native_common_b8_5_0042")

    def test_explicit_checkpoint_persists_without_engine_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            prefill = StockCutScheduler.__new__(StockCutScheduler)
            prefill._cut_log = Path(directory) / "p.jsonl"
            prefill._cut_dropped = 0
            prefill._cut_trace = [{"worker_result_at": 10., "rows": [{"computed_through": 8192}]}]
            receipt = prefill.flush_trace()
            self.assertEqual(receipt["buffer_rows"], 1)
            self.assertEqual(json.loads(prefill._cut_log.read_text())["worker_result_at"], 10.)
            self.assertEqual(prefill._cut_trace, [])

            decode = DecodeQueueScheduler.__new__(DecodeQueueScheduler)
            decode._queue_log = Path(directory) / "d.jsonl"
            decode._queue_trace, decode._queue_dropped, decode._last_queue = [], 0, None
            decode.running, decode.waiting, decode.skipped_waiting = [], [], []
            decode.kv_cache_manager = SimpleNamespace(
                block_pool=SimpleNamespace(get_num_free_blocks=lambda: 1000))
            receipt = decode.flush_trace()
            self.assertEqual(receipt["buffer_rows"], 1)
            snapshot = json.loads(decode._queue_log.read_text())
            self.assertEqual(snapshot["source"], "checkpoint")
            self.assertEqual(snapshot["KV_free_blocks"], 1000)
