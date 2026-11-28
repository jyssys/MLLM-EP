"""CPU-only tests of the native request observer's call and timing contracts."""

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from . import nixl_ready


class FakeAgent:
    def __init__(self):
        self.calls = Counter()
        self.plans = {}

    def transfer(self, handle):
        self.calls["post"] += 1
        status = self.plans[handle][0]
        if isinstance(status, Exception):
            raise status
        return status

    def check_xfer_state(self, handle):
        self.calls["poll"] += 1
        return next(self.plans[handle][1])

    def get_xfer_telemetry(self, handle):
        self.calls["telemetry"] += 1
        return SimpleNamespace(totalBytes=12_582_912, descCount=20, postDuration=11, xferDuration=151)


class FakeWorker:
    def __init__(self):
        self.nixl_wrapper = FakeAgent()
        self.active = {}
        self.failed = set()

    def _read_blocks(self, read_spec, dst_engine_id, request_id, remote_request_id,
                     local_xfer_side_handle, remote_xfer_side_handle):
        if read_spec.skip:
            return
        handle = read_spec.handle
        try:
            self.nixl_wrapper.transfer(handle)
        except RuntimeError:
            self._handle_failed_transfer(request_id, handle)
        else:
            self.active[request_id] = handle

    def _handle_failed_transfer(self, req_id, handle):
        self.failed.add(req_id)

    def get_finished(self):
        done = set()
        for request_id, handle in list(self.active.items()):
            status = self.nixl_wrapper.check_xfer_state(handle)
            if status == "PROC":
                continue
            if status == "DONE":
                self.nixl_wrapper.get_xfer_telemetry(handle)
            else:
                self._handle_failed_transfer(request_id, handle)
            done.add(request_id)
            del self.active[request_id]
        done.update(self.failed)
        self.failed.clear()
        return set(), done

    def submit(self, request_id, post, states=(), *, skip=False):
        handle = object()
        self.nixl_wrapper.plans[handle] = (post, iter(states))
        spec = SimpleNamespace(handle=handle, remote_rank=0, skip=skip)
        self._read_blocks(spec, "P-engine", request_id, "remote-" + request_id, 1, 2)


class NixlReadyTest(TestCase):
    def setUp(self):
        temporary = self.enterContext(TemporaryDirectory())
        self.output = Path(temporary) / "reads.jsonl"
        self.worker = FakeWorker()
        self.observer = nixl_ready.NixlReadObserver(self.worker, self.output)
        self.observer.install()

    def test_existing_calls_and_completion_bounds(self):
        self.worker.submit("asynchronous", "PROC", ("PROC", "DONE"))
        self.assertEqual(self.worker.get_finished(), (set(), set()))
        self.worker.submit("synchronous", "DONE", ("DONE",))
        self.worker.submit("post-failed", RuntimeError("native failure"))
        self.worker.submit("poll-failed", "PROC", ("ERR",))
        self.worker.submit("cache-hit", "PROC", skip=True)
        self.assertEqual(self.worker.get_finished()[1],
                         {"asynchronous", "synchronous", "post-failed", "poll-failed"})
        self.assertEqual(self.worker.nixl_wrapper.calls, {"post": 4, "poll": 4, "telemetry": 2})

        self.assertEqual(self.observer.flush()["written_request_rows"], 4)
        self.assertEqual(self.observer.flush()["written_request_rows"], 4)
        rows = {row["request_id"]: row for row in map(json.loads, self.output.read_text().splitlines())}
        self.assertEqual(len(rows), 4)
        asynchronous, = rows["asynchronous"]["reads"]
        self.assertEqual(asynchronous["remote_request_id"], "remote-asynchronous")
        self.assertEqual(asynchronous["native_telemetry"]["bytes"], 12_582_912)
        self.assertLess(asynchronous["post_return_ns"], asynchronous["last_PROC_query_begin_ns"])
        self.assertEqual(asynchronous["definite_outstanding_after_post_ns"],
                         [asynchronous["post_return_ns"], asynchronous["last_PROC_query_begin_ns"]])
        self.assertEqual(asynchronous["completion_upper_bound_ns"], asynchronous["first_DONE_query_return_ns"])
        self.assertGreaterEqual(rows["asynchronous"]["worker_kv_ready_observed_ns"],
                                asynchronous["first_DONE_query_return_ns"])
        synchronous, = rows["synchronous"]["reads"]
        self.assertEqual(synchronous["completion_upper_bound_ns"], synchronous["post_return_ns"])
        self.assertNotIn("definite_outstanding_after_post_ns", synchronous)
        self.assertIsNone(rows["post-failed"]["worker_kv_ready_observed_ns"])
        self.assertIsNone(rows["poll-failed"]["worker_kv_ready_observed_ns"])
        self.assertNotIn("cache-hit", rows)

    def test_observer_limit_preserves_native_requests(self):
        with patch.object(nixl_ready, "MAX_TRACKED_READS", 1):
            self.worker.submit("tracked", "PROC", ("DONE",))
            self.worker.submit("untracked", "PROC", ("DONE",))
            self.assertEqual(self.worker.get_finished()[1], {"tracked", "untracked"})
        receipt = self.observer.flush()
        self.assertEqual(receipt["written_request_rows"], 1)
        self.assertEqual(receipt["dropped_inflight_reads"], 1)
        self.assertEqual(receipt["tracked_read_rows"], 0)
