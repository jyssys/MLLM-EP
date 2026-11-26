"""Observe actual stock P chunk boundaries without clamping or holding requests."""
import atexit
from itertools import chain
import json
import os
import re
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus

from ..mllm_epd_prefill_heavy_token_split.common import ROOT
from .prefill_progress import candidates


_P_REQUEST_ID = re.compile(r"chatcmpl-(.+):P(?:-[a-zA-Z0-9]{8})?")
_OBSERVER = None


def runtime_id(request_id):
    match = _P_REQUEST_ID.fullmatch(request_id)
    return match[1] if match else None


def flush_stock_trace():
    if _OBSERVER is None:
        raise RuntimeError("No stock trace scheduler in this engine process")
    return _OBSERVER.flush_trace()


class StockCutScheduler(AsyncScheduler):
    """Keep stock scheduling and defer diagnostic JSON work until EngineCore exits."""

    def __init__(self, *args, **kwargs):
        global _OBSERVER
        super().__init__(*args, **kwargs)
        if os.environ.get("MLLM_EPD_ROLE") != "prefill" or not self.vllm_config.scheduler_config.async_scheduling:
            raise ValueError("Stock-cut instrumentation requires the original asynchronous P scheduler")
        self._cut_scheduled = {}
        self._cut_trace = []
        self._cut_dropped = 0
        self._cut_log = ROOT / "raw" / f"stock_prefill_chunks_{os.getpid()}.jsonl"
        atexit.register(self._cut_flush)
        _OBSERVER = self

    def flush_trace(self):
        receipt = {"role": "prefill", "path": str(self._cut_log), "buffer_rows": len(self._cut_trace),
                   "dropped_steps": self._cut_dropped}
        self._cut_flush()
        return receipt

    def _cut_flush(self):
        if not self._cut_trace and not self._cut_dropped:
            return
        with self._cut_log.open("a") as stream:
            for row in self._cut_trace:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if self._cut_dropped:
                stream.write(json.dumps({"dropped_steps": self._cut_dropped}) + "\n")
        self._cut_trace.clear()
        self._cut_dropped = 0

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            self._cut_flush()

    def _update_after_schedule(self, scheduler_output):
        super()._update_after_schedule(scheduler_output)
        rows = []
        for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            end = request.num_computed_tokens
            if not 0 <= end - scheduled < end <= request.num_prompt_tokens:
                continue
            rows.append({
                "engine_id": request_id, "request_id": runtime_id(request_id),
                "computed_from": end - scheduled, "computed_through": end,
                "prompt_tokens": request.num_prompt_tokens, "preemptions": request.num_preemptions,
            })
        if rows:
            self._cut_scheduled[id(scheduler_output)] = (time.monotonic(), rows)

    def update_from_output(self, scheduler_output, model_runner_output):
        scheduled = self._cut_scheduled.pop(id(scheduler_output), None)
        if scheduled is None:
            return super().update_from_output(scheduler_output, model_runner_output)
        began, rows = scheduled
        returned = time.monotonic()
        for row in rows:
            req = self.requests.get(row["engine_id"])
            row["valid_retirement"] = req is not None and req.num_preemptions == row["preemptions"]
            row["later_tokens_inflight"] = (
                max(0, req.num_in_flight_tokens - row["computed_through"] + row["computed_from"])
                if row["valid_retirement"] else None
            )
            if row["valid_retirement"] and row["computed_through"] < row["prompt_tokens"]:
                row["competitors"] = candidates(self, row["engine_id"], scheduler_output.num_scheduled_tokens)
                row["ready_competitors"] = sum(other is not row and
                    other["computed_through"] == other["prompt_tokens"] for other in rows)
        elapsed_ms = 1000 * (returned - began)
        observed = {
            "issued": began, "worker_result_at": returned, "issue_to_result_ms": elapsed_ms,
            "batch_size": len(scheduler_output.num_scheduled_tokens),
            "batch_tokens": scheduler_output.total_num_scheduled_tokens,
            "running": len(self.running),
            "waiting": len(self.waiting) + len(self.skipped_waiting), "rows": rows,
        }
        if len(self._cut_trace) < 16384:
            self._cut_trace.append(observed)
        else:
            self._cut_dropped += 1
        return super().update_from_output(scheduler_output, model_runner_output)


class DecodeQueueScheduler(AsyncScheduler):
    """Record scheduler state transitions on the same host clock as P."""

    def __init__(self, *args, **kwargs):
        global _OBSERVER
        super().__init__(*args, **kwargs)
        if os.environ.get("MLLM_EPD_ROLE") != "decode" or not self.vllm_config.scheduler_config.async_scheduling:
            raise ValueError("Decode queue instrumentation requires the original asynchronous D scheduler")
        self._queue_trace = []
        self._last_queue = None
        self._queue_dropped = 0
        self._queue_log = ROOT / "raw" / f"stock_decode_queue_{os.getpid()}.jsonl"
        self._record_queue("start")
        _OBSERVER = self

    def _record_queue(self, source):
        state = {
            "running": len(self.running), "waiting": len(self.waiting),
            "skipped_waiting": len(self.skipped_waiting),
            "remote_KV_wait": sum(req.status is RequestStatus.WAITING_FOR_REMOTE_KVS
                                  for req in chain(self.waiting, self.skipped_waiting)),
            "generation_phase_running": sum(req.num_computed_tokens >= req.num_prompt_tokens for req in self.running),
            "KV_free_blocks": self.kv_cache_manager.block_pool.get_num_free_blocks(),
        }
        if state == self._last_queue and source not in ("checkpoint", "end"):
            return
        self._last_queue = state
        if len(self._queue_trace) < 65536:
            self._queue_trace.append({"time": time.monotonic(), "source": source, **state})
        else:
            self._queue_dropped += 1

    def add_request(self, request):
        super().add_request(request)
        self._record_queue("add_request")

    def schedule(self, throttle_prefills=False):
        output = super().schedule(throttle_prefills)
        self._record_queue("schedule")
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        output = super().update_from_output(scheduler_output, model_runner_output)
        self._record_queue("update_from_output")
        return output

    def flush_trace(self, source="checkpoint"):
        self._record_queue(source)
        receipt = {"role": "decode", "path": str(self._queue_log), "buffer_rows": len(self._queue_trace),
                   "dropped_steps": self._queue_dropped}
        with self._queue_log.open("a") as stream:
            for row in self._queue_trace:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if self._queue_dropped:
                stream.write(json.dumps({"dropped_steps": self._queue_dropped}) + "\n")
        self._queue_trace.clear()
        self._queue_dropped = 0
        return receipt

    def shutdown(self):
        try:
            self.flush_trace("end")
        finally:
            super().shutdown()
