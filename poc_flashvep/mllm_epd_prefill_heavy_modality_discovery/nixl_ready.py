"""Observe native NIXL READ posting and completion on the owned D worker."""

import atexit
import inspect
import json
import os
from pathlib import Path
import threading
import time


MAX_TRACKED_READS = 4096
MAX_BUFFERED_REQUESTS = 8192
READ_PARAMETERS = (
    "read_spec", "dst_engine_id", "request_id", "remote_request_id",
    "local_xfer_side_handle", "remote_xfer_side_handle",
)


class NixlReadObserver:
    def __init__(self, worker, output):
        self.worker = worker
        self.agent = worker.nixl_wrapper
        self.output = output
        self.installed_ns = time.monotonic_ns()
        self.transfer_id = 0
        self.by_handle = {}
        self.by_request = {}
        self.tracked_read_count = 0
        self.context = threading.local()
        self.completed = []
        self.written = 0
        self.dropped_inflight = 0
        self.dropped_buffered = 0
        self.read_blocks = worker._read_blocks
        self.post_read = self.agent.transfer
        self.check_state = self.agent.check_xfer_state
        self.get_telemetry = self.agent.get_xfer_telemetry
        self.handle_failure = worker._handle_failed_transfer
        self.get_finished = worker.get_finished

    def install(self):
        self.worker._read_blocks = self.observe_read_blocks
        self.agent.transfer = self.observe_post
        self.agent.check_xfer_state = self.observe_status
        self.agent.get_xfer_telemetry = self.observe_telemetry
        self.worker._handle_failed_transfer = self.observe_failure
        self.worker.get_finished = self.observe_finished

    def observe_read_blocks(self, read_spec, dst_engine_id, request_id, remote_request_id,
                            local_xfer_side_handle, remote_xfer_side_handle):
        previous = getattr(self.context, "read", None)
        self.context.read = (request_id, remote_request_id, dst_engine_id, read_spec.remote_rank)
        try:
            return self.read_blocks(read_spec, dst_engine_id, request_id, remote_request_id,
                                    local_xfer_side_handle, remote_xfer_side_handle)
        finally:
            self.context.read = previous

    def observe_post(self, handle, *args, **kwargs):
        context = getattr(self.context, "read", None)
        if context is None:
            return self.post_read(handle, *args, **kwargs)
        begin = time.monotonic_ns()
        status = "RAISED"
        try:
            status = self.post_read(handle, *args, **kwargs)
            return status
        finally:
            end = time.monotonic_ns()
            request_id, remote_request_id, remote_engine, remote_rank = context
            if self.tracked_read_count >= MAX_TRACKED_READS:
                self.dropped_inflight += 1
                if request_id in self.by_request:
                    self.by_request[request_id]["dropped_reads"] += 1
            else:
                self.transfer_id += 1
                handle_key = id(handle)
                if handle_key in self.by_handle:
                    # If native reused a handle before retiring its old request, neither
                    # transfer has an unambiguous completion witness.
                    self.dropped_inflight += 1
                    old_request_id, _ = self.by_handle.pop(handle_key)
                    self.by_request[old_request_id]["dropped_reads"] += 1
                row = {
                    "read_id": self.transfer_id, "remote_request_id": remote_request_id,
                    "remote_engine_id": remote_engine, "remote_rank": remote_rank,
                    "post_begin_ns": begin, "post_return_ns": end, "post_status": status,
                    "completion_lower_bound_ns": begin,
                    "completion_upper_bound_ns": end if status == "DONE" else None,
                    "last_PROC_query_begin_ns": None, "last_PROC_query_return_ns": None,
                    "first_DONE_query_begin_ns": None, "first_DONE_query_return_ns": None,
                    "native_poll_count": 0,
                }
                group = self.by_request.setdefault(request_id, {
                    "reads": {}, "native_failures": 0, "dropped_reads": 0,
                })
                group["reads"][self.transfer_id] = (handle_key, row)
                self.by_handle[handle_key] = (request_id, row)
                self.tracked_read_count += 1

    def observe_status(self, handle):
        tracked = self.by_handle.get(id(handle))
        if tracked is None:
            return self.check_state(handle)
        _, row = tracked
        begin = time.monotonic_ns()
        status = self.check_state(handle)
        end = time.monotonic_ns()
        row["native_poll_count"] += 1
        row["last_native_status"] = status
        if status == "PROC" and row["completion_upper_bound_ns"] is None:
            row["last_PROC_query_begin_ns"] = begin
            row["last_PROC_query_return_ns"] = end
            row["completion_lower_bound_ns"] = begin
        elif status == "DONE" and row["first_DONE_query_return_ns"] is None:
            row["first_DONE_query_begin_ns"] = begin
            row["first_DONE_query_return_ns"] = end
            if row["completion_upper_bound_ns"] is None:
                row["completion_upper_bound_ns"] = end
        return status

    def observe_telemetry(self, handle):
        result = self.get_telemetry(handle)
        tracked = self.by_handle.get(id(handle))
        if tracked is not None:
            _, row = tracked
            try:
                row["native_telemetry"] = {
                    "bytes": int(result.totalBytes), "descriptors": int(result.descCount),
                    "post_duration_us": int(result.postDuration), "transfer_duration_us": int(result.xferDuration),
                }
            except (TypeError, ValueError, AttributeError, OverflowError):
                row["native_telemetry"] = None
        return result

    def observe_failure(self, req_id, handle):
        record = self.by_request.get(req_id)
        if record is not None:
            record["native_failures"] += 1
        return self.handle_failure(req_id, handle)

    def observe_finished(self):
        if not self.by_request:
            return self.get_finished()
        begin = time.monotonic_ns()
        result = self.get_finished()
        end = time.monotonic_ns()
        _, finished_receiving = result
        for request_id in finished_receiving:
            tracked = self.by_request.pop(request_id, None)
            if tracked is None:
                continue
            reads = []
            for handle_key, row in tracked["reads"].values():
                current = self.by_handle.get(handle_key)
                if current is not None and current[1] is row:
                    del self.by_handle[handle_key]
                reads.append(row)
            self.tracked_read_count -= len(reads)
            native_pass = (not tracked["native_failures"] and not tracked["dropped_reads"]
                           and all(row["post_status"] in ("PROC", "DONE")
                                   and row["first_DONE_query_return_ns"] is not None for row in reads))
            if native_pass:
                for row in reads:
                    lower = row["completion_lower_bound_ns"]
                    upper = row["completion_upper_bound_ns"]
                    returned = row["post_return_ns"]
                    row["possible_read_enclosure_ns"] = [row["post_begin_ns"], upper]
                    if row["post_status"] == "PROC":
                        row["post_return_to_observed_done_ns"] = [returned, upper]
                        if lower > returned:
                            row["definite_outstanding_after_post_ns"] = [returned, lower]
            if len(self.completed) >= MAX_BUFFERED_REQUESTS:
                self.dropped_buffered += 1
                continue
            self.completed.append({
                "request_id": request_id, "evidence": "DIAGNOSTIC",
                "get_finished_begin_ns": begin, "get_finished_return_ns": end,
                "worker_kv_ready_observed_ns": end if native_pass else None,
                "native_failures": tracked["native_failures"], "dropped_reads": tracked["dropped_reads"],
                "reads": reads,
            })
        return result

    def receipt(self):
        return {
            "enabled": True, "decode_worker_pid": os.getpid(), "installed_ns": self.installed_ns,
            "path": str(self.output), "written_request_rows": self.written,
            "buffered_request_rows": len(self.completed), "tracked_requests": len(self.by_request),
            "tracked_handles": len(self.by_handle), "tracked_read_rows": self.tracked_read_count,
            "dropped_inflight_reads": self.dropped_inflight,
            "dropped_buffered_requests": self.dropped_buffered,
            "scope": "Host-monotonic brackets around original NIXL async READ post and existing PROC/DONE. "
                     "Completion may precede DONE observation. The last PROC call entry is a lower bound "
                     "on completion when present. Request readiness is the D worker's get_finished return "
                     "before scheduler consumption. Native telemetry is read exclusively from calls already "
                     "made by the worker; outstanding intervals and telemetry durations are not exclusive DMA time.",
        }

    def flush(self):
        if self.completed:
            with self.output.open("a", encoding="utf-8") as stream:
                stream.writelines(json.dumps(row) + "\n" for row in self.completed)
            self.written += len(self.completed)
            self.completed.clear()
        result = self.receipt()
        self.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


_observer = None


def install_nixl_ready_observer():
    """Opt in from the D worker info RPC before the first measured request."""
    global _observer
    if os.environ.get("MLLM_EPD_NIXL_READY") != "1" or os.environ.get("MLLM_EPD_ROLE") != "decode":
        return {"enabled": False}
    if _observer is not None:
        return _observer.receipt()
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import NixlPullConnectorWorker

    if not has_kv_transfer_group():
        return {"enabled": False, "reason": "The decode KV transfer group is unavailable"}
    worker = get_kv_transfer_group().connector_worker
    if (not isinstance(worker, NixlPullConnectorWorker) or worker.world_size != 1 or worker.use_host_buffer
            or worker.vllm_config.kv_transfer_config.kv_role != "kv_consumer"):
        return {"enabled": False, "reason": "Expected the native TP1 direct-device NIXL pull consumer"}
    if tuple(inspect.signature(worker._read_blocks).parameters) != READ_PARAMETERS:
        return {"enabled": False, "reason": "Installed NIXL READ signature differs from the audited source"}
    from ..mllm_epd_prefill_heavy_token_split.common import ROOT

    output = ROOT / "raw" / f"nixl_read_decode_{os.getpid()}.jsonl"
    _observer = NixlReadObserver(worker, output)
    _observer.install()
    atexit.register(_observer.flush)
    return _observer.receipt()


def flush_nixl_ready_observer():
    """Flush already observed rows outside the measured serving interval."""
    return _observer.flush() if _observer is not None else {"enabled": False}
