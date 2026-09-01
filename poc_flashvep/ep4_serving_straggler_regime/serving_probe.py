"""Read-only scheduler-iteration probe for the EP4 serving experiment.

The probe records the scheduler output received by each real vLLM GPU model
runner.  It does not alter scheduling, routing, or model execution.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_INSTALLED = False
_LOCK = threading.Lock()
_SEQ = 0
_CONTROL_CACHE: tuple[int, dict[str, Any]] | None = None


def _control() -> dict[str, Any]:
    global _CONTROL_CACHE
    path_text = os.environ.get("FLASHVEP_MATRIX_CONTROL")
    if not path_text:
        return {}
    path = Path(path_text)
    try:
        stamp = path.stat().st_mtime_ns
    except FileNotFoundError:
        return {}
    if _CONTROL_CACHE is not None and _CONTROL_CACHE[0] == stamp:
        return _CONTROL_CACHE[1]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    _CONTROL_CACHE = (stamp, value)
    return value


def _write(row: dict[str, Any]) -> None:
    directory = Path(os.environ["FLASHVEP_SCHEDULER_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"scheduler_dp{os.environ.get('VLLM_DP_RANK', 'na')}"
        f"_pid{os.getpid()}.jsonl"
    )
    with (directory / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.execute_model

    def patched(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        global _SEQ
        control = _control()
        with _LOCK:
            sequence = _SEQ
            _SEQ += 1
        new_ids = [str(item.req_id) for item in scheduler_output.scheduled_new_reqs]
        cached = scheduler_output.scheduled_cached_reqs
        cached_ids = [str(item) for item in getattr(cached, "req_ids", [])]
        row = {
            "sequence": sequence,
            "pid": os.getpid(),
            "dp_rank": int(os.environ.get("VLLM_DP_RANK", -1)),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "global_rank": os.environ.get("RANK"),
            "timestamp_ns": time.time_ns(),
            "control_wave": control.get("wave"),
            "batch_id": control.get("batch_id"),
            "condition": control.get("condition"),
            "concurrency": control.get("concurrency"),
            "control_iteration": control.get("iteration"),
            "phase": control.get("phase"),
            "measured": bool(control.get("measured", False)),
            "num_scheduled_tokens": {
                str(key): int(value)
                for key, value in scheduler_output.num_scheduled_tokens.items()
            },
            "total_num_scheduled_tokens": int(
                scheduler_output.total_num_scheduled_tokens
            ),
            "scheduled_new_req_ids": new_ids,
            "scheduled_cached_req_ids": cached_ids,
            "finished_req_ids": sorted(
                str(item) for item in scheduler_output.finished_req_ids
            ),
            "scheduled_encoder_inputs": {
                str(key): [int(v) for v in value]
                for key, value in scheduler_output.scheduled_encoder_inputs.items()
            },
        }
        started = time.perf_counter_ns()
        try:
            return original(self, scheduler_output, *args, **kwargs)
        finally:
            row["execute_model_cpu_ms"] = (time.perf_counter_ns() - started) / 1e6
            _write(row)

    GPUModelRunner.execute_model = patched
