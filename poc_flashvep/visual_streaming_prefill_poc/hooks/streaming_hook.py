"""Read-only per-request vision and decoder timing for a bounded run.

The hook does not alter model values, routing, placement, or scheduling.  A
small control file is updated only at request boundaries by the driver.  CUDA
events are synchronized at the end of each naturally sequential vision or
decoder-layer call so the recorded intervals are real device durations.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_LOCK = threading.Lock()
_VISION_ROWS: list[dict[str, Any]] = []
_LAYER_ROWS: list[dict[str, Any]] = []
_VISION_CALL = 0
_ACTIVE_MTIME: int | None = None
_ACTIVE_ID = "unknown"
# The real-pipeline hook sets this while a second image encoder is launched on
# a side stream.  The ordinary PoC hook must not synchronize/copy the result in
# that narrow window, otherwise it would destroy the intended handoff.
_ASYNC_VISION = False


def _result_dir() -> Path | None:
    value = os.environ.get("FLASHVEP_STREAMING_RESULT_DIR")
    return Path(value) if value else None


def _active_id() -> str:
    global _ACTIVE_MTIME, _ACTIVE_ID
    path_value = os.environ.get("FLASHVEP_STREAMING_ACTIVE_PATH")
    if not path_value:
        return _ACTIVE_ID
    path = Path(path_value)
    try:
        mtime = path.stat().st_mtime_ns
        if mtime != _ACTIVE_MTIME:
            _ACTIVE_ID = path.read_text(encoding="utf-8").strip() or "unknown"
            _ACTIVE_MTIME = mtime
    except FileNotFoundError:
        pass
    return _ACTIVE_ID


def _event_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(value.device)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _save_tensor(call_id: int, active: str, output: torch.Tensor) -> str | None:
    root = _result_dir()
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    # Save one shard per CUDA worker.  Comparing the same PID across combined
    # and independent calls avoids TP-shard concatenation assumptions.
    name = f"vision_{call_id:04d}_{active}_pid{os.getpid()}.pt"
    path = root / "vision_outputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output.detach().float().cpu(), path)
    return str(path)


def _write_rows() -> None:
    root = _result_dir()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "vision": _VISION_ROWS, "decoder_layers": _LAYER_ROWS,
               "cuda_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
               "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "torch_cuda": torch.version.cuda}
    path = root / f"timing_worker_pid{os.getpid()}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n", encoding="utf-8")


def _append_live(kind: str, row: dict[str, Any]) -> None:
    """Persist rows before worker teardown (vLLM may terminate children abruptly)."""
    root = _result_dir()
    if root is None:
        return
    path = root / f"timing_worker_pid{os.getpid()}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **row}, default=_jsonable) + "\n")
    except Exception:
        pass


def _layer_index(self: Any) -> int:
    value = getattr(self, "_flashvep_streaming_layer", -1)
    return int(value)


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _INSTALLED = True

    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    vision_original = Qwen3_VisionTransformer.forward
    layer_original = Qwen3MoeDecoderLayer.forward
    init_original = Qwen3MoeDecoderLayer.__init__

    def vision_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _VISION_CALL
        active = _active_id()
        call_id = _VISION_CALL
        _VISION_CALL += 1
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        output = vision_original(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        if _ASYNC_VISION:
            # The caller owns the completion event and will establish a
            # dependency before the embedding is consumed.  Saving a CPU copy
            # here would implicitly synchronize the side stream.
            return output
        end.synchronize()
        duration = _event_elapsed(start, end)
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        grid_list = grid.tolist() if isinstance(grid, torch.Tensor) else grid
        output_path = _save_tensor(call_id, active, output)
        row = {"active_id": active, "call_id": call_id,
                             "duration_ms": duration,
                             "grid_thw": _jsonable(grid_list),
                             "input_shape": list(args[0].shape) if args and isinstance(args[0], torch.Tensor) else None,
                             "output_shape": list(output.shape) if isinstance(output, torch.Tensor) else None,
                             "output_dtype": str(output.dtype) if isinstance(output, torch.Tensor) else None,
                             "output_path": output_path,
                             "pid": os.getpid(),
                             "cuda_device": int(torch.cuda.current_device())}
        _VISION_ROWS.append(row)
        _append_live("vision", row)
        return output

    def init_layer(self: Any, *args: Any, **kwargs: Any) -> None:
        init_original(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
        self._flashvep_streaming_layer = int(match.group(1)) if match else -1

    def layer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        active = _active_id()
        hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
        kind = "prefill" if isinstance(hidden, torch.Tensor) and hidden.shape[0] > 1 else "decode"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        output = layer_original(self, *args, **kwargs)
        end.record(torch.cuda.current_stream())
        end.synchronize()
        row = {"active_id": active, "layer": _layer_index(self),
                            "stage": kind, "duration_ms": _event_elapsed(start, end),
                            "rows": int(hidden.shape[0]) if isinstance(hidden, torch.Tensor) else None,
                            "pid": os.getpid(), "cuda_device": int(torch.cuda.current_device())}
        _LAYER_ROWS.append(row)
        _append_live("decoder", row)
        return output

    vision_forward._flashvep_streaming_wrapped = True  # type: ignore[attr-defined]
    layer_forward._flashvep_streaming_wrapped = True  # type: ignore[attr-defined]
    Qwen3_VisionTransformer.forward = vision_forward
    Qwen3MoeDecoderLayer.__init__ = init_layer
    Qwen3MoeDecoderLayer.forward = layer_forward
    atexit.register(_write_rows)
