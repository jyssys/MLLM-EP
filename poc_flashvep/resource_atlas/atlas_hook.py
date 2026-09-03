"""Read-only NVTX ranges for Qwen3-VL/vLLM/DeepEP profiling.

This module is imported through ``sitecustomize`` in every vLLM worker.  It
only wraps existing Python function boundaries with NVTX ranges and writes a
small per-process proof file; it does not alter routing, streams, scheduling,
weights, or tensor values.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_INSTALLED = False
_LOCK = threading.Lock()
_COUNTS: Counter[str] = Counter()
_PATCHED: list[str] = []
_PROFILER_STARTED = False
_PROFILER_STOPPED = False


def _nvtx_push(name: str) -> None:
    try:
        import torch
        torch.cuda.nvtx.range_push(name)
        _COUNTS[name] += 1
    except Exception:
        pass


def _nvtx_pop() -> None:
    try:
        import torch
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _profiler_api_watch() -> None:
    """Start/stop Nsight's CUDA-profiler API inside each CUDA worker.

    vLLM creates CUDA-owning child processes after the driver is launched.  A
    signal-file watcher keeps the capture window out of model/NCCL warmup while
    making the API calls in the process that owns the CUDA context.  It is
    disabled by default and does not touch execution when the env flag is off.
    """
    global _PROFILER_STARTED, _PROFILER_STOPPED
    result = os.environ.get("FLASHVEP_ATLAS_RESULT_DIR")
    if os.environ.get("FLASHVEP_CUDA_PROFILER_API") != "1" or not result:
        return
    start = os.path.join(result, "cuda_profiler_start.signal")
    stop = os.path.join(result, "cuda_profiler_stop.signal")
    deadline = time.monotonic() + float(os.environ.get("FLASHVEP_PROFILER_WATCH_SECONDS", "180"))
    cudart = None
    while time.monotonic() < deadline and not _PROFILER_STOPPED:
        try:
            if os.path.exists(start) and not _PROFILER_STARTED:
                import torch
                if torch.cuda.is_available():
                    cudart = torch.cuda.cudart()
                    err = cudart.cudaProfilerStart()
                    _PROFILER_STARTED = int(err) == 0
                    with open(os.path.join(result, f"profiler_start_{os.getpid()}.json"), "w") as f:
                        json.dump({"pid": os.getpid(), "cuda_profiler_start_return": int(err), "started": _PROFILER_STARTED}, f)
            if os.path.exists(stop) and _PROFILER_STARTED and not _PROFILER_STOPPED:
                if cudart is None:
                    import torch
                    cudart = torch.cuda.cudart()
                err = cudart.cudaProfilerStop()
                _PROFILER_STOPPED = True
                with open(os.path.join(result, f"profiler_stop_{os.getpid()}.json"), "w") as f:
                    json.dump({"pid": os.getpid(), "cuda_profiler_stop_return": int(err)}, f)
                return
        except Exception as exc:
            try:
                with open(os.path.join(result, f"profiler_api_error_{os.getpid()}.txt"), "w") as f:
                    f.write(repr(exc) + "\n")
            except Exception:
                pass
            return
        time.sleep(0.01)


def _wrap(cls: type, method: str, name: str, *, classify: Callable | None = None) -> None:
    original = getattr(cls, method, None)
    if original is None or getattr(original, "_flashvep_atlas_wrapped", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        label = classify(self, args, kwargs) if classify is not None else name
        _nvtx_push(label)
        try:
            return original(self, *args, **kwargs)
        finally:
            _nvtx_pop()

    wrapped._flashvep_atlas_wrapped = True  # type: ignore[attr-defined]
    wrapped._flashvep_atlas_original = original  # type: ignore[attr-defined]
    setattr(cls, method, wrapped)
    _PATCHED.append(f"{cls.__module__}.{cls.__name__}.{method}->{name}")


def _decoder_label(_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    # Qwen3MoeDecoderLayer receives flattened token rows.  More than one row
    # is the reliable local prefill/decode discriminator; this is a label only.
    hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
    try:
        return "LLM_PREFILL" if int(hidden.shape[0]) > 1 else "LLM_DECODE"
    except Exception:
        return "LLM_PREFILL"


def _write_proof() -> None:
    path = os.environ.get("FLASHVEP_ATLAS_RESULT_DIR")
    if not path:
        return
    try:
        import torch
        cuda = torch.version.cuda
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
    except Exception:
        cuda, device = None, None
    proof = {
        "pid": os.getpid(),
        "cuda_device_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_local": device,
        "torch_cuda": cuda,
        "patched": _PATCHED,
        "nvtx_range_counts": dict(_COUNTS),
        "cuda_profiler_api_started": _PROFILER_STARTED,
        "cuda_profiler_api_stopped": _PROFILER_STOPPED,
    }
    try:
        import vllm
        proof["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception:
        proof["vllm_version"] = "unavailable"
    try:
        from vllm.distributed import get_ep_group
        ep = get_ep_group()
        proof["ep_rank"] = int(ep.rank_in_group)
        proof["ep_world_size"] = int(ep.world_size)
    except Exception:
        pass
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"atlas_hook_{os.getpid()}.json").write_text(json.dumps(proof, indent=2) + "\n")


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _INSTALLED = True
    if os.environ.get("FLASHVEP_CUDA_PROFILER_API") == "1":
        threading.Thread(target=_profiler_api_watch, name="flashvep-profiler-watch", daemon=True).start()
    # Imports are intentionally local: sitecustomize runs before vLLM starts.
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3_VisionMLP, Qwen3_VisionPatchEmbed, Qwen3_VisionPatchMerger,
        Qwen3_VisionBlock,
    )
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
        DeepEPHTPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    # Vision boundaries.
    _wrap(Qwen3_VisionPatchEmbed, "forward", "VISION_PATCH")
    _wrap(Qwen2_5_VisionAttention, "forward", "VISION_ATTN")
    _wrap(Qwen3_VisionMLP, "forward", "VISION_MLP")
    _wrap(Qwen3_VisionPatchMerger, "forward", "VISION_MERGER")

    # Language decoder/attention and router call boundary.  The sparse block
    # range includes the router and its production FusedMoE call; the exact
    # split is reported as source-inferred when no separate router range is
    # available in the installed vLLM path.
    _wrap(Qwen3MoeDecoderLayer, "forward", "LLM_PREFILL", classify=_decoder_label)
    _wrap(Qwen3MoeAttention, "forward", "LLM_ATTN")
    _wrap(Qwen3MoeSparseMoeBlock, "forward", "ROUTER_TOPK")
    _wrap(FusedMoEKernelModularImpl, "_fused_experts", "EXPERT_GEMM")

    # DeepEP HT's exact prepare/finalize Python boundaries.  The underlying
    # collectives retain their own communication stream and event semantics.
    _wrap(DeepEPHTPrepareAndFinalize, "_do_dispatch", "DEEPEP_DISPATCH")
    _wrap(DeepEPHTPrepareAndFinalize, "_finalize", "DEEPEP_COMBINE")

    # FusedMoE.forward is the installed router/expert entry.  This marker is
    # intentionally not called ROUTER_TOPK to avoid claiming finer timing.
    _wrap(FusedMoE, "forward", "LLM_MOE")
    atexit.register(_write_proof)
