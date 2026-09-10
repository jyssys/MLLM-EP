"""Read-only worker observer for MoE CUDA spans and backend-path evidence.

Enabled only when ``RANKFANOUT_TRACE_DIR`` is set.  The hook never changes a
route or output.  It synchronizes at the end of a scheduler execution step to
resolve CUDA events, so its timing is profiling-only and must not be used as
clean request latency.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import re
import threading
import time
from pathlib import Path


_ROOT = os.environ.get("RANKFANOUT_TRACE_DIR")
if _ROOT:
    import torch

    _TLS = threading.local()
    _LOCK = threading.Lock()
    _LAYER_IDS: dict[int, int] = {}
    _PENDING: list[dict] = []
    _ROWS: list[dict] = []
    _PLACEMENT_WRITTEN: set[str] = set()
    _DIRECT_CAPTURED: dict[tuple[str, str, int, int, int], int] = {}
    _INSTALLED = False

    def _context() -> dict:
        path = os.environ.get("RANKFANOUT_CONTEXT_FILE")
        if not path:
            return {}
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return {}

    def _rank_info() -> dict:
        out = {"pid": os.getpid(), "dp_rank": -1, "tp_rank": -1, "ep_rank": -1, "ep_world": -1}
        try:
            from vllm.distributed import get_dp_group, get_ep_group, get_tp_group

            out.update(
                dp_rank=int(get_dp_group().rank_in_group),
                tp_rank=int(get_tp_group().rank_in_group),
                ep_rank=int(get_ep_group().rank_in_group),
                ep_world=int(get_ep_group().world_size),
            )
        except Exception:
            pass
        return out

    def _write() -> None:
        if not _ROWS:
            return
        root = Path(_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"moe_spans_pid{os.getpid()}.jsonl"
        with _LOCK, path.open("a", encoding="utf-8") as handle:
            for row in _ROWS:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            _ROWS.clear()

    def _resolve_pending() -> None:
        if not _PENDING:
            return
        torch.cuda.synchronize()
        for item in _PENDING:
            row = {key: value for key, value in item.items() if key not in {"start", "end"}}
            try:
                row["moe_total_ms"] = float(item["start"].elapsed_time(item["end"]))
            except Exception as exc:
                row["moe_total_ms"] = None
                row["timing_error"] = repr(exc)
            _ROWS.append(row)
        _PENDING.clear()
        _write()

    def _record_placement(expert_map, num_experts: int, backend_path: str) -> None:
        if expert_map is None:
            return
        rank = _rank_info()
        key = f"{rank['ep_rank']}:{backend_path}"
        if key in _PLACEMENT_WRITTEN:
            return
        try:
            values = expert_map.detach().to("cpu", non_blocking=False).to(torch.int64).tolist()
            owned = [index for index, local in enumerate(values) if int(local) >= 0]
            root = Path(_ROOT)
            root.mkdir(parents=True, exist_ok=True)
            payload = {
                **rank,
                "backend_path": backend_path,
                "num_experts_argument": int(num_experts),
                "global_to_local": values,
                "owned_global_experts": owned,
                "timestamp_ns": time.time_ns(),
            }
            (root / f"placement_{backend_path}_ep{rank['ep_rank']}_pid{os.getpid()}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            _PLACEMENT_WRITTEN.add(key)
        except Exception as exc:
            Path(_ROOT, f"placement_error_pid{os.getpid()}.txt").write_text(repr(exc))

    def _wrap_prepare(cls, method: str, backend_path: str) -> None:
        original = getattr(cls, method)
        if getattr(original, "_rankfanout_observer", False):
            return

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            # Modular signature after self: a1, weights, ids, num_experts,
            # expert_map, ... .  Keyword calls are supported as well.
            num_experts = kwargs.get("num_experts", args[3] if len(args) > 3 else -1)
            expert_map = kwargs.get("expert_map", args[4] if len(args) > 4 else None)
            _record_placement(expert_map, int(num_experts), backend_path)
            ctx = getattr(_TLS, "moe_context", None)
            if ctx is not None:
                ctx.setdefault("backend_calls", []).append(f"{backend_path}.{method}")
            return original(self, *args, **kwargs)

        wrapped._rankfanout_observer = True
        setattr(cls, method, wrapped)

    def _wrap_backend_call(cls, method: str, label: str) -> None:
        original = getattr(cls, method)
        if getattr(original, "_rankfanout_observer", False):
            return

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            ctx = getattr(_TLS, "moe_context", None)
            if ctx is not None:
                ctx.setdefault("backend_calls", []).append(label)
            return original(self, *args, **kwargs)

        wrapped._rankfanout_observer = True
        setattr(cls, method, wrapped)

    def install() -> None:
        global _INSTALLED
        if _INSTALLED:
            return
        _INSTALLED = True

        from vllm.forward_context import get_forward_context
        from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_rank
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock

        # vLLM 0.20's DeepEP warmup invokes MoE with a small dummy shape while
        # the routed-expert capturer still carries real DP padding metadata.
        # The stock capturer asserts because those dimensions cannot describe
        # a request.  Ignore only that startup-only mismatch; real request
        # shapes still take the unmodified capture path.  This is a read-only
        # instrumentation compatibility patch, not a backend behavior change.
        original_capture = RoutedExpertsCapturer.capture
        capture_warned = False

        @functools.wraps(original_capture)
        def capture_or_skip_dummy(self, layer_id, topk_ids):
            nonlocal capture_warned
            # Save the actual router result before the stock capturer copies it
            # through its request-slot buffer.  The latter returned zeros for
            # the v0.20 DP2 eager path in our smoke run, while this hook sees
            # the real top-k tensor passed by select_experts.  TP ranks route
            # identically, so TP0 is the non-duplicated evidence row.
            metadata = get_forward_context().dp_metadata
            context = _context()
            if metadata is not None and context.get("run_id") and int(get_tensor_model_parallel_rank()) == 0:
                counts = metadata.num_tokens_across_dp_cpu
                local_tokens = int(counts[self.dp_rank].item())
                total_tokens = int(counts.sum().item())
                rows = int(topk_ids.shape[0])
                selected = None
                if rows == total_tokens:
                    end = int(counts[: self.dp_rank + 1].sum().item())
                    selected = topk_ids[end - local_tokens : end]
                elif rows == local_tokens:
                    selected = topk_ids
                elif local_tokens <= rows < total_tokens:
                    selected = topk_ids[:local_tokens]
                key = (
                    str(context["run_id"]),
                    str(context.get("workload_id", "unknown")),
                    int(context.get("iteration", -1)),
                    int(self.dp_rank),
                    int(layer_id),
                )
                if selected is not None and int(selected.shape[0]) > _DIRECT_CAPTURED.get(key, -1):
                    import numpy as np

                    direct = Path(_ROOT) / "direct_routes"
                    direct.mkdir(parents=True, exist_ok=True)
                    name = f"{key[0]}_{key[1]}_it{key[2]}_dp{key[3]}_layer{key[4]}.npy"
                    np.save(direct / name, selected.detach().to("cpu", non_blocking=False).numpy().astype(np.int16))
                    _DIRECT_CAPTURED[key] = int(selected.shape[0])
            try:
                return original_capture(self, layer_id, topk_ids)
            except AssertionError as error:
                if "unexpected topk_ids batch dim" not in str(error):
                    raise
                if metadata is not None:
                    counts = metadata.num_tokens_across_dp_cpu
                    local_tokens = int(counts[self.dp_rank].item())
                    global_tokens = int(counts.sum().item())
                    rows = int(topk_ids.shape[0])
                    if local_tokens <= rows < global_tokens:
                        self._device_buffer[:local_tokens, layer_id, :] = topk_ids[:local_tokens, :]
                        return None
                if not capture_warned:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Skipping routed-expert capture for non-request DeepEP warmup: %s",
                        error,
                    )
                    capture_warned = True
                return None

        RoutedExpertsCapturer.capture = capture_or_skip_dummy

        original_init = Qwen3MoeDecoderLayer.__init__
        original_layer = Qwen3MoeDecoderLayer.forward
        original_moe = Qwen3MoeSparseMoeBlock.forward

        @functools.wraps(original_init)
        def layer_init(self, vllm_config, prefix=""):
            original_init(self, vllm_config, prefix)
            match = re.search(r"(?:layers|h)\.(\d+)", str(prefix))
            _LAYER_IDS[id(self)] = int(match.group(1)) if match else -1

        @functools.wraps(original_layer)
        def layer_forward(self, *args, **kwargs):
            old = getattr(_TLS, "layer", -1)
            _TLS.layer = _LAYER_IDS.get(id(self), -1)
            try:
                return original_layer(self, *args, **kwargs)
            finally:
                _TLS.layer = old

        @functools.wraps(original_moe)
        def moe_forward(self, hidden_states, *args, **kwargs):
            stream = torch.cuda.current_stream()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            ctx = {
                **_context(),
                **_rank_info(),
                "layer_id": int(getattr(_TLS, "layer", -1)),
                "num_tokens": int(hidden_states.shape[0]),
                "stream_id": int(stream.cuda_stream),
                "host_start_ns": time.perf_counter_ns(),
                "backend_calls": [],
            }
            old = getattr(_TLS, "moe_context", None)
            _TLS.moe_context = ctx
            start.record(stream)
            try:
                return original_moe(self, hidden_states, *args, **kwargs)
            finally:
                end.record(torch.cuda.current_stream())
                ctx["host_end_ns"] = time.perf_counter_ns()
                ctx["start"] = start
                ctx["end"] = end
                _PENDING.append(ctx)
                _TLS.moe_context = old

        Qwen3MoeDecoderLayer.__init__ = layer_init
        Qwen3MoeDecoderLayer.forward = layer_forward
        Qwen3MoeSparseMoeBlock.forward = moe_forward

        # Capture the real DP-local route directly at the router boundary.
        # This is independent of vLLM's request-slot return buffer and sees
        # the exact top-k rows subsequently consumed by the MoE runner.
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

        original_select_experts = BaseRouter.select_experts

        @functools.wraps(original_select_experts)
        def select_experts_with_direct_capture(self, hidden_states, router_logits, **kwargs):
            weights, ids = original_select_experts(self, hidden_states, router_logits, **kwargs)
            context = _context()
            layer_id = int(getattr(_TLS, "layer", -1))
            if context.get("run_id") and layer_id >= 0:
                import numpy as np

                dp_rank = int(get_dp_group().rank_in_group)
                ep_rank = int(get_ep_group().rank_in_group)
                key = (
                    str(context["run_id"]),
                    str(context.get("workload_id", "unknown")),
                    int(context.get("iteration", -1)),
                    ep_rank,
                    layer_id,
                )
                if int(ids.shape[0]) > _DIRECT_CAPTURED.get(key, -1):
                    direct = Path(_ROOT) / "direct_routes"
                    direct.mkdir(parents=True, exist_ok=True)
                    name = f"{key[0]}_{key[1]}_it{key[2]}_ep{key[3]}_layer{key[4]}.npy"
                    np.save(direct / name, ids.detach().to("cpu", non_blocking=False).numpy().astype(np.int16))
                    _DIRECT_CAPTURED[key] = int(ids.shape[0])
            return weights, ids

        BaseRouter.select_experts = select_experts_with_direct_capture

        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize
        from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import MoEPrepareAndFinalizeNaiveDPEPModular

        _wrap_prepare(DeepEPHTPrepareAndFinalize, "prepare_async", "deepep_ht")
        _wrap_prepare(MoEPrepareAndFinalizeNaiveDPEPModular, "prepare", "agrs")

        try:
            import deep_ep

            _wrap_backend_call(deep_ep.Buffer, "get_dispatch_layout", "deep_ep.Buffer.get_dispatch_layout")
            _wrap_backend_call(deep_ep.Buffer, "dispatch", "deep_ep.Buffer.dispatch")
            _wrap_backend_call(deep_ep.Buffer, "combine", "deep_ep.Buffer.combine")
        except Exception:
            pass
        try:
            from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

            _wrap_backend_call(AgRsAll2AllManager, "dispatch", "AgRsAll2AllManager.dispatch/all_gatherv")
            _wrap_backend_call(AgRsAll2AllManager, "combine", "AgRsAll2AllManager.combine/reduce_scatterv")
        except Exception:
            pass

        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            original_execute = GPUModelRunner.execute_model

            @functools.wraps(original_execute)
            def execute_model(self, *args, **kwargs):
                try:
                    return original_execute(self, *args, **kwargs)
                finally:
                    # Profiling-only barrier; clean TTFT runs omit this hook.
                    _resolve_pending()

            GPUModelRunner.execute_model = execute_model
        except Exception as exc:
            Path(_ROOT).mkdir(parents=True, exist_ok=True)
            Path(_ROOT, f"execute_hook_error_pid{os.getpid()}.txt").write_text(repr(exc))

    try:
        install()
    except Exception as exc:
        Path(_ROOT).mkdir(parents=True, exist_ok=True)
        Path(_ROOT, f"install_error_pid{os.getpid()}.txt").write_text(repr(exc))

    @atexit.register
    def _finish() -> None:
        try:
            _resolve_pending()
        except Exception:
            pass
