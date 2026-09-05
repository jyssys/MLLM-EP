"""Install a read-only route/timing observer in vLLM worker processes.

The hook is deliberately local to this PoC.  It observes the top-k tensor at
the FusedMoE boundary and calls the stock implementation unchanged.  A CUDA
event surrounds the stock MoE call; no routing, placement, or scheduler state
is modified.
"""
from __future__ import annotations

import json
import os
import threading
import time
import functools
from pathlib import Path


if os.environ.get("FLASHVEP_ONLINE_TRACE_DIR"):
    import numpy as np
    import torch

    _LOCK = threading.Lock()
    _COUNTER = 0
    _INSTALLED = False
    _TLS = threading.local()
    _STAGE_PATCHED = False
    _INVOCATION = 0

    def _rank_info() -> tuple[int, int, int]:
        try:
            from vllm.distributed import get_dp_group, get_ep_group
            ep = get_ep_group()
            dp = get_dp_group()
            return int(getattr(dp, "rank_in_group", 0)), int(getattr(ep, "rank_in_group", 0)), int(ep.world_size)
        except Exception:
            return 0, 0, 1

    def _append(record: dict, routes: np.ndarray | None = None) -> None:
        global _COUNTER
        root = Path(os.environ["FLASHVEP_ONLINE_TRACE_DIR"])
        (root / "routes").mkdir(parents=True, exist_ok=True)
        with _LOCK:
            _COUNTER += 1
            idx = _COUNTER
            if routes is not None:
                name = f"route_{idx:08d}_dp{record.get('dp_rank', 0)}_l{record.get('layer', -1)}.npz"
                np.savez_compressed(root / "routes" / name, topk_ids=routes.astype(np.int16, copy=False))
                record["route_file"] = str(Path("routes") / name)
            with (root / "invocations.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            # Keep a normalized stage table as well as the per-invocation row.
            # This is intentionally append-only/read-only with respect to the
            # vLLM execution path; it is consumed by the root-cause analyzer.
            for stage in record.get("stage_records", []):
                stage_row = dict(stage)
                stage_row.update({k: record.get(k) for k in (
                    "timestamp_ns", "local_invocation_id", "scheduler_iteration_id",
                    "route_id", "layer", "dp_rank", "ep_rank", "phase", "M",
                    "request_context")})
                with (root / "stages.jsonl").open("a", encoding="utf-8") as sfh:
                    sfh.write(json.dumps(stage_row, separators=(",", ":")) + "\n")

    def _ctx_stage(name: str, thunk):
        """Execute a backend call and capture same-device CUDA elapsed span."""
        ctx = getattr(_TLS, "moe_ctx", None)
        if ctx is None or not torch.cuda.is_available():
            return thunk()
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        t0 = time.perf_counter_ns()
        try:
            return thunk()
        finally:
            end.record(stream)
            ctx["stage_events"].append((name, start, end, int(stream.cuda_stream), t0))

    def _install_stage_wrappers() -> None:
        """Patch only local observer boundaries; stock backend calls unchanged."""
        global _STAGE_PATCHED
        if _STAGE_PATCHED:
            return
        _STAGE_PATCHED = True
        try:
            import deep_ep
            buf_cls = deep_ep.Buffer
            for method, label in (("get_dispatch_layout", "deepep_layout"),
                                   ("dispatch", "deepep_dispatch"),
                                   ("combine", "deepep_combine")):
                original = getattr(buf_cls, method)
                if getattr(original, "_flashvep_stage_wrapper", False):
                    continue
                @functools.wraps(original)
                def wrapped(self, *args, __orig=original, __label=label, **kwargs):
                    return _ctx_stage(__label, lambda: __orig(self, *args, **kwargs))
                wrapped._flashvep_stage_wrapper = True
                setattr(buf_cls, method, wrapped)
        except Exception as exc:
            # The main whole-MoE observer still remains useful if DeepEP is not
            # importable during interpreter bootstrap.
            try:
                Path(os.environ["FLASHVEP_ONLINE_TRACE_DIR"], "stage_wrapper_error.txt").write_text(
                    repr(exc), encoding="utf-8")
            except Exception:
                pass

    def _layer_id(layer: object) -> int:
        try:
            value = getattr(layer, "layer_id", None)
            if value is not None:
                return int(value)
        except Exception:
            pass
        text = str(getattr(layer, "layer_name", ""))
        if not text:
            text = str(getattr(getattr(layer, "mlp", None), "layer_name", ""))
        import re
        match = re.search(r"(?:layers|h)\.(\d+)", text)
        return int(match.group(1)) if match else -1

    def _route_features(ids: np.ndarray, ep_size: int) -> dict:
        if ids.ndim != 2 or ids.shape[0] == 0:
            return {"M": int(ids.shape[0]) if ids.ndim else 0, "top_k": 0}
        top_k = int(ids.shape[1])
        experts_per_rank = max(1, 128 // max(1, ep_size))
        dest = ids // experts_per_rank
        fanout = np.asarray([np.unique(row).size for row in dest], dtype=np.int16)
        ecounts = np.bincount(ids.reshape(-1), minlength=128).astype(int)
        rcounts = np.bincount(dest.reshape(-1), minlength=max(1, ep_size)).astype(int)
        active = ecounts[ecounts > 0]
        mean_e = float(active.mean()) if active.size else 0.0
        cv_e = float(active.std() / mean_e) if mean_e else 0.0
        p = ecounts[ecounts > 0].astype(float)
        p = p / p.sum() if p.size else p
        hhi = float((p * p).sum()) if p.size else 0.0
        ent = float(-(p * np.log(p + 1e-12)).sum()) if p.size else 0.0
        matrix = np.zeros((ep_size, ep_size), dtype=int)
        # With DP-local inputs, sender rank is the local EP rank; this matrix
        # records the token incidence geometry as a conservative source row.
        for d in dest.reshape(-1):
            matrix[0, int(d)] += 1
        return {
            "M": int(ids.shape[0]), "top_k": top_k,
            "total_assignments": int(ids.size),
            "active_experts": int((ecounts > 0).sum()),
            "expert_hist": ecounts.tolist(),
            "rank_loads": rcounts.tolist(),
            "rank_max_mean": float(rcounts.max() / rcounts.mean()) if rcounts.mean() else 0.0,
            "expert_max_mean": float(ecounts.max() / active.mean()) if active.size else 0.0,
            "expert_cv": cv_e, "expert_hhi": hhi, "expert_entropy": ent,
            "fanout_mean": float(fanout.mean()),
            "fanout_p10": float(np.quantile(fanout, .10)),
            "fanout_median": float(np.quantile(fanout, .50)),
            "fanout_p90": float(np.quantile(fanout, .90)),
            "fanout_max": int(fanout.max()),
            "fanout_f1": float((fanout == 1).mean()),
            "fanout_f2": float((fanout == 2).mean()),
            "fanout_f3": float((fanout == 3).mean()),
            "fanout_f4": float((fanout == 4).mean()),
            "sender_dest_matrix": matrix.tolist(),
        }

    def install() -> None:
        global _INSTALLED
        if _INSTALLED:
            return
        _INSTALLED = True
        from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import FusedMoEModularMethod
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

        from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

        original_layer_init = Qwen3MoeDecoderLayer.__init__
        original_layer_forward = Qwen3MoeDecoderLayer.forward
        def layer_init(self, vllm_config, prefix=""):
            original_layer_init(self, vllm_config, prefix)
            import re
            match = re.search(r"(?:layers|h)\.(\d+)", str(prefix))
            self._flashvep_layer_id = int(match.group(1)) if match else -1
        Qwen3MoeDecoderLayer.__init__ = layer_init
        def layer_forward(self, *args, **kwargs):
            previous = getattr(_TLS, "layer", -1)
            _TLS.layer = int(getattr(self, "_flashvep_layer_id", _layer_id(self)))
            try:
                return original_layer_forward(self, *args, **kwargs)
            finally:
                _TLS.layer = previous
        Qwen3MoeDecoderLayer.forward = layer_forward

        original_modular_apply = FusedMoEModularMethod.apply
        original_unquantized_apply = UnquantizedFusedMoEMethod.apply

        _install_stage_wrappers()

        # Expert stage is separated from dispatch/combine by instrumenting the
        # modular kernel's expert call.  The wrapped function only records
        # events; tensor values and execution order are untouched.
        try:
            from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
            original_expert = FusedMoEKernelModularImpl._fused_experts
            if not getattr(original_expert, "_flashvep_stage_wrapper", False):
                @functools.wraps(original_expert)
                def expert_wrapped(self, *args, **kwargs):
                    return _ctx_stage("expert", lambda: original_expert(self, *args, **kwargs))
                expert_wrapped._flashvep_stage_wrapper = True
                FusedMoEKernelModularImpl._fused_experts = expert_wrapped
        except Exception as exc:
            try:
                Path(os.environ["FLASHVEP_ONLINE_TRACE_DIR"], "expert_wrapper_error.txt").write_text(
                    repr(exc), encoding="utf-8")
            except Exception:
                pass

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
            global _INVOCATION
            _INVOCATION += 1
            local_invocation_id = _INVOCATION
            ids = topk_ids.detach().to("cpu").numpy()
            dp_rank, ep_rank, ep_size = _rank_info()
            phase = "decode" if ids.shape[0] <= 1 else "prefill"
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            # Diagnostic-only intervention: expose any outstanding work before
            # entering MoE.  This never runs in the baseline unless explicitly
            # requested by the experiment environment.
            if os.environ.get("FLASHVEP_SYNC_BEFORE_MOE") == "1":
                torch.cuda.synchronize()
            event_start.record(torch.cuda.current_stream())
            t0 = time.perf_counter_ns()
            t1 = t0
            ctx = {"stage_events": []}
            _TLS.moe_ctx = ctx
            original = (original_modular_apply
                        if isinstance(self, FusedMoEModularMethod)
                        else original_unquantized_apply)
            try:
                out = original(self, layer, x, topk_weights, topk_ids, shared_experts_input)
                event_end.record(torch.cuda.current_stream())
                event_end.synchronize()
                t1 = time.perf_counter_ns()
            finally:
                _TLS.moe_ctx = None
            try:
                cuda_ms = float(event_start.elapsed_time(event_end))
            except Exception:
                cuda_ms = (t1 - t0) / 1e6
            record = {
                "timestamp_ns": t0, "layer": int(getattr(_TLS, "layer", _layer_id(layer))), "dp_rank": dp_rank,
                "ep_rank": ep_rank, "ep_size": ep_size, "phase": phase,
                "wall_ms": (t1 - t0) / 1e6, "cuda_ms": cuda_ms,
                "local_invocation_id": local_invocation_id,
                "scheduler_iteration_id": local_invocation_id,
                "scheduler_iteration_source": "local_moe_invocation_proxy",
                "route_id": f"{os.environ.get('FLASHVEP_ONLINE_CONTEXT','unknown')}_dp{dp_rank}_i{local_invocation_id}_l{int(getattr(_TLS, 'layer', _layer_id(layer)))}",
                "request_context": os.environ.get("FLASHVEP_ONLINE_CONTEXT", "unknown"),
            }
            record.update(_route_features(ids, ep_size))
            stages = []
            for name, start, end, stream_id, stage_t0 in ctx["stage_events"]:
                try:
                    end.synchronize()
                    stage_cuda = float(start.elapsed_time(end))
                except Exception:
                    stage_cuda = None
                stages.append({"stage": name, "cuda_ms": stage_cuda,
                               "wall_ms": (time.perf_counter_ns() - stage_t0) / 1e6,
                               "stream_id": stream_id})
            record["stage_records"] = stages
            _append(record, ids)
            return out
        # In vLLM 0.20 an unquantized DeepEP layer normally retains
        # UnquantizedFusedMoEMethod (the modular class is used by some
        # quantization paths).  Patch both so the observer is robust to the
        # backend's selection without touching the stock call.
        UnquantizedFusedMoEMethod.apply = apply
        FusedMoEModularMethod.apply = apply

    try:
        install()
    except Exception as exc:
        Path(os.environ["FLASHVEP_ONLINE_TRACE_DIR"]).mkdir(parents=True, exist_ok=True)
        (Path(os.environ["FLASHVEP_ONLINE_TRACE_DIR"]) / "hook_error.txt").write_text(repr(exc), encoding="utf-8")
