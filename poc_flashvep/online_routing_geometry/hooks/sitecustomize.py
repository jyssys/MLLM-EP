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
    _EVENT_PATCHED = False
    _INVOCATION = 0
    _EXECUTION_STEP = 0
    _LAST_DISPATCH_MS = {}

    def _float_env(name: str, default: float | None = None) -> float | None:
        try:
            value = os.environ.get(name)
            return default if value in (None, "") else float(value)
        except Exception:
            return default

    def _oracle_ids() -> set[int]:
        raw = os.environ.get("FLASHVEP_ORACLE_SYNC_INVOCATIONS", "")
        out = set()
        for item in raw.split(","):
            try:
                if item.strip():
                    out.add(int(item.strip()))
            except ValueError:
                continue
        return out

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

    def _install_event_wait_wrapper() -> None:
        """Observe DeepEP EventOverlap waits without changing their semantics.

        EventHandle has no non-blocking ``query`` API in the installed DeepEP
        build.  We therefore record CUDA events around the existing
        ``current_stream_wait`` enqueue and resolve them only after the stock
        MoE call.  This measures the closest observable downstream dependency
        wait while preserving the original stream/event ordering.
        """
        global _EVENT_PATCHED
        if _EVENT_PATCHED:
            return
        try:
            from deep_ep import utils
            original_wait = utils.EventOverlap.current_stream_wait
            if getattr(original_wait, "_flashvep_wait_wrapper", False):
                _EVENT_PATCHED = True
                return

            @functools.wraps(original_wait)
            def wrapped_wait(self, *args, **kwargs):
                ctx = getattr(_TLS, "moe_ctx", None)
                if ctx is None or not torch.cuda.is_available():
                    return original_wait(self, *args, **kwargs)
                stream = torch.cuda.current_stream()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                try:
                    return original_wait(self, *args, **kwargs)
                finally:
                    end.record(stream)
                    ctx.setdefault("wait_events", []).append(
                        (start, end, int(stream.cuda_stream), time.perf_counter_ns()))

            wrapped_wait._flashvep_wait_wrapper = True
            utils.EventOverlap.current_stream_wait = wrapped_wait
            _EVENT_PATCHED = True
        except Exception as exc:
            try:
                Path(os.environ["FLASHVEP_ONLINE_TRACE_DIR"], "event_wait_wrapper_error.txt").write_text(
                    repr(exc), encoding="utf-8")
            except Exception:
                pass

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
                    ctx = getattr(_TLS, "moe_ctx", None)
                    previous_event = kwargs.get("previous_event")
                    if ctx is not None:
                        ctx.setdefault("previous_events", []).append({
                            "stage": __label,
                            "present": bool(previous_event is not None),
                            "event_handle_present": bool(
                                getattr(previous_event, "event", None) is not None),
                            "ready_query": "UNAVAILABLE_DEEPEP_EVENT_HANDLE",
                        })
                    # Narrow diagnostic intervention: drain only the DeepEP
                    # communication stream before stock dispatch.  The stream
                    # and all event dependencies are otherwise untouched.
                    if __label == "deepep_dispatch" and getattr(_TLS, "comm_drain_requested", False):
                        t0 = time.perf_counter_ns()
                        try:
                            comm = self.get_comm_stream()
                            if ctx is not None:
                                ctx["comm_stream_id"] = int(comm.cuda_stream)
                            comm.synchronize()
                            elapsed = (time.perf_counter_ns() - t0) / 1e6
                            if ctx is not None:
                                ctx["comm_drain_wall_ms"] = ctx.get("comm_drain_wall_ms", 0.0) + elapsed
                                ctx["intervention_applied"] = True
                        except Exception as exc:
                            if ctx is not None:
                                ctx["intervention_error"] = repr(exc)
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

    def _install_request_context_wrapper() -> None:
        """Attach the actual V1 scheduled request set to each MoE call.

        This is measurement-only.  ``SchedulerOutput.num_scheduled_tokens``
        is the authoritative worker-side request set; we do not infer request
        identity from flattened token positions.  That distinction is kept in
        the output so an offline report cannot accidentally claim exact
        per-request MoE attribution.
        """
        global _EXECUTION_STEP
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
            original_execute = GPUModelRunner.execute_model
            if getattr(original_execute, "_flashvep_request_wrapper", False):
                return

            @functools.wraps(original_execute)
            def wrapped_execute(self, scheduler_output, *args, **kwargs):
                global _EXECUTION_STEP
                _EXECUTION_STEP += 1
                old = getattr(_TLS, "execution_context", None)
                reqs = []
                try:
                    reqs = list(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
                except Exception:
                    pass
                file_wave = None
                file_labels = []
                try:
                    context_path = os.environ.get("FLASHVEP_ACTIVE_CONTEXT_FILE")
                    if context_path:
                        with open(context_path, encoding="utf-8") as fh:
                            context = json.load(fh)
                        file_wave = context.get("wave")
                        file_labels = list(context.get("request_labels", []))
                except Exception:
                    pass
                _TLS.execution_context = {
                    "scheduler_step_id": _EXECUTION_STEP,
                    "scheduled_request_ids": [str(x) for x in reqs],
                    "active_wave": os.environ.get("FLASHVEP_ACTIVE_WAVE") if os.environ.get("FLASHVEP_ACTIVE_WAVE") is not None else file_wave,
                    "active_request_labels": [x for x in os.environ.get("FLASHVEP_ACTIVE_REQUEST_LABELS", "").split(",") if x] or file_labels,
                }
                try:
                    return original_execute(self, scheduler_output, *args, **kwargs)
                finally:
                    _TLS.execution_context = old

            wrapped_execute._flashvep_request_wrapper = True
            GPUModelRunner.execute_model = wrapped_execute
        except Exception as exc:
            try:
                Path(os.environ["FLASHVEP_ONLINE_TRACE_DIR"], "request_context_wrapper_error.txt").write_text(repr(exc), encoding="utf-8")
            except Exception:
                pass
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
        _install_event_wait_wrapper()
        _install_request_context_wrapper()

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
            global _INVOCATION, _LAST_DISPATCH_MS
            _INVOCATION += 1
            local_invocation_id = _INVOCATION
            ids = topk_ids.detach().to("cpu").numpy()
            dp_rank, ep_rank, ep_size = _rank_info()
            phase = "decode" if ids.shape[0] <= 1 else "prefill"
            policy = os.environ.get("FLASHVEP_POLICY", "stock").strip().lower()
            threshold = _float_env("FLASHVEP_SIMPLE_PREV_DISPATCH_THRESHOLD_MS")
            prev_key = (dp_rank, ep_rank)
            prev_dispatch_ms = _LAST_DISPATCH_MS.get(prev_key)
            oracle_hit = local_invocation_id in _oracle_ids()
            simple_hit = (policy in {"online_simple", "simple", "p3"}
                          and threshold is not None
                          and prev_dispatch_ms is not None
                          and prev_dispatch_ms >= threshold)
            comm_drain_requested = (
                os.environ.get("FLASHVEP_COMM_STREAM_SYNC") == "1"
                or oracle_hit and os.environ.get("FLASHVEP_ORACLE_INTERVENTION", "comm_stream") == "comm_stream"
                or simple_hit)
            global_sync = os.environ.get("FLASHVEP_SYNC_BEFORE_MOE") == "1"
            if oracle_hit and os.environ.get("FLASHVEP_ORACLE_INTERVENTION", "comm_stream") == "global":
                global_sync = True
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            # Diagnostic-only intervention: expose any outstanding work before
            # entering MoE.  This never runs in the baseline unless explicitly
            # requested by the experiment environment.
            if global_sync:
                torch.cuda.synchronize()
            event_start.record(torch.cuda.current_stream())
            t0 = time.perf_counter_ns()
            t1 = t0
            ctx = {"stage_events": [], "wait_events": [], "previous_events": [],
                   "intervention_policy": policy,
                   "intervention_requested": bool(comm_drain_requested or global_sync),
                   "intervention_kind": ("global_sync" if global_sync
                                         else "comm_stream_drain" if comm_drain_requested
                                         else "none"),
                   "oracle_hit": bool(oracle_hit),
                   "simple_hit": bool(simple_hit),
                   "prev_dispatch_ms": prev_dispatch_ms,
                   "previous_event_ready": "UNAVAILABLE_DEEPEP_EVENT_HANDLE"}
            exec_ctx = getattr(_TLS, "execution_context", {}) or {}
            _TLS.moe_ctx = ctx
            _TLS.comm_drain_requested = comm_drain_requested
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
                _TLS.comm_drain_requested = False
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
                "scheduler_step_id": exec_ctx.get("scheduler_step_id"),
                "scheduled_request_ids": exec_ctx.get("scheduled_request_ids", []),
                "active_wave": exec_ctx.get("active_wave"),
                "active_request_labels": exec_ctx.get("active_request_labels", []),
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
            wait_values = []
            for start, end, stream_id, wait_t0 in ctx.get("wait_events", []):
                try:
                    end.synchronize()
                    wait_values.append(float(start.elapsed_time(end)))
                except Exception:
                    continue
            if wait_values:
                stages.append({"stage": "deepep_event_wait",
                               "cuda_ms": float(sum(wait_values)),
                               "wait_max_ms": float(max(wait_values)),
                               "wait_count": len(wait_values),
                               "wall_ms": float(sum(wait_values)),
                               "stream_id": int(ctx.get("comm_stream_id", 0))})
            record["stage_records"] = stages
            record.update({
                "previous_event_present": bool(any(z.get("present") for z in ctx.get("previous_events", []))),
                "previous_event_count": int(sum(bool(z.get("present")) for z in ctx.get("previous_events", []))),
                "previous_event_ready": ctx.get("previous_event_ready"),
                "previous_event_records": ctx.get("previous_events", []),
                "event_wait_count": len(wait_values),
                "event_wait_cuda_ms": float(sum(wait_values)) if wait_values else 0.0,
                "event_wait_max_ms": float(max(wait_values)) if wait_values else 0.0,
                "comm_stream_id": ctx.get("comm_stream_id"),
                "comm_drain_wall_ms": float(ctx.get("comm_drain_wall_ms", 0.0)),
                "intervention_policy": ctx.get("intervention_policy"),
                "intervention_requested": bool(ctx.get("intervention_requested")),
                "intervention_applied": bool(ctx.get("intervention_applied", global_sync)),
                "intervention_kind": ctx.get("intervention_kind"),
                "oracle_hit": bool(ctx.get("oracle_hit")),
                "simple_hit": bool(ctx.get("simple_hit")),
                "prev_dispatch_ms": ctx.get("prev_dispatch_ms"),
            })
            dispatch_values = [z.get("cuda_ms") for z in stages
                               if z.get("stage") == "deepep_dispatch" and z.get("cuda_ms") is not None]
            if dispatch_values:
                _LAST_DISPATCH_MS[prev_key] = float(dispatch_values[-1])
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
