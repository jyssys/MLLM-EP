"""Low-perturbation observer for the autonomous MoE-EP night sprint.

CUDA events are resolved only after ``query()`` reports completion.  Unlike
the inherited observer, this hook never synchronizes a layer and never copies
routing tensors to CPU on the default path.  It changes no model tensor,
routing decision, scheduler decision, or communication dependency.
"""
from __future__ import annotations

import atexit
import functools
import json
import os
import threading
import time
from pathlib import Path


_ROOT = os.environ.get("FLASHVEP_NIGHT_TRACE_DIR")
if _ROOT:
    import torch

    _LOCK = threading.Lock()
    _TLS = threading.local()
    _PENDING: list[dict] = []
    _ROWS: list[dict] = []
    _STEP = 0
    _INSTALLED = False
    _LAYER_IDS: dict[int, int] = {}
    _LAST_STEP_END_NS: int | None = None
    _SAMPLED_LAYERS = {
        int(x) for x in os.environ.get(
            "FLASHVEP_NIGHT_COMPONENT_LAYERS", "0,12,24,36,47"
        ).split(",") if x.strip()
    }

    def _rank_info() -> dict:
        out = {"dp_rank": -1, "ep_rank": -1, "tp_rank": -1, "pid": os.getpid()}
        try:
            from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
            out.update(
                dp_rank=int(get_dp_group().rank_in_group),
                ep_rank=int(get_ep_group().rank_in_group),
                tp_rank=int(get_tp_group().rank_in_group),
            )
        except Exception:
            pass
        return out

    def _context_file() -> dict:
        path = os.environ.get("FLASHVEP_NIGHT_CONTEXT_FILE")
        if not path:
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _event_pair(label: str, meta: dict | None = None):
        ctx = getattr(_TLS, "step_ctx", None)
        if ctx is None or not torch.cuda.is_available():
            return None
        # Keep the observer cheap enough for multi-hour online runs.  Step
        # timing is still recorded every iteration, while layer/component
        # timing is sampled at fixed early/middle/late layers.
        layer = int(getattr(_TLS, "layer", -1))
        if label in {"decoder_layer", "attention", "moe_block"} and layer not in _SAMPLED_LAYERS:
            return None
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        item = {
            "label": label,
            "start": start,
            "end": end,
            "stream_id": int(stream.cuda_stream),
            "layer": layer,
            "host_start_ns": time.perf_counter_ns(),
        }
        if meta:
            item.update(meta)
        ctx["events"].append(item)
        return item

    def _close_event(item) -> None:
        if item is not None:
            item["end"].record(torch.cuda.current_stream())
            item["host_end_ns"] = time.perf_counter_ns()

    def _write_rows(force: bool = False) -> None:
        flush_n = max(1, int(os.environ.get("FLASHVEP_NIGHT_FLUSH_EVERY", "64")))
        if not _ROWS or (not force and len(_ROWS) < flush_n):
            return
        root = Path(_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"steps_pid{os.getpid()}.jsonl"
        with _LOCK, path.open("a", encoding="utf-8") as fh:
            for row in _ROWS:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            _ROWS.clear()

    def _resolve(record: dict) -> dict:
        row = {k: v for k, v in record.items() if k not in {"step_start", "step_end", "events"}}
        try:
            row["step_cuda_ms"] = float(record["step_start"].elapsed_time(record["step_end"]))
        except Exception:
            row["step_cuda_ms"] = None
        stages = []
        for item in record["events"]:
            try:
                cuda_ms = float(item["start"].elapsed_time(item["end"]))
            except Exception:
                cuda_ms = None
            stages.append({
                "stage": item["label"],
                "layer": item.get("layer", -1),
                "M": item.get("M"),
                "cuda_ms": cuda_ms,
                "stream_id": item.get("stream_id"),
                "host_enqueue_us": (item.get("host_end_ns", item["host_start_ns"]) - item["host_start_ns"]) / 1e3,
            })
        row["stages"] = stages
        return row

    def _drain(force: bool = False) -> None:
        keep = []
        for record in _PENDING:
            ready = force
            if not ready:
                try:
                    ready = bool(record["step_end"].query())
                except Exception:
                    ready = False
            if ready:
                _ROWS.append(_resolve(record))
            else:
                keep.append(record)
        _PENDING[:] = keep
        _write_rows(force=force)

    def _wrap_stage(cls, method: str, label: str, meta_fn=None) -> None:
        original = getattr(cls, method)
        if getattr(original, "_flashvep_night", False):
            return

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            meta = meta_fn(self, args, kwargs) if meta_fn else None
            ev = _event_pair(label, meta)
            try:
                return original(self, *args, **kwargs)
            finally:
                _close_event(ev)

        wrapped._flashvep_night = True
        setattr(cls, method, wrapped)

    def install() -> None:
        global _INSTALLED, _STEP, _LAST_STEP_END_NS
        if _INSTALLED:
            return
        _INSTALLED = True

        from vllm.model_executor.models.qwen3_moe import (
            Qwen3MoeAttention,
            Qwen3MoeDecoderLayer,
            Qwen3MoeSparseMoeBlock,
        )

        original_init = Qwen3MoeDecoderLayer.__init__
        original_layer_forward = Qwen3MoeDecoderLayer.forward

        @functools.wraps(original_init)
        def layer_init(self, vllm_config, prefix=""):
            original_init(self, vllm_config, prefix)
            import re
            match = re.search(r"(?:layers|h)\.(\d+)", str(prefix))
            _LAYER_IDS[id(self)] = int(match.group(1)) if match else -1

        @functools.wraps(original_layer_forward)
        def layer_forward(self, *args, **kwargs):
            old = getattr(_TLS, "layer", -1)
            _TLS.layer = _LAYER_IDS.get(id(self), -1)
            ev = _event_pair("decoder_layer")
            try:
                return original_layer_forward(self, *args, **kwargs)
            finally:
                _close_event(ev)
                _TLS.layer = old

        Qwen3MoeDecoderLayer.__init__ = layer_init
        Qwen3MoeDecoderLayer.forward = layer_forward

        _wrap_stage(Qwen3MoeAttention, "forward", "attention")
        _wrap_stage(
            Qwen3MoeSparseMoeBlock,
            "forward",
            "moe_block",
            lambda _self, args, kwargs: {
                "M": int((args[0] if args else kwargs["hidden_states"]).shape[0])
            },
        )

        # DeepEP component enqueue spans are sampled on representative layers.
        # They are not interpreted as complete communication time unless the
        # returned dependency is consumed on the same stream.
        try:
            import deep_ep
            for name, label in (
                ("get_dispatch_layout", "deepep_layout_enqueue"),
                ("dispatch", "deepep_dispatch_enqueue"),
                ("combine", "deepep_combine_enqueue"),
            ):
                original = getattr(deep_ep.Buffer, name)
                if getattr(original, "_flashvep_night", False):
                    continue

                @functools.wraps(original)
                def deep_wrapped(self, *args, __original=original, __label=label, **kwargs):
                    if int(getattr(_TLS, "layer", -1)) not in _SAMPLED_LAYERS:
                        return __original(self, *args, **kwargs)
                    ev = _event_pair(__label)
                    try:
                        return __original(self, *args, **kwargs)
                    finally:
                        _close_event(ev)

                deep_wrapped._flashvep_night = True
                setattr(deep_ep.Buffer, name, deep_wrapped)
        except Exception as exc:
            Path(_ROOT).mkdir(parents=True, exist_ok=True)
            Path(_ROOT, f"component_hook_error_pid{os.getpid()}.txt").write_text(repr(exc), encoding="utf-8")

        # Source-derived diagnostic for the recurrent pageable CPU metadata
        # allocation/copy in the DeepEP prepare path.  This records host cost
        # only and never synchronizes CUDA, so it is not used as a production
        # optimization or a replacement for the stock metadata object.
        if os.environ.get("FLASHVEP_NIGHT_METADATA_TRACE") == "1":
            try:
                from vllm.model_executor.layers.fused_moe.modular_kernel import (
                    ExpertTokensMetadata,
                )
                original_meta = ExpertTokensMetadata.make_from_list
                if not getattr(original_meta, "_flashvep_night", False):
                    @functools.wraps(original_meta)
                    def make_meta(expert_num_tokens_list, device):
                        t0 = time.perf_counter_ns()
                        out = original_meta(expert_num_tokens_list, device)
                        path = Path(_ROOT) / f"metadata_pid{os.getpid()}.jsonl"
                        with _LOCK, path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({
                                "pid": os.getpid(),
                                "host_ms": (time.perf_counter_ns() - t0) / 1e6,
                                "num_experts": len(expert_num_tokens_list),
                                "sum_tokens": int(sum(expert_num_tokens_list)),
                                "device": str(device),
                                "timestamp_ns": time.time_ns(),
                            }, separators=(",", ":")) + "\n")
                        return out
                    make_meta._flashvep_night = True
                    ExpertTokensMetadata.make_from_list = staticmethod(make_meta)
            except Exception as exc:
                Path(_ROOT).mkdir(parents=True, exist_ok=True)
                Path(_ROOT, f"metadata_hook_error_pid{os.getpid()}.txt").write_text(repr(exc), encoding="utf-8")

        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        original_execute = GPUModelRunner.execute_model
        original_dummy_run = GPUModelRunner._dummy_run

        @functools.wraps(original_dummy_run)
        def dummy_run(self, *args, **kwargs):
            # DP coordination may invoke a one-token dummy forward while one
            # engine has no useful work.  Capture this explicitly so idle-rank
            # participation is distinguishable from an absent worker step.
            ev = _event_pair("dummy_run")
            try:
                return original_dummy_run(self, *args, **kwargs)
            finally:
                _close_event(ev)

        dummy_run._flashvep_night = True
        GPUModelRunner._dummy_run = dummy_run

        # `execute_dummy_batch` is an RPC on GPUWorker and can bypass
        # GPUModelRunner.execute_model entirely.  Log its occurrence separately
        # so a DP rank that has no useful requests is not mistaken for a silent
        # participant.  Only host duration is recorded here; no synchronize is
        # inserted in this path.
        try:
            from vllm.v1.worker.gpu_worker import Worker as GPUWorker
            original_execute_dummy_batch = GPUWorker.execute_dummy_batch

            @functools.wraps(original_execute_dummy_batch)
            def execute_dummy_batch(self, *args, **kwargs):
                t0 = time.perf_counter_ns()
                try:
                    return original_execute_dummy_batch(self, *args, **kwargs)
                finally:
                    path = Path(_ROOT) / f"dummy_pid{os.getpid()}.jsonl"
                    with _LOCK, path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "pid": os.getpid(), "host_ms":
                            (time.perf_counter_ns() - t0) / 1e6,
                            "timestamp_ns": time.time_ns(),
                        }, separators=(",", ":")) + "\n")

            execute_dummy_batch._flashvep_night = True
            GPUWorker.execute_dummy_batch = execute_dummy_batch
        except Exception as exc:
            Path(_ROOT).mkdir(parents=True, exist_ok=True)
            Path(_ROOT, f"dummy_hook_error_pid{os.getpid()}.txt").write_text(repr(exc), encoding="utf-8")

        @functools.wraps(original_execute)
        def execute(self, scheduler_output, *args, **kwargs):
            global _STEP, _LAST_STEP_END_NS
            _drain(False)
            _STEP += 1
            now_ns = time.perf_counter_ns()
            token_map = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
            counts = [int(v) for v in token_map.values()]
            num_decode = sum(v == 1 for v in counts)
            num_prefill = sum(v > 1 for v in counts)
            phase = "mixed" if num_decode and num_prefill else "decode" if num_decode else "prefill"
            ctx_file = _context_file()
            record = {
                "timestamp_ns": now_ns,
                "step_id": _STEP,
                "host_gap_ms": None if _LAST_STEP_END_NS is None else (now_ns - _LAST_STEP_END_NS) / 1e6,
                "phase": phase,
                "num_requests": len(counts),
                "num_decode_requests": num_decode,
                "num_prefill_requests": num_prefill,
                "scheduled_tokens": sum(counts),
                "scheduled_tokens_per_request": counts,
                "request_ids": [str(x) for x in token_map.keys()],
                "scheduled_encoder_inputs": {
                    str(k): list(v) for k, v in (getattr(scheduler_output, "scheduled_encoder_inputs", {}) or {}).items()
                },
                "context": ctx_file,
                "events": [],
                **_rank_info(),
            }
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
            record["step_start"] = start
            record["step_end"] = end
            old = getattr(_TLS, "step_ctx", None)
            _TLS.step_ctx = record
            host_start = time.perf_counter_ns()
            try:
                return original_execute(self, scheduler_output, *args, **kwargs)
            finally:
                record["host_execute_ms"] = (time.perf_counter_ns() - host_start) / 1e6
                end.record(torch.cuda.current_stream())
                _LAST_STEP_END_NS = time.perf_counter_ns()
                _TLS.step_ctx = old
                _PENDING.append(record)

        execute._flashvep_night = True
        GPUModelRunner.execute_model = execute

    def _finish() -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _drain(True)
        except Exception:
            pass

    atexit.register(_finish)
    try:
        install()
    except Exception as exc:
        Path(_ROOT).mkdir(parents=True, exist_ok=True)
        Path(_ROOT, f"hook_error_pid{os.getpid()}.txt").write_text(repr(exc), encoding="utf-8")
