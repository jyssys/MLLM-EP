"""Read-only DeepSeek-V2 fused-MoE route and CUDA timing hook.

The hook wraps existing vLLM calls only.  It records local expert assignment
histograms and a CUDA-event span around the already selected expert kernel.
No route, placement, capacity, or scheduler decision is changed.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_CTX = threading.local()
_PENDING: list[dict[str, Any]] = []
_FLUSHED = False


def _layer(prefix: str) -> int:
    m = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(m.group(1)) if m else -1


def _ev() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _control() -> dict[str, Any]:
    p = os.environ.get("FLASHVEP_RL_CONTROL")
    # Some vLLM worker launch paths preserve the raw-output variable but not
    # experiment-only environment variables.  Derive the same absolute path
    # from the raw directory so request metadata remains observable without
    # changing model execution.
    if not p and os.environ.get("FLASHVEP_RL_RAW_DIR"):
        p = str(Path(os.environ["FLASHVEP_RL_RAW_DIR"]).parent / "control.json")
    if not p:
        return {}
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _flush() -> None:
    global _FLUSHED
    if _FLUSHED:
        return
    _FLUSHED = True
    if not _PENDING:
        return
    torch.cuda.synchronize()
    from vllm.distributed import get_ep_group

    out = Path(os.environ["FLASHVEP_RL_RAW_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    rank = int(get_ep_group().rank_in_group)
    path = out / f"rank{rank}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in _PENDING:
            item = {k: v for k, v in row.items()
                    if k not in ("expert", "dispatch_ms_event", "combine_start", "combine_end")}
            ds, de = row.get("dispatch_ms_event", (None, None))
            item["dispatch_ms"] = float(ds.elapsed_time(de)) if ds is not None else None
            if row.get("combine_start") is not None:
                item["combine_ms"] = float(row["combine_start"].elapsed_time(row["combine_end"]))
            else:
                item["combine_ms"] = None
            item["expert_ms"] = float(row["expert_start"].elapsed_time(row["expert_end"]))
            item.pop("expert_start", None); item.pop("expert_end", None)
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    proof = {
        "status": "ok", "ep_rank": rank, "rows": len(_PENDING),
        "timing": "CUDA event around existing FusedMoEKernelModularImpl._fused_experts",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "route_or_placement_mutation": False,
        "shared_expert_separate": True,
    }
    (out / f"rank{rank}.proof.json").write_text(json.dumps(proof, indent=2) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

    orig_init = DeepseekV2DecoderLayer.__init__
    orig_forward = DeepseekV2DecoderLayer.forward
    orig_prepare = FusedMoEKernelModularImpl._prepare
    orig_experts = FusedMoEKernelModularImpl._fused_experts
    orig_finalize = FusedMoEKernelModularImpl._finalize

    def init(self: Any, *a: Any, **kw: Any) -> None:
        orig_init(self, *a, **kw)
        prefix = str(kw.get("prefix", a[1] if len(a) > 1 else ""))
        self._flashvep_rl_layer = _layer(prefix)

    def forward(self: Any, *a: Any, **kw: Any) -> Any:
        prior = getattr(_CTX, "layer", -1)
        layer = int(getattr(self, "_flashvep_rl_layer", -1))
        if layer == 0:
            entry = _control()
            if entry.get("flush"):
                _flush()
            _CTX.entry = entry
            _CTX.forward_index = int(getattr(_CTX, "forward_index", 0)) + 1
        _CTX.layer = layer
        try:
            return orig_forward(self, *a, **kw)
        finally:
            _CTX.layer = prior

    def prepare(self: Any, *a: Any, **kw: Any) -> Any:
        entry = dict(getattr(_CTX, "entry", {}))
        layer = int(getattr(_CTX, "layer", -1))
        # Dummy/profile and first-use tuning forwards run before a request
        # control record exists.  They must not be mistaken for natural
        # workload invocations.
        if not entry.get("instrument", False) or layer < 0:
            return orig_prepare(self, *a, **kw)
        st, en = _ev(), _ev(); stream = torch.cuda.current_stream(); st.record(stream)
        value = orig_prepare(self, *a, **kw); en.record(stream)
        worker_dp = int(os.environ.get("VLLM_DP_RANK", -1))
        domains = entry.get("domains", [])
        domain = (domains[worker_dp] if isinstance(domains, list)
                  and 0 <= worker_dp < len(domains)
                  else entry.get("domain", "unknown"))
        _CTX.record = {
            "batch_id": entry.get("batch_id", "natural"),
            "condition": entry.get("condition", "natural"),
            "domain": domain,
            "step": int(entry.get("step", -1)),
            "measured": bool(entry.get("measured", True)),
            "scheduler_iteration": int(getattr(_CTX, "forward_index", 0)),
            "worker_dp_rank": worker_dp,
            "ep_rank": int(get_ep_group().rank_in_group), "layer": layer,
            "dispatch_ms_event": (st, en),
        }
        return value

    def experts(self: Any, *a: Any, **kw: Any) -> torch.Tensor:
        rec = getattr(_CTX, "record", None)
        if rec is None:
            return orig_experts(self, *a, **kw)
        names = ("in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                 "topk_ids", "activation", "global_num_experts", "local_num_experts",
                 "expert_map", "apply_router_weight_on_input", "expert_tokens_meta")
        vals = dict(zip(names, a, strict=False)); vals.update(kw)
        meta = vals.get("expert_tokens_meta")
        hist = None
        if meta is not None and getattr(meta, "expert_num_tokens_cpu", None) is not None:
            hist = [int(x) for x in meta.expert_num_tokens_cpu.tolist()]
        topk = vals.get("topk_ids")
        topk_shape = list(topk.shape) if torch.is_tensor(topk) else None
        st, en = _ev(), _ev(); st.record(torch.cuda.current_stream())
        out = orig_experts(self, *a, **kw); en.record(torch.cuda.current_stream())
        rec.update({
            "expert_histogram": hist or [],
            "total_assignments": int(sum(hist or [])),
            "dispatched_rows": int(vals["a1q"].shape[0]),
            "topk_shape": topk_shape,
            "expert_start": st, "expert_end": en,
            "expert_backend": type(self.fused_experts).__name__,
            "prepare_finalize_backend": type(self.prepare_finalize).__name__,
        })
        return out

    def finalize(self: Any, *a: Any, **kw: Any) -> Any:
        rec = getattr(_CTX, "record", None)
        if rec is None:
            return orig_finalize(self, *a, **kw)
        cs, ce = _ev(), _ev(); cs.record(torch.cuda.current_stream())
        value = orig_finalize(self, *a, **kw)
        ce.record(torch.cuda.current_stream())
        rec["combine_start"], rec["combine_end"] = cs, ce
        _PENDING.append(rec)
        _CTX.record = None
        return value

    DeepseekV2DecoderLayer.__init__ = init
    DeepseekV2DecoderLayer.forward = forward
    FusedMoEKernelModularImpl._prepare = prepare
    FusedMoEKernelModularImpl._fused_experts = experts
    FusedMoEKernelModularImpl._finalize = finalize


__all__ = ["install"]
