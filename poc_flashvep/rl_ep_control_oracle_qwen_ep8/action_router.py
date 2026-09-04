"""Bounded Capacity-Aware action hook for the Qwen3-Moe router.

This is an experiment-only wrapper around vLLM's existing ``BaseRouter``.
TEMP_BALANCE follows Capacity-Aware-MoE's score-ranked capacity selection:
the router's full logits are over-selected, per-expert overflow is retained
for the highest scoring tokens, and the remaining top-k slots are selected
from the surviving candidates.  KEEP never changes router outputs.  No
expert weights, placement, scheduler, or production vLLM files are changed.

The hook records before/after route counts and changed assignment fraction so
quality and route side effects are explicit.  It is intentionally not called
an implementation of the official package: it reuses the same deterministic
capacity-selection semantics in a local bounded wrapper.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_CTX = threading.local()
_COUNT = 0


def _layer_of(module: Any) -> int:
    import re
    text = str(getattr(module, "layer_name", ""))
    m = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", text)
    return int(m.group(1)) if m else -1


def _control() -> dict[str, Any]:
    p = os.environ.get("FLASHVEP_MATRIX_CONTROL", "")
    try:
        return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}
    except Exception:
        return {}


def _capacity_select(
    scores: torch.Tensor,
    top_k: int,
    factor: float,
    overselect: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Select top-k with score-ranked per-expert capacity clipping.

    ``scores`` is the router's full (token, expert) logit tensor.  Invalid
    slots use the existing vLLM sentinel ``num_experts`` and zero weight; the
    production DeepEP path already supports ignored invalid experts.
    """
    if scores.ndim != 2 or scores.shape[0] <= 1:
        return torch.empty((0, top_k), device=scores.device), torch.empty((0, top_k), device=scores.device), {"capacity": None}
    # Capacity-Aware-MoE computes capacity from top-k*T/E and supports an
    # overselect pass to expose local alternatives before clipping.
    t, experts = scores.shape
    cap = max(1, int(torch.ceil(torch.tensor(factor * top_k * t / experts, device=scores.device)).item()))
    probs = torch.softmax(scores.float(), dim=-1)
    cand_k = min(experts, max(top_k, int(top_k * overselect)))
    cand_w, cand_ids = torch.topk(probs, k=cand_k, dim=-1, sorted=False)
    mask = torch.zeros((t, experts), dtype=torch.bool, device=scores.device)
    mask.scatter_(1, cand_ids, True)
    usage_before = mask.sum(dim=0)
    overflow = torch.nonzero(usage_before > cap, as_tuple=False).flatten()
    # The loop is over experts, not tokens; this mirrors the reference's
    # score-ranked overflow selection and is bounded to 128 experts.
    for expert in overflow.tolist():
        rows = torch.nonzero(mask[:, expert], as_tuple=False).flatten()
        keep_n = min(cap, int(rows.numel()))
        if keep_n < int(rows.numel()):
            keep_local = torch.topk(probs[rows, expert], k=keep_n, sorted=False).indices
            keep_rows = rows.index_select(0, keep_local)
            mask[rows, expert] = False
            mask[keep_rows, expert] = True
    masked = probs.masked_fill(~mask, float("-inf"))
    weights, ids = torch.topk(masked, k=top_k, dim=-1, sorted=False)
    valid = torch.isfinite(weights)
    safe_ids = ids.masked_fill(~valid, experts)
    safe_weights = weights.masked_fill(~valid, 0.0)
    safe_weights = safe_weights / safe_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    stats = {
        "capacity": cap,
        "candidate_k": cand_k,
        "overflow_experts": int(overflow.numel()),
        "invalid_slots": int((~valid).sum().item()),
        "assignments_before": int(t * top_k),
        "assignments_after": int(valid.sum().item()),
    }
    return safe_weights.to(scores.dtype), safe_ids.to(torch.int32), stats


def _record(
    layer: int,
    action: str,
    before: torch.Tensor,
    after: torch.Tensor,
    stats: dict[str, Any],
) -> None:
    global _COUNT
    _COUNT += 1
    out = os.environ.get("FLASHVEP_ACTION_RAW_DIR")
    if not out:
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"router_pid{os.getpid()}.jsonl"
    b = before.detach().to("cpu", dtype=torch.int32)
    a = after.detach().to("cpu", dtype=torch.int32)
    row = {
        "pid": os.getpid(),
        "layer": int(layer),
        "action": action,
        "rows": int(before.shape[0]),
        "top_k": int(before.shape[1]),
        "changed_assignments": int((b != a).sum().item()),
        "changed_fraction": float((b != a).float().mean().item()),
        "before_hist": torch.bincount(b.reshape(-1), minlength=128).tolist(),
        "after_hist": torch.bincount(a.clamp_max(127).reshape(-1), minlength=128).tolist(),
        **stats,
    }
    with path.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl

    original_select = BaseRouter.select_experts
    original_forward = Qwen3MoeDecoderLayer.forward
    original_fused = FusedMoEKernelModularImpl._fused_experts
    migration_done = {"value": False}

    def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        prior = getattr(_CTX, "layer", -1)
        _CTX.layer = _layer_of(self)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            _CTX.layer = prior

    def patched_select(self: Any, hidden_states: torch.Tensor,
                       router_logits: torch.Tensor, *,
                       input_ids: torch.Tensor | None = None):
        weights, ids = original_select(self, hidden_states, router_logits, input_ids=input_ids)
        action = os.environ.get("FLASHVEP_ACTION", "KEEP").upper()
        # A2 is deliberately stronger but still uses the same bounded action
        # mechanism. PERSIST_BALANCE is placement-only and never mutates this
        # route path.
        if action not in {"TEMP_BALANCE", "CAPACITY_MILD", "CAPACITY_STRONG"}:
            return weights, ids

        arm = os.environ.get("FLASHVEP_ACTION_ARM_FILE", "")
        # Do not alter vLLM's model-profile/dummy forwards.  The four driver
        # processes arm the hook only after all model workers finish init.
        if not arm or not Path(arm).exists():
            return weights, ids
        if router_logits.ndim != 2 or router_logits.shape[0] <= 1:
            return weights, ids
        factor = 1.25 if action in {"TEMP_BALANCE", "CAPACITY_MILD"} else 1.50
        try:
            new_w, new_ids, stats = _capacity_select(router_logits, int(self.top_k), factor)
            _record(int(getattr(_CTX, "layer", -1)), action, ids, new_ids, stats)
            # DeepEP's layout path in this vLLM build requires the router's
            # original index dtype (usually int64); the local selector uses
            # int32 only internally.
            return new_w.to(weights.dtype), new_ids.to(ids.dtype)
        except Exception as exc:
            # Never change a production route because an experiment hook is
            # incompatible with a particular runner; leave a diagnostic.
            out = os.environ.get("FLASHVEP_ACTION_RAW_DIR")
            if out:
                Path(out).mkdir(parents=True, exist_ok=True)
                with (Path(out) / f"router_pid{os.getpid()}_error.log").open("a") as f:
                    f.write(f"layer={getattr(_CTX, 'layer', -1)} {type(exc).__name__}: {exc}\n")
            return weights, ids

    def patched_fused(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        """One-shot, out-of-band migration timing on actual expert tensors."""
        if (os.environ.get("FLASHVEP_MIGRATION_BENCH") == "1"
                and not migration_done["value"]
                and Path(os.environ.get("FLASHVEP_MIGRATION_ARM_FILE", "__missing__")).exists()):
            migration_done["value"] = True
            try:
                names = ("in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights",
                         "topk_ids", "activation", "global_num_experts", "local_num_experts",
                         "expert_map", "apply_router_weight_on_input", "expert_tokens_meta")
                vals = dict(zip(names, args, strict=False)); vals.update(kwargs)
                w1, w2 = vals["w1"], vals["w2"]
                from vllm.distributed import get_ep_group
                import torch.distributed as dist
                group = get_ep_group().device_group
                # One physical expert's complete BF16 weights: this is the
                # exact tensor shape/dtype held by the Qwen worker. Broadcast
                # models migration of one expert replica to all EP ranks.
                b1 = w1[0].contiguous().clone()
                b2 = w2[0].contiguous().clone()
                dist.barrier(group=group)
                st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
                st.record(torch.cuda.current_stream())
                dist.broadcast(b1, src=0, group=group)
                dist.broadcast(b2, src=0, group=group)
                en.record(torch.cuda.current_stream()); torch.cuda.synchronize()
                ms = float(st.elapsed_time(en))
                out = os.environ.get("FLASHVEP_MIGRATION_RAW_DIR", "")
                if out:
                    Path(out).mkdir(parents=True, exist_ok=True)
                    (Path(out) / f"migration_rank{get_ep_group().rank_in_group}.json").write_text(
                        json.dumps({"status": "ok", "ep_rank": int(get_ep_group().rank_in_group),
                                    "w1_shape": list(w1[0].shape), "w2_shape": list(w2[0].shape),
                                    "dtype": str(w1.dtype), "bytes": int(b1.numel()*b1.element_size()+b2.numel()*b2.element_size()),
                                    "broadcast_two_tensors_ms": ms}, indent=2) + "\n")
            except Exception as exc:
                out = os.environ.get("FLASHVEP_MIGRATION_RAW_DIR", "")
                if out:
                    Path(out).mkdir(parents=True, exist_ok=True)
                    (Path(out) / f"migration_rank{os.getpid()}_error.log").write_text(f"{type(exc).__name__}: {exc}\n")
        return original_fused(self, *args, **kwargs)
    BaseRouter.select_experts = patched_select
    Qwen3MoeDecoderLayer.forward = patched_forward
    FusedMoEKernelModularImpl._fused_experts = patched_fused
