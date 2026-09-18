"""Dense semantic emulation of token-to-expert route pruning.

Every expert is still evaluated by the reference implementation.  A selected
route's effective gate coefficient is set to zero immediately before the
routed-expert combine.  This exactly emulates omission without renormalizing
the retained routes, but does not claim a measured sparse-runtime speedup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch


NUM_EXPERTS = 256
TOP_K = 8
BLOCK_LENGTH = 32
EP_TARGETS = (4, 8)
PRESSURE_BINS = np.asarray([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf])


@dataclass(frozen=True)
class RoutePruningConfig:
    policy: str
    mass_budget: float
    min_k: int = 4
    target_ep: int = 8

    def __post_init__(self):
        if self.policy not in ("vanilla", "p2", "p4"):
            raise ValueError(f"unsupported route-pruning policy: {self.policy}")
        if not 0 <= self.mass_budget <= 1:
            raise ValueError("mass_budget must lie in [0,1]")
        if not 1 <= self.min_k <= TOP_K:
            raise ValueError("min_k must lie in [1,8]")
        if self.target_ep not in EP_TARGETS:
            raise ValueError("target_ep must be 4 or 8")
        if self.policy == "vanilla" and self.mass_budget != 0:
            raise ValueError("vanilla requires a zero mass budget")


def _effective_ranks(weights: np.ndarray) -> np.ndarray:
    order = np.argsort(-weights, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int8)
    rows = np.arange(len(weights))[:, None]
    ranks[rows, order] = np.arange(1, TOP_K + 1, dtype=np.int8)[None, :]
    return ranks


def _rank_pressure(expert_ids: np.ndarray, ep: int) -> tuple[np.ndarray, np.ndarray]:
    owners = expert_ids // (NUM_EXPERTS // ep)
    load = np.bincount(owners.reshape(-1), minlength=ep).astype(np.float64)
    pressure = np.maximum(load - load.mean(), 0.0)
    normalized = pressure / max(load.mean(), 1e-12)
    return pressure, normalized


def select_pruned_routes(
    expert_ids: np.ndarray,
    effective_weights: np.ndarray,
    config: RoutePruningConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a deterministic route omission mask and auditable statistics."""
    expert_ids = np.asarray(expert_ids, dtype=np.int16)
    weights = np.asarray(effective_weights, dtype=np.float32)
    if expert_ids.shape != weights.shape or expert_ids.ndim != 2 or expert_ids.shape[1] != TOP_K:
        raise ValueError("expert_ids and weights must be matching [tokens,8] arrays")
    normalized = np.divide(
        weights, weights.sum(axis=1, keepdims=True),
        out=np.zeros_like(weights), where=weights.sum(axis=1, keepdims=True) != 0,
    )
    ranks = _effective_ranks(weights)
    pressure, normalized_pressure = _rank_pressure(expert_ids, config.target_ep)
    removed = np.zeros(expert_ids.shape, dtype=bool)
    if config.policy != "vanilla" and config.mass_budget > 0 and len(expert_ids):
        owners = expert_ids // (NUM_EXPERTS // config.target_ep)
        eligible = ranks > config.min_k
        if config.policy == "p2":
            score = np.divide(1.0, normalized, out=np.zeros_like(normalized), where=normalized > 0)
        else:
            route_pressure = pressure[owners]
            eligible &= route_pressure > 0
            score = np.divide(
                route_pressure, normalized,
                out=np.zeros_like(normalized), where=normalized > 0,
            )
        score[~eligible] = -1
        order = np.argsort(-score.reshape(-1), kind="stable")
        budget = config.mass_budget * len(expert_ids)
        spent = 0.0
        flat_mass = normalized.reshape(-1)
        flat_removed = removed.reshape(-1)
        flat_score = score.reshape(-1)
        for index in order:
            if flat_score[index] < 0:
                break
            cost = float(flat_mass[index])
            if spent + cost <= budget + 1e-12:
                flat_removed[index] = True
                spent += cost
    owners = expert_ids // (NUM_EXPERTS // config.target_ep)
    removed_pressure = normalized_pressure[owners[removed]] if np.any(removed) else np.empty(0)
    pressure_hist, _ = np.histogram(removed_pressure, bins=PRESSURE_BINS)
    retained_k = TOP_K - removed.sum(axis=1)
    return removed, {
        "normalized_weights": normalized,
        "ranks": ranks,
        "rank_load": np.bincount(owners.reshape(-1), minlength=config.target_ep),
        "rank_pressure": pressure,
        "removed_mass": float(normalized[removed].sum()),
        "removed_routes": int(removed.sum()),
        "retained_k_hist": np.bincount(retained_k, minlength=TOP_K + 1),
        "removed_pressure_hist": pressure_hist,
    }


def _route_geometry(expert_ids: np.ndarray, kept: np.ndarray, ep: int):
    owners = expert_ids // (NUM_EXPERTS // ep)
    source = np.arange(len(expert_ids), dtype=np.int16) % ep
    unique = np.zeros((ep, ep), dtype=np.uint16)
    fanout_sum = 0
    for destination in range(ep):
        sent = np.any((owners == destination) & kept, axis=1)
        if np.any(sent):
            unique[:, destination] = np.bincount(source[sent], minlength=ep).astype(np.uint16)
    for row in range(len(expert_ids)):
        fanout_sum += len(np.unique(owners[row, kept[row]]))
    return unique, fanout_sum


class RoutePruningEmulator:
    """Install route-weight intervention hooks for one request at a time."""

    def __init__(self, model, config: RoutePruningConfig, request_id: int):
        self.model = model
        self.config = config
        self.request_id = int(request_id)
        self.routed_layers = [
            index for index, layer in enumerate(model.model.layers)
            if hasattr(layer.mlp, "gate")
        ]
        self._original_forward = None
        self._original_moe: dict[int, Any] = {}
        self._context: dict[str, Any] | None = None
        self._last_context: dict[str, Any] | None = None
        self._block_iterations: dict[int, int] = {}
        self.invocations: list[dict[str, Any]] = []
        self.iterations: list[dict[str, Any]] = []

    def _finalize_previous(self, next_ids: torch.Tensor | None):
        previous = self._last_context
        if previous is None:
            return
        if next_ids is None:
            masked_after = 0
            terminated = True
        else:
            next_block = next_ids.shape[1] // BLOCK_LENGTH - 1
            masked_after = (
                int((next_ids[:, -BLOCK_LENGTH:] == previous["mask_id"]).sum())
                if next_block == previous["block_id"] else 0
            )
            terminated = False
        self.iterations.append({
            "block_id": previous["block_id"],
            "iteration_id": previous["iteration_id"],
            "nfe": previous["nfe"],
            "masked_before": previous["masked_before"],
            "masked_after": masked_after,
            "accepted": previous["masked_before"] - masked_after,
            "terminated": terminated,
        })
        self._last_context = None

    def _wrapped_moe(self, layer_id: int, original):
        emulator = self

        def wrapped(_module_self, x, topk_ids, topk_weight):
            context = emulator._context
            if context is None:
                raise RuntimeError("MoE executed outside route-pruning context")
            ids = topk_ids.detach().reshape(-1, TOP_K).cpu().numpy().astype(np.int16)
            weights = topk_weight.detach().reshape(-1, TOP_K).float().cpu().numpy()
            removed, selection = select_pruned_routes(ids, weights, emulator.config)
            kept = ~removed
            remaining_ids = ids[kept]
            histogram = np.bincount(remaining_ids, minlength=NUM_EXPERTS).astype(np.uint32)
            row = {
                "request_id": emulator.request_id,
                "block_id": context["block_id"],
                "iteration_id": context["iteration_id"],
                "nfe": context["nfe"],
                "layer_id": layer_id,
                "physical_rows": len(ids),
                "original_routes": int(ids.size),
                "remaining_routes": int(kept.sum()),
                "removed_mass": selection["removed_mass"],
                "histogram": histogram,
                "k_hist": selection["retained_k_hist"].astype(np.uint32),
                "pressure_hist": selection["removed_pressure_hist"].astype(np.uint32),
                "unique_experts": int(np.count_nonzero(histogram)),
            }
            for ep in EP_TARGETS:
                unique, fanout_sum = _route_geometry(ids, kept, ep)
                row[f"unique_ep{ep}"] = unique
                row[f"fanout_sum_ep{ep}"] = fanout_sum
            emulator.invocations.append(row)
            pruned_weight = topk_weight.masked_fill(
                torch.as_tensor(removed, dtype=torch.bool, device=topk_weight.device), 0
            )
            return original(x, topk_ids, pruned_weight)

        return wrapped

    def install(self, mask_id: int):
        if self._original_forward is not None:
            raise RuntimeError("route-pruning emulator already installed")
        self._block_iterations.clear()
        self._original_forward = self.model.forward
        emulator = self

        def observed_forward(model_self, input_ids, *args, **kwargs):
            emulator._finalize_previous(input_ids.detach())
            block_id = input_ids.shape[1] // BLOCK_LENGTH - 1
            iteration_id = emulator._block_iterations.get(block_id, 0)
            emulator._block_iterations[block_id] = iteration_id + 1
            context = {
                "block_id": int(block_id),
                "iteration_id": int(iteration_id),
                "nfe": int(sum(emulator._block_iterations.values()) - 1),
                "masked_before": int((input_ids[:, -BLOCK_LENGTH:] == mask_id).sum()),
                "mask_id": int(mask_id),
            }
            emulator._context = context
            result = emulator._original_forward(input_ids, *args, **kwargs)
            emulator._context = None
            emulator._last_context = context
            return result

        self.model.forward = MethodType(observed_forward, self.model)
        for layer_id in self.routed_layers:
            module = self.model.model.layers[layer_id].mlp
            original = module.moe_infer
            self._original_moe[layer_id] = original
            module.moe_infer = MethodType(self._wrapped_moe(layer_id, original), module)

    def stop(self):
        if self._original_forward is None:
            return
        self._finalize_previous(None)
        self.model.forward = self._original_forward
        for layer_id, original in self._original_moe.items():
            self.model.model.layers[layer_id].mlp.moe_infer = original
        self._original_moe.clear()
        self._original_forward = None
        self._context = None

    def summary(self) -> dict[str, Any]:
        physical = sum(row["physical_rows"] for row in self.invocations)
        original = sum(row["original_routes"] for row in self.invocations)
        remaining = sum(row["remaining_routes"] for row in self.invocations)
        removed_mass = sum(row["removed_mass"] for row in self.invocations)
        k_hist = np.sum([row["k_hist"] for row in self.invocations], axis=0)
        return {
            "nominal_removed_mass_fraction": self.config.mass_budget,
            "actual_removed_mass_fraction": removed_mass / physical if physical else 0.0,
            "original_routes": original,
            "fresh_routes": remaining,
            "route_reduction_fraction": 1 - remaining / original if original else 0.0,
            "avg_k": remaining / physical if physical else 0.0,
            "k_hist": k_hist.tolist() if len(self.invocations) else [0] * (TOP_K + 1),
            "invocations": len(self.invocations),
        }

    def save(self, path: Path, metadata: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.invocations
        arrays: dict[str, np.ndarray] = {}
        scalar_fields = (
            "request_id", "block_id", "iteration_id", "nfe", "layer_id",
            "physical_rows", "original_routes", "remaining_routes", "removed_mass",
            "unique_experts", "fanout_sum_ep4", "fanout_sum_ep8",
        )
        for name in scalar_fields:
            arrays[name] = np.asarray([row[name] for row in rows])
        for name in ("histogram", "k_hist", "pressure_hist", "unique_ep4", "unique_ep8"):
            arrays[name] = np.stack([row[name] for row in rows])
        for name in ("block_id", "iteration_id", "nfe", "masked_before", "masked_after", "accepted", "terminated"):
            arrays[f"iteration_{name}"] = np.asarray([row[name] for row in self.iterations])
        document = {
            "schema_version": 1,
            "semantics": "dense exact-output emulation by zeroing selected effective gate coefficients",
            "no_weight_renormalization": True,
            "dense_wall_time_not_speedup": True,
            "config": asdict(self.config),
            **metadata,
        }
        arrays["metadata_json"] = np.asarray(json.dumps(document, sort_keys=True))
        np.savez_compressed(path, **arrays)
