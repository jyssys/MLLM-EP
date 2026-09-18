"""Reference-compatible dense emulation for block caching and FrontierEP.

The model still executes dense attention/shared-expert work.  Cached routed-MoE
and post-layer values are substituted only to test semantics; wall time is not
a sparse-runtime claim.  Structural counters describe the intended fresh EP
worklist.  Optional observers audit completed-block invariance and lossless
temporal coding without changing the returned model output.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch


BLOCK_LENGTH = 32
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 2048
ROUTED_EPS = (4, 8)
SELECTED_LAYERS = (1, 5, 10, 14, 19)


def _balanced_sources(count: int, ep: int) -> np.ndarray:
    base, remainder = divmod(count, ep)
    sizes = np.full(ep, base, dtype=np.int64)
    sizes[:remainder] += 1
    return np.repeat(np.arange(ep, dtype=np.int16), sizes)


def _traffic(expert_ids: np.ndarray, source_ranks: np.ndarray, ep: int):
    owners = np.asarray(expert_ids, dtype=np.int16) // (NUM_EXPERTS // ep)
    sources = np.asarray(source_ranks, dtype=np.int16)
    assignment = np.zeros((ep, ep), dtype=np.int32)
    unique = np.zeros((ep, ep), dtype=np.int32)
    if len(owners):
        np.add.at(assignment, (np.repeat(sources, TOP_K), owners.reshape(-1)), 1)
        for destination in range(ep):
            selected = np.any(owners == destination, axis=1)
            if np.any(selected):
                unique[:, destination] = np.bincount(
                    sources[selected], minlength=ep
                ).astype(np.int32)
        ordered = np.sort(owners, axis=1)
        fanout = 1 + np.count_nonzero(ordered[:, 1:] != ordered[:, :-1], axis=1)
        return assignment, unique, float(fanout.mean()), int(fanout.max())
    return assignment, unique, 0.0, 0


def _vector_similarity(current: torch.Tensor, reference: torch.Tensor):
    current32, reference32 = current.float(), reference.float()
    current_flat = current32.reshape(current32.shape[0], -1)
    reference_flat = reference32.reshape(reference32.shape[0], -1)
    cosine = torch.nn.functional.cosine_similarity(current_flat, reference_flat, dim=-1)
    relative_l2 = ((current_flat - reference_flat).norm(dim=-1)
                   / reference_flat.norm(dim=-1).clamp_min(1e-12))
    exact = (current.reshape(current.shape[0], -1)
             == reference.reshape(reference.shape[0], -1)).float().mean(dim=-1)
    return cosine, relative_l2, exact


def encode_word_bitmap(current: np.ndarray, predictor: np.ndarray):
    """Exact BF16 XOR bitmap codec used only for correctness and byte accounting."""

    current = np.asarray(current, dtype=np.uint16).reshape(-1)
    predictor = np.asarray(predictor, dtype=np.uint16).reshape(-1)
    if current.shape != predictor.shape:
        raise ValueError("current/predictor shape mismatch")
    xor = np.bitwise_xor(current, predictor)
    changed = xor != 0
    bitmap = np.packbits(changed.astype(np.uint8), bitorder="little")
    literals = xor[changed]
    return bitmap, literals, current.size


def decode_word_bitmap(
    predictor: np.ndarray, bitmap: np.ndarray, literals: np.ndarray, words: int
) -> np.ndarray:
    predictor = np.asarray(predictor, dtype=np.uint16).reshape(-1)
    changed = np.unpackbits(
        np.asarray(bitmap, dtype=np.uint8), bitorder="little"
    )[:words].astype(bool)
    xor = np.zeros(words, dtype=np.uint16)
    xor[changed] = np.asarray(literals, dtype=np.uint16)
    return np.bitwise_xor(predictor[:words], xor)


def _codec_metrics(current: torch.Tensor, predictor: torch.Tensor) -> dict[str, np.ndarray]:
    """Vectorized exact-codec statistics for BF16 vectors."""

    current_float = current.float().reshape(current.shape[0], -1)
    predictor_float = predictor.float().reshape(predictor.shape[0], -1)
    delta_float = (current_float - predictor_float).abs()
    cosine = torch.nn.functional.cosine_similarity(
        current_float, predictor_float, dim=-1
    ).cpu().numpy().astype(np.float32)
    relative_l2 = (
        (current_float - predictor_float).norm(dim=-1)
        / predictor_float.norm(dim=-1).clamp_min(1e-12)
    ).cpu().numpy().astype(np.float32)
    delta_quantiles = torch.quantile(
        delta_float, torch.tensor([0.5, 0.9, 0.99], device=delta_float.device), dim=-1
    ).cpu().numpy().astype(np.float32)
    current_words = current.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    predictor_words = predictor.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    current_words = current_words.reshape(current.shape[0], -1)
    predictor_words = predictor_words.reshape(predictor.shape[0], -1)
    xor = np.bitwise_xor(current_words, predictor_words)
    unchanged = xor == 0
    xor_bytes = xor.view(np.uint8).reshape(len(xor), -1)
    zero_bytes = xor_bytes == 0
    words = xor.shape[1]
    byte_count = xor_bytes.shape[1]
    word_bitmap = math.ceil(words / 8) + 2 * np.count_nonzero(~unchanged, axis=1)
    byte_bitmap = math.ceil(byte_count / 8) + np.count_nonzero(~zero_bytes, axis=1)
    entropy_bits = np.empty(len(xor), dtype=np.float32)
    for index, vector in enumerate(xor_bytes):
        count = np.bincount(vector, minlength=256).astype(np.float64)
        probability = count[count > 0] / len(vector)
        entropy_bits[index] = float(-(probability * np.log2(probability)).sum())
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "delta_abs_p50": delta_quantiles[0],
        "delta_abs_p90": delta_quantiles[1],
        "delta_abs_p99": delta_quantiles[2],
        "exact_word_fraction": unchanged.mean(axis=1).astype(np.float32),
        "xor_zero_word_fraction": unchanged.mean(axis=1).astype(np.float32),
        "xor_zero_byte_fraction": zero_bytes.mean(axis=1).astype(np.float32),
        "xor_byte_entropy": entropy_bits,
        "word_bitmap_bytes": word_bitmap.astype(np.int32),
        "byte_bitmap_bytes": byte_bitmap.astype(np.int32),
        "raw_bytes": np.full(len(xor), byte_count, dtype=np.int32),
    }


@dataclass(frozen=True)
class FrontierConfig:
    name: str
    finalization_updates: int | None
    audit_exactness: bool = False
    capture_delta: bool = False

    def __post_init__(self):
        if self.finalization_updates not in (None, 1, 2):
            raise ValueError("finalization_updates must be None/1/2")
        if self.audit_exactness and self.finalization_updates is not None:
            raise ValueError("exactness audit must retain the vanilla graph")


@dataclass
class FrontierGenerationResult:
    generated: torch.Tensor
    records: list[dict[str, Any]]
    exactness: list[dict[str, Any]]
    delta: list[dict[str, Any]]


class FrontierController:
    def __init__(self, model, config: FrontierConfig, request_id: int):
        self.model = model
        self.config = config
        self.request_id = int(request_id)
        self.routed_layers = [
            index for index, layer in enumerate(model.model.layers)
            if hasattr(layer.mlp, "gate")
        ]
        self.route_slot = {layer: slot for slot, layer in enumerate(self.routed_layers)}
        self.records: list[dict[str, Any]] = []
        self.exactness: list[dict[str, Any]] = []
        self.delta: list[dict[str, Any]] = []
        self._handles = []
        self._original_moe = {}
        self._context: dict[str, Any] | None = None
        self._post_cache: dict[int, torch.Tensor] = {}
        self._moe_cache: dict[int, torch.Tensor] = {}
        self._audit_reference: dict[tuple[str, int], torch.Tensor] = {}
        self._delta_state: dict[int, list[dict[str, Any]]] = {}
        self._route_age: dict[int, np.ndarray] = {}
        self._valid = np.zeros(0, dtype=bool)
        self._remaining_updates = np.zeros(0, dtype=np.int16)
        self._block_id = -1
        self._install()

    @staticmethod
    def _grow_vector(vector: np.ndarray, size: int, fill=0):
        if len(vector) >= size:
            return vector
        result = np.full(size, fill, dtype=vector.dtype)
        result[:len(vector)] = vector
        return result

    @staticmethod
    def _grow_tensor(cache: torch.Tensor | None, source: torch.Tensor):
        rows = source.shape[0]
        if cache is None:
            return torch.zeros_like(source)
        if cache.shape[0] >= rows:
            return cache
        return torch.cat((cache, torch.zeros_like(source[cache.shape[0]:])), dim=0)

    def begin_iteration(
        self, *, block_id: int, iteration_id: int, nfe: int,
        physical_rows: int, masked: np.ndarray,
    ):
        if self._context is not None:
            raise RuntimeError("nested FrontierEP context")
        block_changed = int(block_id) != self._block_id
        if block_changed:
            self._delta_state.clear()
            self._route_age.clear()
        self._valid = self._grow_vector(self._valid, physical_rows, False)
        self._remaining_updates = self._grow_vector(
            self._remaining_updates, physical_rows, 0
        )
        current_start = physical_rows - BLOCK_LENGTH
        current_positions = np.arange(current_start, physical_rows)
        fresh = ~self._valid[:physical_rows]
        if self.config.audit_exactness or block_changed:
            # Extending the SDPA sequence changes BF16 kernel tiling enough to
            # perturb deep prefix states despite mathematical block causality.
            # Refresh the full prefix once at each new block length; cache only
            # repeated refinements at the same physical shape.
            fresh[:] = True
        elif self.config.finalization_updates is None:
            fresh[current_start:physical_rows] = True
            fresh |= self._remaining_updates[:physical_rows] > 0
        else:
            fresh[current_positions[masked]] = True
            fresh |= self._remaining_updates[:physical_rows] > 0
        current_remaining = self._remaining_updates[current_start:physical_rows]
        current_committed = ~masked
        sealed = current_committed & (current_remaining == 0) & self._valid[current_start:]
        finalizing = current_committed & (current_remaining > 0)
        layer_count = len(self.routed_layers)
        context: dict[str, Any] = {
            "block_id": int(block_id), "iteration_id": int(iteration_id),
            "nfe": int(nfe), "physical_rows": int(physical_rows),
            "current_start": int(current_start), "masked": masked.copy(),
            "fresh": fresh.copy(), "sealed": sealed.copy(),
            "finalizing": finalizing.copy(),
            "routes": np.full((layer_count, BLOCK_LENGTH, TOP_K), -1, dtype=np.int16),
            "route_weights": np.zeros((layer_count, BLOCK_LENGTH, TOP_K), dtype=np.float16),
            "hist": np.zeros((layer_count, NUM_EXPERTS), dtype=np.uint16),
        }
        for ep in ROUTED_EPS:
            context[f"u_ep{ep}"] = np.zeros((layer_count, ep, ep), dtype=np.int32)
            context[f"a_ep{ep}"] = np.zeros((layer_count, ep, ep), dtype=np.int32)
            context[f"fanout_mean_ep{ep}"] = np.zeros(layer_count, dtype=np.float32)
            context[f"fanout_max_ep{ep}"] = np.zeros(layer_count, dtype=np.int8)
        self._context = context
        self._block_id = int(block_id)

    def _audit_tensor(self, boundary: str, layer_id: int, tensor: torch.Tensor):
        context = self._context
        if context is None or not self.config.audit_exactness:
            return
        current_start = int(context["current_start"])
        if current_start <= 0:
            return
        value = tensor.detach()[0] if tensor.ndim >= 3 and tensor.shape[0] == 1 else tensor.detach()
        value = value[:current_start]
        key = (boundary, int(layer_id))
        reference = self._audit_reference.get(key)
        if reference is None:
            self._audit_reference[key] = value.clone()
            return
        compared = min(len(reference), len(value))
        if compared:
            cosine, relative_l2, exact = _vector_similarity(
                value[:compared], reference[:compared]
            )
            self.exactness.append({
                "request_id": self.request_id,
                "block_id": int(context["block_id"]),
                "iteration_id": int(context["iteration_id"]),
                "layer_id": int(layer_id), "boundary": boundary,
                "rows": int(compared),
                "cosine_min": float(cosine.min().item()),
                "cosine_mean": float(cosine.mean().item()),
                "relative_l2_max": float(relative_l2.max().item()),
                "relative_l2_mean": float(relative_l2.mean().item()),
                "exact_element_fraction": float(exact.mean().item()),
                "bit_exact_rows": int((exact == 1).sum().item()),
            })
        # Keep the immediately preceding refinement as the audit reference.
        # This makes iteration>0 a true same-shape comparison while the first
        # iteration of a new block captures the sequence-extension boundary.
        self._audit_reference[key] = value.clone()

    def _decoder_pre_hook(self, layer_id: int):
        def hook(_module, inputs):
            self._audit_tensor("layer_input", layer_id, inputs[0])
        return hook

    def _qkv_hook(self, layer_id: int, attention):
        q_width = attention.num_heads * attention.head_dim
        kv_width = attention.num_key_value_heads * attention.head_dim

        def hook(_module, _inputs, output):
            self._audit_tensor("key_projection", layer_id, output[..., q_width:q_width + kv_width])
            self._audit_tensor("value_projection", layer_id, output[..., q_width + kv_width:])
        return hook

    def _router_hook(self, layer_id: int):
        slot = self.route_slot[layer_id]

        def hook(_module, _inputs, output):
            context = self._context
            if context is None:
                raise RuntimeError("router outside refinement context")
            ids = output[0].detach().reshape(-1, TOP_K)
            weights = output[1].detach().reshape(-1, TOP_K)
            rows = int(context["physical_rows"])
            start = int(context["current_start"])
            context["routes"][slot] = ids[start:].cpu().numpy().astype(np.int16)
            context["route_weights"][slot] = weights[start:].cpu().numpy().astype(np.float16)
            selected = context["fresh"]
            selected_ids = ids.cpu().numpy()[selected]
            context["hist"][slot] = np.bincount(
                selected_ids.reshape(-1), minlength=NUM_EXPERTS
            ).astype(np.uint16)
            for ep in ROUTED_EPS:
                sources = _balanced_sources(rows, ep)[selected]
                a, u, mean_fanout, max_fanout = _traffic(selected_ids, sources, ep)
                context[f"a_ep{ep}"][slot] = a
                context[f"u_ep{ep}"][slot] = u
                context[f"fanout_mean_ep{ep}"][slot] = mean_fanout
                context[f"fanout_max_ep{ep}"][slot] = max_fanout
            self._audit_router(layer_id, ids, weights)
        return hook

    def _audit_router(self, layer_id: int, ids: torch.Tensor, weights: torch.Tensor):
        context = self._context
        if context is None or not self.config.audit_exactness:
            return
        current_start = int(context["current_start"])
        if current_start <= 0:
            return
        for name, value in (("router_ids", ids), ("router_weights", weights)):
            value = value[:current_start].detach()
            key = (name, int(layer_id))
            reference = self._audit_reference.get(key)
            if reference is None:
                self._audit_reference[key] = value.clone()
                continue
            compared = min(len(reference), len(value))
            exact = value[:compared] == reference[:compared]
            if name == "router_ids":
                self.exactness.append({
                    "request_id": self.request_id, "block_id": int(context["block_id"]),
                    "iteration_id": int(context["iteration_id"]), "layer_id": int(layer_id),
                    "boundary": "router_topk", "rows": int(compared),
                    "cosine_min": 1.0, "cosine_mean": 1.0,
                    "relative_l2_max": 0.0, "relative_l2_mean": 0.0,
                    "exact_element_fraction": float(exact.float().mean().item()),
                    "bit_exact_rows": int(exact.all(dim=-1).sum().item()),
                })
            else:
                cosine, relative_l2, exact_fraction = _vector_similarity(
                    value[:compared], reference[:compared]
                )
                self.exactness.append({
                    "request_id": self.request_id, "block_id": int(context["block_id"]),
                    "iteration_id": int(context["iteration_id"]), "layer_id": int(layer_id),
                    "boundary": name, "rows": int(compared),
                    "cosine_min": float(cosine.min().item()),
                    "cosine_mean": float(cosine.mean().item()),
                    "relative_l2_max": float(relative_l2.max().item()),
                    "relative_l2_mean": float(relative_l2.mean().item()),
                    "exact_element_fraction": float(exact_fraction.mean().item()),
                    "bit_exact_rows": int((exact_fraction == 1).sum().item()),
                })
            self._audit_reference[key] = value.clone()

    def _wrap_moe(self, layer_id: int, module, original):
        controller = self

        def wrapped(_module_self, x, topk_ids, topk_weight):
            fresh_output = original(x, topk_ids, topk_weight)
            context = controller._context
            if context is None:
                return fresh_output
            controller._audit_tensor(
                "routed_expert_output", layer_id, fresh_output.unsqueeze(0)
            )
            if controller.config.capture_delta and layer_id in SELECTED_LAYERS:
                controller._capture_delta(
                    layer_id, x, topk_ids, topk_weight, fresh_output
                )
            if controller.config.audit_exactness:
                return fresh_output
            cache = controller._grow_tensor(
                controller._moe_cache.get(layer_id), fresh_output
            )
            selected = torch.as_tensor(context["fresh"], device=x.device)
            output = fresh_output.clone()
            output[~selected] = cache[~selected]
            cache[selected] = fresh_output[selected].detach()
            controller._moe_cache[layer_id] = cache
            return output

        return MethodType(wrapped, module)

    def _capture_delta(
        self, layer_id: int, x: torch.Tensor, ids: torch.Tensor,
        weights: torch.Tensor, output: torch.Tensor,
    ):
        context = self._context
        assert context is not None
        start = int(context["current_start"])
        current = {
            "input": x[start:].detach().clone(),
            "output": output[start:].detach().clone(),
            "ids": ids[start:].detach().clone(),
            "weights": weights[start:].detach().clone(),
            "mask": context["masked"].copy(),
        }
        history = self._delta_state.setdefault(layer_id, [])
        previous = history[-1] if history else None
        if previous is not None:
            current_ids = current["ids"].cpu().numpy().astype(np.int16)
            previous_ids = previous["ids"].cpu().numpy().astype(np.int16)
            mask = current["mask"] & previous["mask"]
            stay = (current_ids[:, :, None] == previous_ids[:, None, :]).any(axis=2)
            eligible = np.flatnonzero(mask & np.any(stay, axis=1))
            if len(eligible):
                age = self._route_age.get(layer_id)
                if age is None:
                    age = np.zeros((BLOCK_LENGTH, TOP_K), dtype=np.int16)
                next_age = np.ones_like(age)
                for position in range(BLOCK_LENGTH):
                    lookup = {int(expert): slot for slot, expert in enumerate(previous_ids[position])}
                    for slot, expert in enumerate(current_ids[position]):
                        prior_slot = lookup.get(int(expert))
                        if prior_slot is not None:
                            next_age[position, slot] = age[position, prior_slot] + 1
                self._route_age[layer_id] = next_age
                index = torch.as_tensor(eligible, device=x.device)
                self._append_delta_records(
                    layer_id, eligible, "adjacent", current, previous, index,
                    current_ids, previous_ids, next_age,
                )
                wrong_positions = np.roll(eligible, 1)
                wrong_index = torch.as_tensor(wrong_positions, device=x.device)
                self._append_delta_records(
                    layer_id, eligible, "wrong_token", current, previous, index,
                    current_ids, previous_ids, next_age, predictor_index=wrong_index,
                )
                self._append_delta_records(
                    layer_id, eligible, "zero_predictor", current, previous, index,
                    current_ids, previous_ids, next_age, zero_predictor=True,
                )
                if len(history) >= 2:
                    older = history[-2]
                    older_ids = older["ids"].cpu().numpy().astype(np.int16)
                    older_mask = older["mask"]
                    older_stay = (
                        current_ids[:, :, None] == older_ids[:, None, :]
                    ).any(axis=2)
                    nonadjacent = np.flatnonzero(
                        current["mask"] & older_mask & np.any(older_stay, axis=1)
                    )
                    if len(nonadjacent):
                        older_index = torch.as_tensor(nonadjacent, device=x.device)
                        self._append_delta_records(
                            layer_id, nonadjacent, "nonadjacent", current, older,
                            older_index, current_ids, older_ids, next_age,
                        )
        history.append(current)
        if len(history) > 2:
            del history[0]

    def _append_delta_records(
        self, layer_id: int, positions: np.ndarray, control: str,
        current: dict[str, Any], predictor: dict[str, Any],
        current_index: torch.Tensor, current_ids: np.ndarray,
        predictor_ids: np.ndarray, route_age: np.ndarray,
        predictor_index: torch.Tensor | None = None, zero_predictor: bool = False,
    ):
        context = self._context
        assert context is not None
        pred_index = current_index if predictor_index is None else predictor_index
        for boundary in ("dispatch", "combine"):
            current_value = current["input" if boundary == "dispatch" else "output"][current_index]
            if zero_predictor:
                predictor_value = torch.zeros_like(current_value)
            else:
                predictor_value = predictor["input" if boundary == "dispatch" else "output"][pred_index]
            metrics = _codec_metrics(current_value, predictor_value)
            for ordinal, position in enumerate(positions):
                stay_count = len(
                    set(map(int, current_ids[position]))
                    & set(map(int, predictor_ids[int(pred_index[ordinal].item())]))
                )
                ages = route_age[position]
                self.delta.append({
                    "request_id": self.request_id,
                    "block_id": int(context["block_id"]),
                    "iteration_id": int(context["iteration_id"]),
                    "layer_id": int(layer_id), "token_position": int(position),
                    "control": control, "boundary": boundary,
                    "mask_ratio": float(context["masked"].mean()),
                    "stay_edges": int(stay_count),
                    "max_route_age": int(ages.max(initial=0)),
                    "current_ids": current_ids[position].copy(),
                    "predictor_ids": predictor_ids[int(pred_index[ordinal].item())].copy(),
                    **{key: value[ordinal].item() for key, value in metrics.items()},
                })

    def _decoder_hook(self, layer_id: int):
        controller = self

        def hook(_module, _inputs, output):
            context = controller._context
            if context is None:
                return output
            hidden = output[0]
            controller._audit_tensor("post_layer", layer_id, hidden)
            if controller.config.audit_exactness:
                return output
            flat = hidden[0]
            cache = controller._grow_tensor(controller._post_cache.get(layer_id), flat)
            selected = torch.as_tensor(context["fresh"], device=flat.device)
            modified = flat.clone()
            modified[~selected] = cache[~selected]
            cache[selected] = flat[selected].detach()
            controller._post_cache[layer_id] = cache
            return (modified.unsqueeze(0),) + tuple(output[1:])

        return hook

    def _install(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            if layer_id in SELECTED_LAYERS:
                self._handles.append(layer.register_forward_pre_hook(
                    self._decoder_pre_hook(layer_id)
                ))
                self._handles.append(layer.attention.query_key_value.register_forward_hook(
                    self._qkv_hook(layer_id, layer.attention)
                ))
            if hasattr(layer.mlp, "gate"):
                self._handles.append(layer.mlp.gate.register_forward_hook(
                    self._router_hook(layer_id)
                ))
                original = layer.mlp.moe_infer
                self._original_moe[layer_id] = original
                layer.mlp.moe_infer = self._wrap_moe(layer_id, layer.mlp, original)
            self._handles.append(layer.register_forward_hook(self._decoder_hook(layer_id)))

    def capture_logits(self, logits: torch.Tensor):
        self._audit_tensor("logits", 20, logits)

    def finalize_iteration(self, transfer: torch.Tensor):
        context = self._context
        if context is None:
            raise RuntimeError("no active FrontierEP context")
        transfer_np = transfer.detach().cpu().numpy()[0].astype(bool)
        fresh = context["fresh"]
        rows = int(context["physical_rows"])
        current_start = int(context["current_start"])
        before = self._remaining_updates[:rows].copy()
        refreshed = fresh & (before > 0)
        self._remaining_updates[:rows][refreshed] -= 1
        if self.config.finalization_updates is not None:
            positions = current_start + np.flatnonzero(transfer_np)
            self._remaining_updates[positions] = self.config.finalization_updates
        self._valid[:rows] = True
        record = {key: value for key, value in context.items()}
        record.update(
            transfer=transfer_np,
            live_mask_count=int(context["masked"].sum()),
            newly_committed_count=int(context["finalizing"].sum()),
            sealed_count=int(context["sealed"].sum()),
            fresh_rows=int(fresh.sum()),
            fresh_current_rows=int(fresh[current_start:].sum()),
            fresh_prior_rows=int(fresh[:current_start].sum()),
        )
        self.records.append(record)
        self._context = None

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer_id, original in self._original_moe.items():
            self.model.model.layers[layer_id].mlp.moe_infer = original
        self._original_moe.clear()
        self._context = None


def _transfer_mask(confidence, masked, threshold, num_to_transfer):
    transfer = torch.zeros_like(masked, dtype=torch.bool)
    restricted = torch.where(masked, confidence, -torch.inf)
    high = restricted[0] > threshold
    if int(high.sum().item()) >= num_to_transfer:
        transfer[0] = high
    else:
        count = min(num_to_transfer, int(masked.sum().item()))
        if count:
            _, index = torch.topk(restricted[0], k=count)
            transfer[0, index] = True
    return transfer


@torch.no_grad()
def generate_frontier(
    model, inputs: torch.Tensor, *, config: FrontierConfig,
    threshold: float = 0.95, block_length: int = 32, steps: int = 32,
    gen_length: int = 2048, temperature: float = 0.0,
    eos_early_stop: bool = True, eos_id: int = 156892,
    mask_id: int = 156895, request_id: int = 0,
) -> FrontierGenerationResult:
    if block_length != BLOCK_LENGTH:
        raise ValueError("FrontierEP PoC is fixed to block_length=32")
    input_ids = inputs.to(model.device)
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    attention_mask = (
        block_mask.repeat_interleave(block_length, 0)
        .repeat_interleave(block_length, 1).unsqueeze(0).unsqueeze(0)
    ).log().to(torch.bfloat16)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids.clone()
    prefill_blocks = prompt_length // block_length
    schedule = model._get_num_transfer_tokens(block_length, steps)
    controller = FrontierController(model, config, request_id)
    nfe = 0
    try:
        for num_block in range(prefill_blocks, num_blocks):
            end = (num_block + 1) * block_length
            cur_x = x[:, :end]
            cur_mask = attention_mask[:, :, :end, :end]
            cur_positions = position_ids[:, :end]
            logical_block = num_block - prefill_blocks
            for step in range(steps):
                masked = cur_x[:, -block_length:] == mask_id
                if int(masked.sum()) == 0:
                    break
                controller.begin_iteration(
                    block_id=logical_block, iteration_id=step, nfe=nfe,
                    physical_rows=end, masked=masked[0].cpu().numpy(),
                )
                logits = model.forward(
                    cur_x, attention_mask=cur_mask, position_ids=cur_positions,
                ).logits
                controller.capture_logits(logits)
                nfe += 1
                active_logits = logits[:, -block_length:, :]
                x0, probability = model._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=None, top_p=None
                )
                confidence = torch.where(masked, probability, -torch.inf)
                transfer = _transfer_mask(
                    confidence, masked, threshold, int(schedule[step].item())
                )
                controller.finalize_iteration(transfer)
                if transfer.any():
                    cur_x[:, -block_length:][transfer] = x0[transfer]
                if eos_early_stop and (x0[transfer] == eos_id).any():
                    eos_positions = (cur_x[0] == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions):
                        eos_position = int(eos_positions[0].item())
                        if (cur_x[0, prompt_length:eos_position] != mask_id).all():
                            final = x[:, :total_length][:, :eos_position + 1]
                            return FrontierGenerationResult(
                                final, controller.records, controller.exactness,
                                controller.delta,
                            )
            x[:, :end] = cur_x
            if eos_id is not None and (x[0, prompt_length:end] == eos_id).any():
                break
        answer = x[:, :prompt_length + gen_length]
        eos_positions = (answer[0][prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
        first = int(eos_positions[0].item()) if len(eos_positions) else gen_length
        return FrontierGenerationResult(
            answer[:, prompt_length:prompt_length + first + 1],
            controller.records, controller.exactness, controller.delta,
        )
    finally:
        controller.close()


def save_frontier_trace(
    path: Path, result: FrontierGenerationResult, metadata: dict[str, Any]
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = result.records
    payload: dict[str, np.ndarray] = {}
    for key in (
        "block_id", "iteration_id", "nfe", "physical_rows", "current_start",
        "live_mask_count", "newly_committed_count", "sealed_count", "fresh_rows",
        "fresh_current_rows", "fresh_prior_rows",
    ):
        payload[key] = np.asarray([record[key] for record in records])
    for key in ("masked", "sealed", "finalizing", "transfer"):
        payload[key] = np.stack([record[key] for record in records])
    payload["routes"] = np.stack([record["routes"] for record in records])
    payload["route_weights"] = np.stack([record["route_weights"] for record in records])
    payload["hist"] = np.stack([record["hist"] for record in records])
    for ep in ROUTED_EPS:
        for prefix in ("a", "u", "fanout_mean", "fanout_max"):
            key = f"{prefix}_ep{ep}"
            payload[key] = np.stack([record[key] for record in records])
    if result.delta:
        integer = (
            "request_id", "block_id", "iteration_id", "layer_id", "token_position",
            "stay_edges", "max_route_age", "word_bitmap_bytes", "byte_bitmap_bytes",
            "raw_bytes",
        )
        floats = (
            "mask_ratio", "exact_word_fraction", "xor_zero_word_fraction",
            "xor_zero_byte_fraction", "xor_byte_entropy", "cosine",
            "relative_l2", "delta_abs_p50", "delta_abs_p90", "delta_abs_p99",
        )
        for key in integer:
            payload[f"delta_{key}"] = np.asarray(
                [record[key] for record in result.delta], dtype=np.int32
            )
        for key in floats:
            payload[f"delta_{key}"] = np.asarray(
                [record[key] for record in result.delta], dtype=np.float32
            )
        for key in ("control", "boundary"):
            payload[f"delta_{key}"] = np.asarray(
                [record[key] for record in result.delta], dtype=np.str_
            )
        payload["delta_current_ids"] = np.stack(
            [record["current_ids"] for record in result.delta]
        ).astype(np.int16)
        payload["delta_predictor_ids"] = np.stack(
            [record["predictor_ids"] for record in result.delta]
        ).astype(np.int16)
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True), dtype=np.str_
    )
    np.savez_compressed(path, **payload)
    if result.exactness:
        path.with_suffix(".exactness.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in result.exactness)
        )
