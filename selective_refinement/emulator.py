"""Official-loop-compatible dense emulation for selective refinement.

The model still executes every row.  F0/F1 replace already-computed values to
probe semantics; none of the wall time produced here is a sparse-runtime
speedup.  Compact structural counters are collected for offline EP projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from .policy import PolicyConfig, choose_active_set


BLOCK_LENGTH = 32
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 2048
ROUTED_EPS = (1, 4, 8)
SUBSTRATES = ("s0", "s1", "s2")


def _balanced_sources(count: int, ep: int) -> np.ndarray:
    base, remainder = divmod(count, ep)
    sizes = np.full(ep, base, dtype=np.int64)
    sizes[:remainder] += 1
    return np.repeat(np.arange(ep, dtype=np.int16), sizes)


def _traffic(expert_ids: np.ndarray, source_ranks: np.ndarray, ep: int):
    expert_ids = np.asarray(expert_ids, dtype=np.int16)
    source_ranks = np.asarray(source_ranks, dtype=np.int16)
    owners = expert_ids // (NUM_EXPERTS // ep)
    assignment = np.zeros((ep, ep), dtype=np.int32)
    unique = np.zeros((ep, ep), dtype=np.int32)
    np.add.at(assignment, (np.repeat(source_ranks, TOP_K), owners.reshape(-1)), 1)
    for destination in range(ep):
        tokens = np.any(owners == destination, axis=1)
        if np.any(tokens):
            unique[:, destination] = np.bincount(
                source_ranks[tokens], minlength=ep
            ).astype(np.int32)
    if len(expert_ids):
        ordered = np.sort(owners, axis=1)
        fanout = 1 + np.count_nonzero(ordered[:, 1:] != ordered[:, :-1], axis=1)
        fanout_mean, fanout_max = float(fanout.mean()), int(fanout.max())
    else:
        fanout_mean = 0.0
        fanout_max = 0
    return assignment, unique, fanout_mean, fanout_max


def _similarity(left: torch.Tensor, right: torch.Tensor):
    left32, right32 = left.float(), right.float()
    dot = (left32 * right32).sum(dim=-1)
    cosine = dot / (left32.norm(dim=-1) * right32.norm(dim=-1)).clamp_min(1e-12)
    rel_l2 = (left32 - right32).norm(dim=-1) / right32.norm(dim=-1).clamp_min(1e-12)
    return cosine.detach().cpu().numpy(), rel_l2.detach().cpu().numpy()


@dataclass
class GenerationResult:
    generated: torch.Tensor
    records: list[dict[str, Any]]
    drift: list[dict[str, Any]]
    edge_stability: list[dict[str, Any]] | None = None


class SelectiveDenseEmulator:
    """Per-request hook set for structural tracing and F0/F1 emulation."""

    def __init__(self, model, semantics: str, policy: PolicyConfig, request_id: int,
                 capture_adjacent_hidden: bool = False,
                 capture_temporal_edges: bool = False,
                 temporal_edge_layers: tuple[int, ...] = (1, 5, 10, 14, 19)):
        if semantics not in ("none", "f0", "f1"):
            raise ValueError("semantics must be none/f0/f1")
        self.model = model
        self.semantics = semantics
        self.policy = policy
        self.request_id = int(request_id)
        self.capture_adjacent_hidden = bool(capture_adjacent_hidden)
        self.capture_temporal_edges = bool(capture_temporal_edges)
        self.temporal_edge_layers = tuple(int(layer) for layer in temporal_edge_layers)
        self.routed_layers = [
            index for index, layer in enumerate(model.model.layers)
            if hasattr(layer.mlp, "gate")
        ]
        self.route_slot = {layer: slot for slot, layer in enumerate(self.routed_layers)}
        self.records: list[dict[str, Any]] = []
        self.drift: list[dict[str, Any]] = []
        self.edge_stability: list[dict[str, Any]] = []
        self._handles = []
        self._original_moe_infer = {}
        self._context: dict[str, Any] | None = None
        self._block_id = -1
        self._f0_cache: dict[int, torch.Tensor] = {}
        self._f1_cache: dict[int, torch.Tensor] = {}
        self._previous_hidden: dict[int, torch.Tensor] = {}
        self._previous_hidden_mask: torch.Tensor | None = None
        self._previous_edge_state: dict[int, dict[str, Any]] = {}
        self.previous_confidence: np.ndarray | None = None
        self.cached_confidence: np.ndarray | None = None
        self.previous_token_rank_load: dict[int, np.ndarray] = {}
        self.previous_base_rank_load: dict[int, np.ndarray] = {}
        self.freeze_age = np.zeros(BLOCK_LENGTH, dtype=np.int16)
        self.newly_decoded = np.zeros(BLOCK_LENGTH, dtype=bool)
        self._install()

    def _reset_block(self, block_id: int):
        self._block_id = int(block_id)
        self._f0_cache.clear()
        self._f1_cache.clear()
        self._previous_hidden.clear()
        self._previous_hidden_mask = None
        self._previous_edge_state.clear()
        self.previous_confidence = None
        self.cached_confidence = None
        self.previous_token_rank_load.clear()
        self.previous_base_rank_load.clear()
        self.freeze_age.fill(0)
        self.newly_decoded.fill(False)

    def choose(self, masked: np.ndarray, block_id: int, iteration: int):
        if block_id != self._block_id:
            self._reset_block(block_id)
        token_load = self.previous_token_rank_load.get(self.policy.target_ep)
        base_load = self.previous_base_rank_load.get(self.policy.target_ep)
        return choose_active_set(
            masked, self.previous_confidence, self.freeze_age, self.policy,
            block_iteration=iteration, token_rank_load=token_load,
            base_rank_load=base_load,
            random_seed=20260917 + self.request_id * 1009 + block_id * 37 + iteration,
        )

    def begin_iteration(
        self, *, block_id: int, iteration: int, nfe: int, physical_rows: int,
        masked: np.ndarray, active: np.ndarray, forced: np.ndarray,
    ):
        if self._context is not None:
            raise RuntimeError("nested selective-refinement context")
        if block_id != self._block_id:
            self._reset_block(block_id)
        current_start = physical_rows - BLOCK_LENGTH
        full_masks = {}
        s0 = np.ones(physical_rows, dtype=bool)
        s1 = s0.copy()
        s1[current_start:][masked & ~active] = False
        s2 = np.zeros(physical_rows, dtype=bool)
        s2[current_start:][(masked & active) | self.newly_decoded] = True
        full_masks.update(s0=s0, s1=s1, s2=s2)
        layer_count = len(self.routed_layers)
        context: dict[str, Any] = {
            "block_id": int(block_id), "iteration_id": int(iteration), "nfe": int(nfe),
            "physical_rows": int(physical_rows), "masked": masked.copy(),
            "active": active.copy(), "forced": forced.copy(),
            "newly_decoded": self.newly_decoded.copy(), "full_masks": full_masks,
            "routes": np.full((layer_count, BLOCK_LENGTH, TOP_K), -1, dtype=np.int16),
            "route_weights": np.zeros((layer_count, BLOCK_LENGTH, TOP_K), dtype=np.float16),
        }
        for substrate in SUBSTRATES:
            context[f"hist_{substrate}"] = np.zeros((layer_count, NUM_EXPERTS), dtype=np.uint16)
            for ep in ROUTED_EPS:
                context[f"a_{substrate}_ep{ep}"] = np.zeros((layer_count, ep, ep), dtype=np.int32)
                context[f"u_{substrate}_ep{ep}"] = np.zeros((layer_count, ep, ep), dtype=np.int32)
                context[f"fanout_mean_{substrate}_ep{ep}"] = np.zeros(layer_count, dtype=np.float32)
                context[f"fanout_max_{substrate}_ep{ep}"] = np.zeros(layer_count, dtype=np.int8)
        self._context = context

    def _router_hook(self, layer_id: int):
        slot = self.route_slot[layer_id]

        def hook(_module, _inputs, output):
            context = self._context
            if context is None:
                raise RuntimeError("router outside active refinement context")
            topk_ids = output[0].detach().reshape(-1, TOP_K).cpu().numpy().astype(np.int16)
            topk_weight = output[1].detach().reshape(-1, TOP_K).cpu().numpy().astype(np.float16)
            physical = context["physical_rows"]
            if len(topk_ids) != physical:
                raise RuntimeError("router row count mismatch")
            current_start = physical - BLOCK_LENGTH
            context["routes"][slot] = topk_ids[current_start:]
            context["route_weights"][slot] = topk_weight[current_start:]
            for substrate in SUBSTRATES:
                include = context["full_masks"][substrate]
                ids = topk_ids[include]
                context[f"hist_{substrate}"][slot] = np.bincount(
                    ids.reshape(-1), minlength=NUM_EXPERTS
                ).astype(np.uint16)
                # Traffic is evaluated for both targets using the same
                # original-row-preserving compaction contract.
                for ep in ROUTED_EPS:
                    sources = _balanced_sources(physical, ep)[include]
                    a, u, mean_fanout, max_fanout = _traffic(ids, sources, ep)
                    context[f"a_{substrate}_ep{ep}"][slot] = a
                    context[f"u_{substrate}_ep{ep}"][slot] = u
                    context[f"fanout_mean_{substrate}_ep{ep}"][slot] = mean_fanout
                    context[f"fanout_max_{substrate}_ep{ep}"][slot] = max_fanout

        return hook

    def _record_drift(
        self, boundary: str, layer_id: int, fresh: torch.Tensor,
        cached: torch.Tensor, positions: np.ndarray,
    ):
        if positions.size == 0:
            return
        index = torch.as_tensor(positions, device=fresh.device)
        cosine, rel_l2 = _similarity(fresh[index], cached[index])
        age = self.freeze_age[positions]
        for position, token_age, cos, l2 in zip(positions, age, cosine, rel_l2):
            self.drift.append({
                "request_id": self.request_id, "block_id": self._block_id,
                "iteration_id": int(self._context["iteration_id"]),
                "boundary": boundary, "layer_id": int(layer_id),
                "position": int(position), "freeze_age": int(token_age),
                "cosine": float(cos), "relative_l2": float(l2),
            })

    def _wrap_moe_infer(self, layer_id: int, module, original):
        controller = self

        def wrapped(_module_self, x, topk_ids, topk_weight):
            fresh = original(x, topk_ids, topk_weight)
            context = controller._context
            if context is None:
                return fresh
            if controller.capture_temporal_edges and layer_id in controller.temporal_edge_layers:
                controller._capture_temporal_edge_branches(
                    layer_id, _module_self, x, topk_ids, topk_weight, fresh
                )
            if controller.semantics != "f0":
                return fresh
            seq = context["physical_rows"]
            current = fresh.view(1, seq, -1)[:, -BLOCK_LENGTH:, :][0]
            cache = controller._f0_cache.get(layer_id)
            frozen = context["masked"] & ~context["active"]
            if cache is None:
                cache = current.detach().clone()
            elif np.any(frozen):
                positions = np.flatnonzero(frozen)
                controller._record_drift("routed_moe_output", layer_id, current, cache, positions)
                modified = fresh.view(1, seq, -1).clone()
                index = torch.as_tensor(positions + seq - BLOCK_LENGTH, device=fresh.device)
                cache_index = torch.as_tensor(positions, device=cache.device)
                modified[0, index] = cache[cache_index]
                fresh = modified.view(seq, -1)
            update = np.flatnonzero(~frozen)
            if update.size:
                update_index = torch.as_tensor(update, device=cache.device)
                cache[update_index] = current.detach()[update_index]
            controller._f0_cache[layer_id] = cache
            return fresh

        return MethodType(wrapped, module)

    @staticmethod
    def _vector_metrics(current: torch.Tensor, previous: torch.Tensor):
        current32, previous32 = current.float(), previous.float()
        current_norm = current32.norm(dim=-1)
        previous_norm = previous32.norm(dim=-1)
        cosine = (current32 * previous32).sum(dim=-1) / (
            current_norm * previous_norm
        ).clamp_min(1e-12)
        relative_l2 = (current32 - previous32).norm(dim=-1) / previous_norm.clamp_min(1e-12)
        norm_ratio = current_norm / previous_norm.clamp_min(1e-12)
        return (
            cosine.detach().cpu().numpy(),
            relative_l2.detach().cpu().numpy(),
            norm_ratio.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def _capture_temporal_edge_branches(
        self, layer_id: int, module, x: torch.Tensor, topk_ids: torch.Tensor,
        topk_weight: torch.Tensor, fresh: torch.Tensor,
    ):
        """Capture exact adjacent expert-branch drift without changing outputs.

        The production MoE result above is returned untouched.  Only five
        selected routed layers re-execute the 32 current-block rows to expose
        per-branch vectors for this bounded diagnostic trace.
        """

        context = self._context
        assert context is not None
        physical = int(context["physical_rows"])
        if x.shape[0] != physical:
            raise RuntimeError("temporal-edge row count mismatch")
        current_x = x[-BLOCK_LENGTH:].detach()
        current_ids = topk_ids[-BLOCK_LENGTH:].detach()
        current_weight = topk_weight[-BLOCK_LENGTH:].detach()
        current_mask_np = context["masked"].astype(bool, copy=False)
        current_mask = torch.as_tensor(current_mask_np, device=x.device)
        branch = torch.zeros(
            (BLOCK_LENGTH, TOP_K, x.shape[-1]), dtype=x.dtype, device=x.device
        )
        if bool(current_mask.any()):
            masked_ids = current_ids[current_mask]
            for expert_id in torch.unique(masked_ids).tolist():
                route = (current_ids == int(expert_id)) & current_mask[:, None]
                position, slot = route.nonzero(as_tuple=True)
                if position.numel():
                    branch[position, slot] = module.experts[int(expert_id)](current_x[position])

        previous = self._previous_edge_state.get(layer_id)
        age = np.ones((BLOCK_LENGTH, TOP_K), dtype=np.int16)
        if previous is not None:
            previous_ids = previous["ids"]
            previous_mask = previous["mask"]
            pairs: list[tuple[int, int, int, int]] = []
            for position in np.flatnonzero(current_mask_np & previous_mask):
                prior = {int(expert): slot for slot, expert in enumerate(previous_ids[position])}
                for current_slot, expert in enumerate(current_ids[position].tolist()):
                    prior_slot = prior.get(int(expert))
                    if prior_slot is not None:
                        pairs.append((int(position), current_slot, prior_slot, int(expert)))
                        age[position, current_slot] = int(previous["age"][position, prior_slot]) + 1
            if pairs:
                position = torch.as_tensor([pair[0] for pair in pairs], device=x.device)
                current_slot = torch.as_tensor([pair[1] for pair in pairs], device=x.device)
                previous_slot = torch.as_tensor([pair[2] for pair in pairs], device=x.device)
                current_branch = branch[position, current_slot]
                previous_branch = previous["branch"].to(x.device)[position, previous_slot]
                raw_cos, raw_l2, norm_ratio = self._vector_metrics(current_branch, previous_branch)
                current_w = current_weight[position, current_slot].float()
                previous_w = previous["weight"].to(x.device)[position, previous_slot].float()
                r0_cos, r0_l2, _ = self._vector_metrics(
                    current_branch.float() * current_w[:, None],
                    previous_branch.float() * previous_w[:, None],
                )
                _r1_cos, r1_l2, _ = self._vector_metrics(
                    current_branch.float() * current_w[:, None],
                    previous_branch.float() * current_w[:, None],
                )
                hidden_cos, hidden_l2, _ = self._vector_metrics(
                    current_x[position], previous["input"].to(x.device)[position]
                )
                current_combined = fresh[-BLOCK_LENGTH:][position]
                previous_combined = previous["combined"].to(x.device)[position]
                post_cos, post_l2, _ = self._vector_metrics(current_combined, previous_combined)
                current_w_np = current_w.detach().cpu().numpy()
                previous_w_np = previous_w.detach().cpu().numpy()
                for ordinal, pair in enumerate(pairs):
                    pos, cur_slot, prior_slot, expert = pair
                    self.edge_stability.append({
                        "request_id": self.request_id,
                        "block_id": self._block_id,
                        "iteration_id": int(context["iteration_id"]),
                        "layer_id": int(layer_id),
                        "token_position": pos,
                        "expert_id": expert,
                        "current_slot": cur_slot,
                        "previous_slot": prior_slot,
                        "route_age": int(age[pos, cur_slot]),
                        "mask_ratio": float(current_mask_np.mean()),
                        "previous_confidence": float(
                            self.previous_confidence[pos]
                            if self.previous_confidence is not None else np.nan
                        ),
                        "current_weight": float(current_w_np[ordinal]),
                        "previous_weight": float(previous_w_np[ordinal]),
                        "weight_abs_drift": float(abs(current_w_np[ordinal] - previous_w_np[ordinal])),
                        "weight_relative_drift": float(
                            abs(current_w_np[ordinal] - previous_w_np[ordinal])
                            / max(abs(previous_w_np[ordinal]), 1e-12)
                        ),
                        "raw_output_cosine": float(raw_cos[ordinal]),
                        "raw_output_relative_l2": float(raw_l2[ordinal]),
                        "raw_output_norm_ratio": float(norm_ratio[ordinal]),
                        "weighted_r0_cosine": float(r0_cos[ordinal]),
                        "weighted_r0_relative_l2": float(r0_l2[ordinal]),
                        "weighted_r1_relative_l2": float(r1_l2[ordinal]),
                        "input_hidden_cosine": float(hidden_cos[ordinal]),
                        "input_hidden_relative_l2": float(hidden_l2[ordinal]),
                        "post_moe_cosine": float(post_cos[ordinal]),
                        "post_moe_relative_l2": float(post_l2[ordinal]),
                    })

        self._previous_edge_state[layer_id] = {
            "ids": current_ids.detach().cpu().numpy().astype(np.int16),
            "mask": current_mask_np.copy(),
            "weight": current_weight.detach().cpu(),
            "input": current_x.detach().cpu(),
            "branch": branch.detach().cpu(),
            "combined": fresh[-BLOCK_LENGTH:].detach().cpu(),
            "age": age,
        }

    def _decoder_hook(self, layer_id: int):
        controller = self

        def hook(_module, _inputs, output):
            context = controller._context
            if context is None:
                return output
            hidden = output[0]
            current = hidden[:, -BLOCK_LENGTH:, :][0]
            # Selected layers supply adjacent hidden-state predictability even
            # when no freeze emulation is active.
            if controller.capture_adjacent_hidden and layer_id in (0, 9, 19):
                previous = controller._previous_hidden.get(layer_id)
                if previous is not None and controller._previous_hidden_mask is not None:
                    positions = np.flatnonzero(
                        context["masked"] & controller._previous_hidden_mask.cpu().numpy()
                    )
                    if positions.size:
                        index = torch.as_tensor(positions, device=current.device)
                        cosine, rel_l2 = _similarity(current[index], previous[index])
                        for position, cos, l2 in zip(positions, cosine, rel_l2):
                            controller.drift.append({
                                "request_id": controller.request_id,
                                "block_id": controller._block_id,
                                "iteration_id": int(context["iteration_id"]),
                                "boundary": "adjacent_hidden", "layer_id": layer_id,
                                "position": int(position), "freeze_age": 0,
                                "cosine": float(cos), "relative_l2": float(l2),
                            })
                controller._previous_hidden[layer_id] = current.detach().clone()
            if controller.semantics != "f1":
                return output
            cache = controller._f1_cache.get(layer_id)
            frozen = context["masked"] & ~context["active"]
            if cache is None:
                cache = current.detach().clone()
            elif np.any(frozen):
                positions = np.flatnonzero(frozen)
                controller._record_drift("post_layer_state", layer_id, current, cache, positions)
                modified = hidden.clone()
                index = torch.as_tensor(positions + hidden.shape[1] - BLOCK_LENGTH,
                                        device=hidden.device)
                cache_index = torch.as_tensor(positions, device=cache.device)
                modified[0, index] = cache[cache_index]
                output = (modified,) + tuple(output[1:])
            update = np.flatnonzero(~frozen)
            if update.size:
                update_index = torch.as_tensor(update, device=cache.device)
                cache[update_index] = current.detach()[update_index]
            controller._f1_cache[layer_id] = cache
            return output

        return hook

    def _install(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, "gate"):
                self._handles.append(layer.mlp.gate.register_forward_hook(
                    self._router_hook(layer_id)
                ))
                if self.semantics == "f0" or (
                    self.capture_temporal_edges and layer_id in self.temporal_edge_layers
                ):
                    original = layer.mlp.moe_infer
                    self._original_moe_infer[layer_id] = original
                    layer.mlp.moe_infer = self._wrap_moe_infer(layer_id, layer.mlp, original)
            self._handles.append(layer.register_forward_hook(self._decoder_hook(layer_id)))

    def finalize_iteration(
        self, confidence: torch.Tensor, transfer: torch.Tensor,
        counterfactual_transfer: torch.Tensor,
    ):
        context = self._context
        if context is None:
            raise RuntimeError("no active refinement context")
        conf = confidence.detach().float().cpu().numpy()[0]
        transfer_np = transfer.detach().cpu().numpy()[0].astype(bool)
        counterfactual_np = counterfactual_transfer.detach().cpu().numpy()[0].astype(bool)
        record = {key: value for key, value in context.items() if key != "full_masks"}
        confidence_error = np.full(BLOCK_LENGTH, np.nan, dtype=np.float32)
        masked_positions = np.flatnonzero(context["masked"])
        if masked_positions.size:
            reference_confidence = conf if self.cached_confidence is None else self.cached_confidence
            confidence_error[masked_positions] = np.abs(
                conf[masked_positions] - reference_confidence[masked_positions]
            )
        record.update(
            confidence=conf.astype(np.float32),
            transfer=transfer_np,
            counterfactual_transfer=counterfactual_np,
            confidence_cache_error=confidence_error,
            commit_disagreement=(transfer_np != counterfactual_np),
        )
        self.records.append(record)
        # Build next-step online rank features only from the just-observed routes.
        routes = context["routes"]
        current_mask = context["masked"]
        for ep in ROUTED_EPS:
            owners = routes // (NUM_EXPERTS // ep)
            per_token = np.zeros((BLOCK_LENGTH, ep), dtype=np.float32)
            for rank in range(ep):
                per_token[:, rank] = np.count_nonzero(owners == rank, axis=(0, 2))
            full_rank = context[f"hist_s0"].reshape(-1, NUM_EXPERTS).sum(axis=0)
            full_rank = full_rank.reshape(ep, NUM_EXPERTS // ep).sum(axis=1).astype(np.float32)
            masked_rank = per_token[current_mask].sum(axis=0)
            self.previous_token_rank_load[ep] = per_token
            self.previous_base_rank_load[ep] = full_rank - masked_rank
        self.previous_confidence = conf.copy()
        if self.cached_confidence is None:
            self.cached_confidence = conf.copy()
        else:
            refreshed = context["active"] | ~context["masked"]
            self.cached_confidence[refreshed] = conf[refreshed]
        prior_age = self.freeze_age.copy()
        self.freeze_age[~context["masked"]] = 0
        self.freeze_age[context["masked"] & context["active"]] = 0
        self.freeze_age[context["masked"] & ~context["active"]] = (
            prior_age[context["masked"] & ~context["active"]] + 1
        )
        self.newly_decoded = transfer_np.copy()
        self._previous_hidden_mask = torch.as_tensor(context["masked"].copy())
        self._context = None

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer_id, original in self._original_moe_infer.items():
            self.model.model.layers[layer_id].mlp.moe_infer = original
        self._original_moe_infer.clear()
        self._context = None


def _transfer_mask(confidence, eligible, threshold, num_to_transfer):
    transfer = torch.zeros_like(eligible, dtype=torch.bool)
    restricted = torch.where(eligible, confidence, -torch.inf)
    high = restricted[0] > threshold
    if int(high.sum().item()) >= num_to_transfer:
        transfer[0] = high
    else:
        count = min(num_to_transfer, int(eligible.sum().item()))
        if count:
            _, index = torch.topk(restricted[0], k=count)
            transfer[0, index] = True
    return transfer


@torch.no_grad()
def generate_selective(
    model, inputs: torch.Tensor, *, policy: PolicyConfig,
    semantics: str = "none", temperature: float = 0.0,
    threshold: float = 0.95, block_length: int = 32, steps: int = 32,
    gen_length: int = 2048, eos_early_stop: bool = True,
    eos_id: int = 156892, mask_id: int = 156895, request_id: int = 0,
    capture_adjacent_hidden: bool = False,
    capture_temporal_edges: bool = False,
    temporal_edge_layers: tuple[int, ...] = (1, 5, 10, 14, 19),
) -> GenerationResult:
    """Replicate official generation with optional dense freeze emulation."""

    if block_length != BLOCK_LENGTH:
        raise ValueError("this PoC is fixed to block_length=32")
    steps = min(steps, gen_length)
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
    controller = SelectiveDenseEmulator(
        model, semantics, policy, request_id,
        capture_adjacent_hidden=capture_adjacent_hidden,
        capture_temporal_edges=capture_temporal_edges,
        temporal_edge_layers=temporal_edge_layers,
    )
    nfe = 0
    try:
        for num_block in range(prefill_blocks, num_blocks):
            end = (num_block + 1) * block_length
            cur_x = x[:, :end]
            cur_mask = attention_mask[:, :, :end, :end]
            cur_positions = position_ids[:, :end]
            logical_block = num_block - prefill_blocks
            for step in range(steps):
                masked_tensor = cur_x[:, -block_length:] == mask_id
                if int(masked_tensor.sum()) == 0:
                    break
                masked = masked_tensor[0].detach().cpu().numpy()
                active, forced = controller.choose(masked, logical_block, step)
                controller.begin_iteration(
                    block_id=logical_block, iteration=step, nfe=nfe,
                    physical_rows=end, masked=masked, active=active, forced=forced,
                )
                logits = model.forward(
                    cur_x, attention_mask=cur_mask, position_ids=cur_positions,
                ).logits
                nfe += 1
                active_logits = logits[:, -block_length:, :]
                x0, x0_p = model._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=None, top_p=None
                )
                confidence = torch.where(masked_tensor, x0_p, -torch.inf)
                active_tensor = torch.as_tensor(active, device=model.device).unsqueeze(0)
                eligible = masked_tensor & active_tensor
                num_to_transfer = int(schedule[step].item())
                transfer = _transfer_mask(confidence, eligible, threshold, num_to_transfer)
                counterfactual = _transfer_mask(
                    confidence, masked_tensor, threshold, num_to_transfer
                )
                controller.finalize_iteration(confidence, transfer, counterfactual)
                if transfer.any():
                    cur_x[:, -block_length:][transfer] = x0[transfer]
                if eos_early_stop and (x0[transfer] == eos_id).any():
                    eos_positions = (cur_x[0] == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions):
                        eos_position = int(eos_positions[0].item())
                        if (cur_x[0, prompt_length:eos_position] != mask_id).all():
                            final = x[:, :total_length][:, :eos_position + 1]
                            return GenerationResult(
                                final, controller.records, controller.drift,
                                controller.edge_stability,
                            )
            x[:, :end] = cur_x
            if eos_id is not None and (x[0, prompt_length:end] == eos_id).any():
                break
        answer = x[:, :prompt_length + gen_length]
        eos_positions = (answer[0][prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
        first = int(eos_positions[0].item()) if len(eos_positions) else gen_length
        return GenerationResult(
            answer[:, prompt_length:prompt_length + first + 1],
            controller.records, controller.drift, controller.edge_stability,
        )
    finally:
        controller.close()


def save_request_trace(path: Path, result: GenerationResult, metadata: dict[str, Any]):
    """Save one compact fixed-shape request trace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = result.records
    if not records:
        raise ValueError("cannot save an empty trace")
    scalar = ("block_id", "iteration_id", "nfe", "physical_rows")
    vectors = ("masked", "active", "forced", "newly_decoded", "confidence",
               "transfer", "counterfactual_transfer", "confidence_cache_error",
               "commit_disagreement")
    payload: dict[str, np.ndarray] = {
        key: np.asarray([row[key] for row in records]) for key in scalar
    }
    payload.update({key: np.stack([row[key] for row in records]) for key in vectors})
    payload["routes"] = np.stack([row["routes"] for row in records])
    payload["route_weights"] = np.stack([row["route_weights"] for row in records])
    for substrate in SUBSTRATES:
        payload[f"hist_{substrate}"] = np.stack([row[f"hist_{substrate}"] for row in records])
        for ep in ROUTED_EPS:
            for prefix in ("a", "u", "fanout_mean", "fanout_max"):
                key = f"{prefix}_{substrate}_ep{ep}"
                payload[key] = np.stack([row[key] for row in records])
    if result.drift:
        for key in ("request_id", "block_id", "iteration_id", "layer_id", "position", "freeze_age"):
            payload[f"drift_{key}"] = np.asarray([row[key] for row in result.drift], dtype=np.int32)
        payload["drift_boundary"] = np.asarray([row["boundary"] for row in result.drift], dtype=np.str_)
        payload["drift_cosine"] = np.asarray([row["cosine"] for row in result.drift], dtype=np.float32)
        payload["drift_relative_l2"] = np.asarray([row["relative_l2"] for row in result.drift], dtype=np.float32)
    if result.edge_stability:
        integer_keys = (
            "request_id", "block_id", "iteration_id", "layer_id", "token_position",
            "expert_id", "current_slot", "previous_slot", "route_age",
        )
        float_keys = tuple(
            key for key in result.edge_stability[0]
            if key not in integer_keys
        )
        for key in integer_keys:
            payload[f"edge_{key}"] = np.asarray(
                [row[key] for row in result.edge_stability], dtype=np.int32
            )
        for key in float_keys:
            payload[f"edge_{key}"] = np.asarray(
                [row[key] for row in result.edge_stability], dtype=np.float32
            )
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(path, **payload)
