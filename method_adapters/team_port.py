"""Auditable LLaDA2.0-mini port of TEAM's DCD and LAC mechanisms.

The public TEAM implementation is SDAR-specific.  This adapter preserves the
official LLaDA2 decoder, threshold, prompt and transfer schedule while porting
the two expert-execution mechanisms that can be expressed without changing
the checkpoint:

* decoded-token delayed/reuse caching (routed-MoE output boundary), and
* hot/cold masked-token classification plus necessary-expert restricted routing.

The public four-way speculative exploration code is audited and its candidate
construction is unit-tested, but it is not executed in this port.  Results are
therefore labelled TEAM-PORT-DCD-LAC, never official TEAM.  The model executes
dense work and replaces outputs to validate semantics; timing comes only from
offline fresh-route projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from selective_refinement.emulator import (
    BLOCK_LENGTH, NUM_EXPERTS, TOP_K, GenerationResult, _traffic, _transfer_mask,
)


ROUTED_EPS = (1, 4, 8)


@dataclass(frozen=True)
class TeamPortConfig:
    hot_confidence: float = 0.7
    hot_distance: int = 3
    speculative_width: int = 4


def team_speculative_candidates(
    tokens: torch.Tensor, proposals: torch.Tensor, masked: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the official base/top1/top2/both candidate family.

    This helper is independent of model execution so exact construction can be
    tested even though the LLaDA2 port does not claim the SDAR-specific SEH
    acceptance path.
    """

    if tokens.shape[0] != 1 or proposals.shape != tokens.shape or masked.shape != tokens.shape:
        raise ValueError("TEAM candidate construction expects matching [1, block] tensors")
    eligible = torch.where(masked[0])[0]
    if not len(eligible):
        return tokens.repeat(4, 1), eligible
    chosen = eligible[: min(2, len(eligible))]
    blocks = [tokens.clone() for _ in range(4)]
    blocks[1][0, chosen[0]] = proposals[0, chosen[0]]
    if len(chosen) > 1:
        blocks[2][0, chosen[1]] = proposals[0, chosen[1]]
        blocks[3][0, chosen[1]] = proposals[0, chosen[1]]
    blocks[3][0, chosen[0]] = proposals[0, chosen[0]]
    return torch.cat(blocks, dim=0), chosen


def classify_hot_cold(
    masked: np.ndarray, previous_confidence: np.ndarray | None,
    decoded: np.ndarray, *, threshold: float = 0.7, distance: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """TEAM hot/cold classification using only prior-refinement observations."""

    masked = np.asarray(masked, dtype=bool)
    decoded = np.asarray(decoded, dtype=bool)
    hot = np.zeros_like(masked)
    positions = np.flatnonzero(masked)
    if not len(positions):
        return hot, hot.copy()
    confidence = (np.zeros(len(masked), dtype=np.float32) if previous_confidence is None
                  else np.asarray(previous_confidence, dtype=np.float32))
    hot[positions] |= confidence[positions] >= threshold
    # Official TEAM always keeps the two exploration candidates unrestricted.
    order = positions[np.argsort(confidence[positions], kind="stable")]
    hot[order[-min(2, len(order)):]] = True
    anchors = np.flatnonzero(decoded)
    if len(anchors):
        nearest = np.abs(positions[:, None] - anchors[None, :]).min(axis=1)
        hot[positions[nearest <= distance]] = True
    return hot, masked & ~hot


class TeamPortController:
    def __init__(self, model, request_id: int, config: TeamPortConfig):
        self.model = model
        self.request_id = int(request_id)
        self.config = config
        self.routed_layers = [
            i for i, layer in enumerate(model.model.layers) if hasattr(layer.mlp, "gate")
        ]
        self.layer_slot = {layer: slot for slot, layer in enumerate(self.routed_layers)}
        self.records: list[dict[str, Any]] = []
        self.previous_confidence: np.ndarray | None = None
        self.newly_decoded = np.zeros(BLOCK_LENGTH, dtype=bool)
        self._moe_cache: dict[int, torch.Tensor] = {}
        self._route_cache: dict[int, np.ndarray] = {}
        self._weight_cache: dict[int, np.ndarray] = {}
        self._original_gate = {}
        self._original_moe = {}
        self._handles = []
        self._context: dict[str, Any] | None = None
        self._block = -1
        self._install()

    def reset_block(self, block_id: int):
        self._block = int(block_id)
        self.previous_confidence = None
        self.newly_decoded.fill(False)
        self._moe_cache.clear(); self._route_cache.clear(); self._weight_cache.clear()

    def begin(self, block_id: int, iteration: int, nfe: int,
              physical_rows: int, masked: np.ndarray):
        if block_id != self._block:
            self.reset_block(block_id)
        decoded = ~masked
        reused = decoded & ~self.newly_decoded
        fresh = ~reused
        hot, cold = classify_hot_cold(
            masked, self.previous_confidence, decoded,
            threshold=self.config.hot_confidence, distance=self.config.hot_distance,
        )
        semantic = np.full(BLOCK_LENGTH, 2, dtype=np.int8)
        semantic[masked] = 0
        semantic[self.newly_decoded] = 1
        layer_count = len(self.routed_layers)
        self._context = {
            "block_id": int(block_id), "iteration_id": int(iteration), "nfe": int(nfe),
            "physical_rows": int(physical_rows), "masked": masked.copy(),
            "fresh": fresh, "reused": reused, "hot": hot, "cold": cold,
            "semantic_state": semantic,
            "routes": np.full((layer_count, BLOCK_LENGTH, TOP_K), -1, dtype=np.int16),
            "route_weights": np.zeros((layer_count, BLOCK_LENGTH, TOP_K), dtype=np.float16),
            "necessary_expert_count": np.zeros(layer_count, dtype=np.int16),
        }
        for ep in ROUTED_EPS:
            self._context[f"a_ep{ep}"] = np.zeros((layer_count, ep, ep), dtype=np.int32)
            self._context[f"u_ep{ep}"] = np.zeros((layer_count, ep, ep), dtype=np.int32)
            self._context[f"fanout_mean_ep{ep}"] = np.zeros(layer_count, dtype=np.float32)
            self._context[f"fanout_max_ep{ep}"] = np.zeros(layer_count, dtype=np.int8)
        self._context["hist"] = np.zeros((layer_count, NUM_EXPERTS), dtype=np.uint16)

    def _wrap_gate(self, layer_id: int, gate, original):
        controller = self
        def wrapped(_gate_self, hidden_states):
            topk_idx, topk_weight, logits = original(hidden_states)
            context = controller._context
            if context is None:
                return topk_idx, topk_weight, logits
            rows = topk_idx.shape[0]
            start = rows - BLOCK_LENGTH
            cold_full = torch.zeros(rows, dtype=torch.bool, device=topk_idx.device)
            fresh_full = torch.zeros(rows, dtype=torch.bool, device=topk_idx.device)
            cold_full[start:] = torch.as_tensor(context["cold"], device=topk_idx.device)
            fresh_full[start:] = torch.as_tensor(context["fresh"], device=topk_idx.device)
            necessary_rows = fresh_full & ~cold_full
            if cold_full.any() and necessary_rows.any():
                necessary = torch.zeros(gate.num_experts, dtype=torch.bool, device=logits.device)
                necessary[topk_idx[necessary_rows].reshape(-1)] = True
                # At least one unrestricted top-8 token is guaranteed by the
                # top-two hot rule, so the restricted set has >=8 experts.
                scores = torch.sigmoid(logits.float())
                restricted = scores[cold_full].masked_fill(~necessary[None, :], -torch.inf)
                values, ids = torch.topk(restricted, k=gate.top_k, dim=-1)
                values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-20)
                values = values * gate.routed_scaling_factor
                topk_idx = topk_idx.clone(); topk_weight = topk_weight.clone()
                topk_idx[cold_full] = ids
                topk_weight[cold_full] = values.to(topk_weight.dtype)
                context["necessary_expert_count"][controller.layer_slot[layer_id]] = int(necessary.sum())
            return topk_idx, topk_weight, logits
        return MethodType(wrapped, gate)

    def _route_hook(self, layer_id: int):
        slot = self.layer_slot[layer_id]
        controller = self
        def hook(_module, _inputs, output):
            context = controller._context
            if context is None:
                return
            ids = output[0].detach().reshape(-1, TOP_K).cpu().numpy().astype(np.int16)
            weights = output[1].detach().reshape(-1, TOP_K).cpu().numpy().astype(np.float16)
            current_ids = ids[-BLOCK_LENGTH:].copy()
            current_weights = weights[-BLOCK_LENGTH:].copy()
            cached_ids = controller._route_cache.get(layer_id)
            cached_weights = controller._weight_cache.get(layer_id)
            reused = context["reused"]
            if cached_ids is not None and np.any(reused):
                current_ids[reused] = cached_ids[reused]
                current_weights[reused] = cached_weights[reused]
            context["routes"][slot] = current_ids
            context["route_weights"][slot] = current_weights
            fresh_ids = current_ids[context["fresh"]]
            context["hist"][slot] = np.bincount(
                fresh_ids.reshape(-1), minlength=NUM_EXPERTS
            ).astype(np.uint16)
            for ep in ROUTED_EPS:
                sources = np.zeros(len(fresh_ids), dtype=np.int16) if ep == 1 else np.arange(
                    len(fresh_ids), dtype=np.int64
                ) * ep // max(len(fresh_ids), 1)
                sources = np.minimum(sources, ep - 1).astype(np.int16)
                a, u, mean_fanout, max_fanout = _traffic(fresh_ids, sources, ep)
                context[f"a_ep{ep}"][slot] = a; context[f"u_ep{ep}"][slot] = u
                context[f"fanout_mean_ep{ep}"][slot] = mean_fanout
                context[f"fanout_max_ep{ep}"][slot] = max_fanout
            if cached_ids is None:
                cached_ids = current_ids.copy(); cached_weights = current_weights.copy()
            else:
                cached_ids[context["fresh"]] = current_ids[context["fresh"]]
                cached_weights[context["fresh"]] = current_weights[context["fresh"]]
            controller._route_cache[layer_id] = cached_ids
            controller._weight_cache[layer_id] = cached_weights
        return hook

    def _wrap_moe(self, layer_id: int, module, original):
        controller = self
        def wrapped(_module_self, x, topk_ids, topk_weight):
            output = original(x, topk_ids, topk_weight)
            context = controller._context
            if context is None:
                return output
            seq = context["physical_rows"]
            current = output.view(1, seq, -1)[0, -BLOCK_LENGTH:]
            cache = controller._moe_cache.get(layer_id)
            if cache is None:
                cache = current.detach().clone()
            elif np.any(context["reused"]):
                positions = torch.as_tensor(np.flatnonzero(context["reused"]), device=output.device)
                modified = output.view(1, seq, -1).clone()
                modified[0, positions + seq - BLOCK_LENGTH] = cache[positions]
                output = modified.view(seq, -1)
            update = torch.as_tensor(np.flatnonzero(context["fresh"]), device=cache.device)
            if len(update):
                cache[update] = current.detach()[update]
            controller._moe_cache[layer_id] = cache
            return output
        return MethodType(wrapped, module)

    def _install(self):
        for layer_id in self.routed_layers:
            module = self.model.model.layers[layer_id].mlp
            gate = module.gate
            self._original_gate[layer_id] = gate.forward
            gate.forward = self._wrap_gate(layer_id, gate, gate.forward)
            self._handles.append(gate.register_forward_hook(self._route_hook(layer_id)))
            self._original_moe[layer_id] = module.moe_infer
            module.moe_infer = self._wrap_moe(layer_id, module, module.moe_infer)

    def finish(self, confidence: torch.Tensor, transfer: torch.Tensor):
        if self._context is None:
            raise RuntimeError("TEAM iteration was not started")
        conf = confidence.detach().float().cpu().numpy()[0]
        moved = transfer.detach().cpu().numpy()[0].astype(bool)
        self._context["confidence"] = conf.astype(np.float32)
        self._context["transfer"] = moved
        self.records.append(self._context)
        self.previous_confidence = conf.copy()
        self.newly_decoded = moved.copy()
        self._context = None

    def close(self):
        for handle in self._handles: handle.remove()
        for layer_id in self.routed_layers:
            module = self.model.model.layers[layer_id].mlp
            module.gate.forward = self._original_gate[layer_id]
            module.moe_infer = self._original_moe[layer_id]


@torch.no_grad()
def generate_team_port(
    model, inputs: torch.Tensor, *, threshold: float = 0.95,
    block_length: int = 32, steps: int = 32, gen_length: int = 2048,
    eos_early_stop: bool = True, eos_id: int = 156892,
    mask_id: int = 156895, request_id: int = 0,
    config: TeamPortConfig = TeamPortConfig(),
) -> GenerationResult:
    if block_length != BLOCK_LENGTH:
        raise ValueError("TEAM port is fixed to block_length=32")
    input_ids = inputs.to(model.device); prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    attention_mask = block_mask.repeat_interleave(32, 0).repeat_interleave(32, 1)[None, None].log().to(torch.bfloat16)
    position_ids = torch.arange(total_length, device=model.device)[None]
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    schedule = model._get_num_transfer_tokens(block_length, steps)
    controller = TeamPortController(model, request_id, config)
    nfe = 0
    try:
        for num_block in range(prefill_blocks, num_blocks):
            end = (num_block + 1) * block_length
            cur_x = x[:, :end]
            logical_block = num_block - prefill_blocks
            for step in range(steps):
                masked_tensor = cur_x[:, -block_length:] == mask_id
                if not masked_tensor.any(): break
                controller.begin(logical_block, step, nfe, end,
                                 masked_tensor[0].cpu().numpy())
                logits = model.forward(
                    cur_x, attention_mask=attention_mask[:, :, :end, :end],
                    position_ids=position_ids[:, :end],
                ).logits[:, -block_length:]
                nfe += 1
                x0, x0_p = model._sample_with_temperature_topk_topp(
                    logits, temperature=0.0, top_k=None, top_p=None
                )
                confidence = torch.where(masked_tensor, x0_p, -torch.inf)
                transfer = _transfer_mask(confidence, masked_tensor, threshold,
                                          int(schedule[step].item()))
                controller.finish(confidence, transfer)
                cur_x[:, -block_length:][transfer] = x0[transfer]
                if eos_early_stop and (x0[transfer] == eos_id).any():
                    eos_positions = (cur_x[0] == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions):
                        position = int(eos_positions[0])
                        if (cur_x[0, prompt_length:position] != mask_id).all():
                            return GenerationResult(x[:, :position + 1], controller.records, [])
            x[:, :end] = cur_x
            if (x[0, prompt_length:end] == eos_id).any(): break
        answer = x[:, :prompt_length + gen_length]
        eos_positions = (answer[0, prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
        first = int(eos_positions[0]) if len(eos_positions) else gen_length
        return GenerationResult(answer[:, prompt_length:prompt_length + first + 1],
                                controller.records, [])
    finally:
        controller.close()


def save_team_trace(path: Path, result: GenerationResult, metadata: dict[str, Any]):
    records = result.records
    if not records: raise ValueError("cannot save empty TEAM trace")
    payload: dict[str, np.ndarray] = {}
    for key in ("block_id", "iteration_id", "nfe", "physical_rows"):
        payload[key] = np.asarray([row[key] for row in records])
    for key in ("masked", "fresh", "reused", "hot", "cold", "semantic_state",
                "confidence", "transfer", "necessary_expert_count", "routes",
                "route_weights", "hist"):
        payload[key] = np.stack([row[key] for row in records])
    for ep in ROUTED_EPS:
        for prefix in ("a", "u", "fanout_mean", "fanout_max"):
            key = f"{prefix}_ep{ep}"
            payload[key] = np.stack([row[key] for row in records])
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
