"""Aggregate discovery trace for repeated-refinement EP analyses.

The v2 schema keeps per-position-class expert histograms and exact routes for
the *current diffusion block* only.  Prefix rows are intentionally aggregated:
this preserves the H1/H2 load statistics while bounding storage for the 128
request discovery cohort.  Current-block routes are sufficient to reconstruct
the dLLM-specific co-routing, placement, and batching controls used by H3/H4.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any
import json

import numpy as np
import torch

from .mapping import ExpertOwnership, SourcePartition
from .schema import TraceBundle, invocation_slices
from .traffic import build_traffic


DISCOVERY_SCHEMA_VERSION = 2
POSITION_CLASSES = (
    "PROMPT_PREFIX",
    "PRIOR_GENERATED_BLOCKS",
    "CURRENT_BLOCK_MASKED",
    "CURRENT_BLOCK_NEWLY_ACCEPTED",
    "CURRENT_BLOCK_DECODED",
)


SCALAR_FIELDS = (
    "request_id",
    "block_id",
    "iteration_id",
    "nfe",
    "layer_id",
    "prompt_tokens",
    "physical_rows",
    "masked_current_block",
    "decoded_current_block",
    "newly_accepted_current_block",
    "accepted_this_iteration",
    "remaining_mask_after",
    "normalized_block_progress",
    "confidence_mean",
    "confidence_min",
    "confidence_max",
    "terminated",
)


def _balanced_sources(row_count: int, ep: int) -> np.ndarray:
    return SourcePartition(ep).ranks(np.arange(row_count, dtype=np.int32))


def _traffic_fields(expert_ids: np.ndarray, hidden_size: int, ep: int) -> dict[str, np.ndarray]:
    traffic = build_traffic(
        expert_ids,
        np.arange(len(expert_ids), dtype=np.int32),
        ExpertOwnership(256, ep),
        SourcePartition(ep),
        hidden_size,
    )
    loads = traffic.rank_expert_assignments
    mean = float(loads.mean()) if len(loads) else 0.0
    return {
        f"rank_load_ep{ep}": loads.astype(np.int32),
        f"unique_rows_ep{ep}": traffic.unique_activation_matrix.sum(axis=0).astype(np.int32),
        f"assignment_matrix_ep{ep}": traffic.assignment_matrix.astype(np.int32),
        f"unique_matrix_ep{ep}": traffic.unique_activation_matrix.astype(np.int32),
        f"outgoing_bytes_ep{ep}": (
            (traffic.unique_activation_matrix - np.diag(np.diag(traffic.unique_activation_matrix)))
            * hidden_size * 2
        ).sum(axis=1).astype(np.int64),
        f"incoming_bytes_ep{ep}": (
            (traffic.unique_activation_matrix - np.diag(np.diag(traffic.unique_activation_matrix)))
            * hidden_size * 2
        ).sum(axis=0).astype(np.int64),
        f"mean_fanout_ep{ep}": np.float32(traffic.token_fanout.mean() if len(traffic.token_fanout) else 0),
        f"max_fanout_ep{ep}": np.int8(traffic.token_fanout.max(initial=0)),
        f"max_mean_ep{ep}": np.float32(loads.max(initial=0) / mean if mean else 0),
        f"rank_cv_ep{ep}": np.float32(loads.std() / mean if mean else 0),
        f"critical_rank_ep{ep}": np.int8(np.argmax(loads) if loads.sum() else -1),
    }


@dataclass(frozen=True)
class DiscoveryTrace:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if int(self.metadata.get("schema_version", -1)) != DISCOVERY_SCHEMA_VERSION:
            raise ValueError("unsupported discovery trace schema")
        missing = sorted(set(SCALAR_FIELDS) - self.arrays.keys())
        if missing:
            raise ValueError(f"missing scalar fields: {missing}")
        count = len(self.arrays["request_id"])
        for name, value in self.arrays.items():
            if len(value) != count:
                raise ValueError(f"field {name!r} has inconsistent invocation count")
        if self.arrays["expert_counts_by_class"].shape[1:] != (len(POSITION_CLASSES), 256):
            raise ValueError("expert_counts_by_class must be [N,5,256]")
        top_k = int(self.metadata["top_k"])
        block = int(self.metadata["block_length"])
        if self.arrays["current_expert_ids"].shape[1:] != (block, top_k):
            raise ValueError("current_expert_ids has wrong shape")
        ids = self.arrays["current_expert_ids"]
        valid = ids >= 0
        if np.any(ids[valid] >= 256):
            raise ValueError("expert id out of range")

    def save(self, path: Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.arrays)
        payload["metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "DiscoveryTrace":
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files if name != "metadata_json"}
            metadata = json.loads(str(source["metadata_json"]))
        result = cls(arrays, metadata)
        result.validate()
        return result

    @classmethod
    def concatenate(cls, traces: list["DiscoveryTrace"]) -> "DiscoveryTrace":
        if not traces:
            raise ValueError("no traces")
        for trace in traces:
            trace.validate()
        comparable = ("model", "revision", "threshold", "block_length", "top_k", "hidden_size")
        reference = traces[0].metadata
        for trace in traces[1:]:
            if any(trace.metadata.get(key) != reference.get(key) for key in comparable):
                raise ValueError("incompatible discovery trace metadata")
        arrays = {
            name: np.concatenate([trace.arrays[name] for trace in traces], axis=0)
            for name in traces[0].arrays
        }
        metadata = dict(reference)
        metadata["requests"] = sorted({int(item) for item in arrays["request_id"]})
        metadata["merged_shards"] = len(traces)
        return cls(arrays, metadata)


def _records_to_trace(records: list[dict[str, Any]], metadata: dict[str, Any]) -> DiscoveryTrace:
    if not records:
        raise ValueError("no invocation records")
    arrays: dict[str, np.ndarray] = {}
    scalar_dtypes = {
        "request_id": np.int32, "block_id": np.int16, "iteration_id": np.int16,
        "nfe": np.int32, "layer_id": np.int16, "prompt_tokens": np.int32,
        "physical_rows": np.int32, "masked_current_block": np.int16,
        "decoded_current_block": np.int16, "newly_accepted_current_block": np.int16,
        "accepted_this_iteration": np.int16, "remaining_mask_after": np.int16,
        "normalized_block_progress": np.float32, "confidence_mean": np.float32,
        "confidence_min": np.float32, "confidence_max": np.float32, "terminated": bool,
    }
    for name, dtype in scalar_dtypes.items():
        arrays[name] = np.asarray([record[name] for record in records], dtype=dtype)
    vector_fields = sorted(set(records[0]) - set(SCALAR_FIELDS))
    for name in vector_fields:
        value = np.stack([np.asarray(record[name]) for record in records])
        arrays[name] = value
    full_metadata = dict(metadata)
    full_metadata.update(
        schema_version=DISCOVERY_SCHEMA_VERSION,
        position_classes=list(POSITION_CLASSES),
        aggregate_semantics="full physical rows aggregated; current block exact routes retained",
        physical_row_semantics="vanilla_full_rows",
        virtual_targets=[2, 4, 8],
    )
    result = DiscoveryTrace(arrays, full_metadata)
    result.validate()
    return result


class DiscoveryTraceCollector:
    """Low-storage official-HF observer for one request at a time."""

    def __init__(self, model, mask_id: int, block_length: int, metadata: dict[str, Any]):
        self.model = model
        self.mask_id = int(mask_id)
        self.block_length = int(block_length)
        self.metadata = dict(metadata)
        self.records: list[dict[str, Any]] = []
        self._handles = []
        self._original_forward = None
        self._original_sampler = None
        self._context = None
        self._last_context = None
        self._block_iterations: dict[int, int] = {}
        self._request_id = -1
        self._prompt_tokens = -1
        self._previous_ids: torch.Tensor | None = None

    def _position_classes(self, ids: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        length = len(ids)
        current_start = length - self.block_length
        classes = np.full(length, -1, dtype=np.int8)
        classes[: self._prompt_tokens] = 0
        classes[self._prompt_tokens : current_start] = 1
        current = ids[current_start:]
        current_positions = np.arange(current_start, length)
        generated_position = current_positions >= self._prompt_tokens
        masked = (current == self.mask_id) & generated_position
        newly = np.zeros(self.block_length, dtype=bool)
        if previous is not None and len(previous) == length:
            prior = previous[current_start:]
            newly = (prior == self.mask_id) & (current != self.mask_id) & generated_position
        current_classes = classes[current_start:]
        current_classes[masked] = 2
        current_classes[newly] = 3
        current_classes[generated_position & ~masked & ~newly] = 4
        if np.any(classes < 0):
            raise RuntimeError("position classification left unassigned rows")
        return classes

    def _finalize_previous(self, next_ids: torch.Tensor | None) -> None:
        context = self._last_context
        if context is None:
            return
        if next_ids is not None and next_ids.shape[1] == context["physical_rows"]:
            current = next_ids[0, -self.block_length :].detach().cpu().numpy()
            masked_after = int(np.count_nonzero(current == self.mask_id))
            terminated = False
        else:
            masked_after = 0
            terminated = next_ids is None
        accepted = context["masked_before"] - masked_after
        for index in context["record_indices"]:
            self.records[index]["accepted_this_iteration"] = accepted
            self.records[index]["remaining_mask_after"] = masked_after
            self.records[index]["confidence_mean"] = context["confidence_mean"]
            self.records[index]["confidence_min"] = context["confidence_min"]
            self.records[index]["confidence_max"] = context["confidence_max"]
            self.records[index]["terminated"] = terminated
        self._previous_ids = context["input_ids"].copy()
        self._last_context = None

    def _router_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            context = self._context
            if context is None:
                raise RuntimeError("router outside observed forward")
            topk_ids, topk_weights, _ = output
            ids = topk_ids.detach().reshape(-1, topk_ids.shape[-1]).cpu().numpy().astype(np.int16)
            weights = topk_weights.detach().reshape(-1, topk_weights.shape[-1]).float().cpu().numpy()
            physical = context["physical_rows"]
            if len(ids) != physical:
                raise RuntimeError("router rows differ from input rows")
            classes = context["classes"]
            counts = np.zeros((len(POSITION_CLASSES), 256), dtype=np.uint16)
            mass = np.zeros((len(POSITION_CLASSES), 256), dtype=np.float32)
            class_tokens = np.zeros(len(POSITION_CLASSES), dtype=np.uint16)
            for class_id in range(len(POSITION_CLASSES)):
                selected = classes == class_id
                class_tokens[class_id] = int(selected.sum())
                if selected.any():
                    counts[class_id] = np.bincount(ids[selected].reshape(-1), minlength=256)
                    np.add.at(mass[class_id], ids[selected].reshape(-1), weights[selected].reshape(-1))
            current_start = physical - self.block_length
            current_ids = ids[current_start:]
            current_weights = weights[current_start:].astype(np.float16)
            current_classes = classes[current_start:]
            record: dict[str, Any] = {
                "request_id": self._request_id,
                "block_id": context["block_id"],
                "iteration_id": context["iteration_id"],
                "nfe": context["nfe"],
                "layer_id": layer_id,
                "prompt_tokens": self._prompt_tokens,
                "physical_rows": physical,
                "masked_current_block": int(np.count_nonzero(current_classes == 2)),
                "decoded_current_block": int(np.count_nonzero(current_classes == 4)),
                "newly_accepted_current_block": int(np.count_nonzero(current_classes == 3)),
                "accepted_this_iteration": -1,
                "remaining_mask_after": -1,
                "normalized_block_progress": context["iteration_id"] / 31.0,
                "confidence_mean": np.nan,
                "confidence_min": np.nan,
                "confidence_max": np.nan,
                "terminated": False,
                "class_token_counts": class_tokens,
                "expert_counts_by_class": counts,
                "router_mass_by_class": mass.astype(np.float16),
                "current_expert_ids": current_ids,
                "current_router_weights": current_weights,
                "current_position_class": current_classes.astype(np.int8),
            }
            for ep in (2, 4, 8):
                record.update(_traffic_fields(ids, int(self.metadata["hidden_size"]), ep))
                record[f"current_source_rank_ep{ep}"] = _balanced_sources(physical, ep)[current_start:].astype(np.int8)
            context["record_indices"].append(len(self.records))
            self.records.append(record)
        return hook

    def start(self, request_id: int, prompt_tokens: int) -> None:
        if self._original_forward is not None:
            raise RuntimeError("collector already active")
        self._request_id = int(request_id)
        self._prompt_tokens = int(prompt_tokens)
        self._block_iterations.clear()
        self._last_context = None
        self._previous_ids = None
        self._original_forward = self.model.forward
        collector = self

        def observed_forward(model_self, input_ids, *args, **kwargs):
            collector._finalize_previous(input_ids.detach())
            if input_ids.shape[1] % collector.block_length:
                raise RuntimeError("official generation window is not block aligned")
            prefill_blocks = collector._prompt_tokens // collector.block_length
            block_id = input_ids.shape[1] // collector.block_length - 1 - prefill_blocks
            if block_id < 0:
                raise RuntimeError("official generation window ended inside the prompt")
            iteration_id = collector._block_iterations.get(block_id, 0)
            collector._block_iterations[block_id] = iteration_id + 1
            cpu_ids = input_ids.detach().cpu()[0].numpy()
            classes = collector._position_classes(cpu_ids, collector._previous_ids)
            context = {
                "block_id": block_id,
                "iteration_id": iteration_id,
                "nfe": sum(collector._block_iterations.values()) - 1,
                "physical_rows": int(input_ids.shape[1]),
                "input_ids": cpu_ids.copy(),
                "classes": classes,
                "masked_before": int(np.count_nonzero(classes == 2)),
                "confidence_mean": np.nan,
                "confidence_min": np.nan,
                "confidence_max": np.nan,
                "record_indices": [],
            }
            collector._context = context
            result = collector._original_forward(input_ids, *args, **kwargs)
            collector._context = None
            collector._last_context = context
            return result

        self.model.forward = MethodType(observed_forward, self.model)
        self._original_sampler = self.model._sample_with_temperature_topk_topp

        def observed_sampler(*args, **kwargs):
            result = collector._original_sampler(*args, **kwargs)
            context = collector._last_context
            if context is None:
                raise RuntimeError("sampler without forward")
            probabilities = result[1].detach().float().cpu().reshape(-1)
            active = context["classes"][-collector.block_length:] == 2
            current_probabilities = probabilities[-collector.block_length:][active]
            if current_probabilities.numel():
                context["confidence_mean"] = float(current_probabilities.mean())
                context["confidence_min"] = float(current_probabilities.min())
                context["confidence_max"] = float(current_probabilities.max())
            return result

        self.model._sample_with_temperature_topk_topp = observed_sampler
        for layer_id, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, "gate"):
                self._handles.append(layer.mlp.gate.register_forward_hook(self._router_hook(layer_id)))

    def stop(self) -> None:
        if self._original_forward is None:
            return
        self._finalize_previous(None)
        self.model.forward = self._original_forward
        self.model._sample_with_temperature_topk_topp = self._original_sampler
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._original_forward = None
        self._original_sampler = None
        self._context = None

    def trace(self) -> DiscoveryTrace:
        if self._original_forward is not None:
            raise RuntimeError("stop collector first")
        metadata = dict(self.metadata)
        metadata["requests"] = [self._request_id]
        return _records_to_trace(self.records, metadata)


def from_heavy_trace(
    trace: TraceBundle, prompt_tokens_by_request: dict[int, int] | None = None
) -> DiscoveryTrace:
    """Convert a v1 full-row trace into discovery v2 for Stage-0/audit reuse."""

    trace.validate()
    hidden = int(trace.metadata["hidden_size"])
    block_length = int(trace.metadata["block_length"])
    min_layer = int(np.min(trace.rows["layer_id"]))
    max_layer = int(np.max(trace.rows["layer_id"]))
    invocations = list(invocation_slices(trace.rows))
    iteration_lookup = {
        (int(trace.iterations["request_id"][i]), int(trace.iterations["block_id"][i]),
         int(trace.iterations["iteration_id"][i]), int(trace.iterations["nfe"][i])): i
        for i in range(len(trace.iterations["request_id"]))
    }
    request_lengths: dict[int, list[int]] = {}
    block_sets: dict[int, set[int]] = defaultdict(set)
    for key, selected in invocations:
        block_sets[key[0]].add(key[1])
        if key[4] == min_layer:
            request_lengths.setdefault(key[0], []).append(selected.stop - selected.start)
    aligned_prompt_floor = {
        request: min(lengths) - block_length for request, lengths in request_lengths.items()
    }
    prompt_tokens = dict(aligned_prompt_floor)
    if prompt_tokens_by_request is not None:
        for request in request_lengths:
            if request not in prompt_tokens_by_request:
                raise ValueError(f"missing exact prompt length for request {request}")
            value = int(prompt_tokens_by_request[request])
            if value // block_length * block_length != aligned_prompt_floor[request]:
                raise ValueError(f"prompt length is inconsistent with physical trace for {request}")
            prompt_tokens[request] = value
    block_maps: dict[int, dict[int, int]] = {}
    for request in request_lengths:
        original = sorted(block_sets[request])
        block_maps[request] = {block: index for index, block in enumerate(original)}
    previous_masks: dict[tuple[int, int], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for key, selected in invocations:
        request, original_block, iteration, nfe, layer = key
        positions = trace.rows["token_position"][selected]
        physical = len(positions)
        ids = trace.rows["expert_ids"][selected].astype(np.int16)
        weights = trace.rows["router_weights"][selected].astype(np.float32)
        is_masked = trace.rows["is_masked"][selected]
        prompt = prompt_tokens[request]
        current_start = physical - block_length
        classes = np.full(physical, 1, dtype=np.int8)
        classes[:prompt] = 0
        current_positions = np.arange(current_start, physical)
        generated_position = current_positions >= prompt
        current_mask = is_masked[current_start:] & generated_position
        prior = previous_masks.get((request, original_block))
        newly = ((prior & ~current_mask & generated_position) if prior is not None
                 else np.zeros(block_length, dtype=bool))
        classes[current_start:][current_mask] = 2
        classes[current_start:][newly] = 3
        classes[current_start:][generated_position & ~current_mask & ~newly] = 4
        if layer == max_layer:
            previous_masks[(request, original_block)] = current_mask.copy()
        counts = np.zeros((5, 256), dtype=np.uint16)
        mass = np.zeros((5, 256), dtype=np.float32)
        class_tokens = np.zeros(5, dtype=np.uint16)
        for class_id in range(5):
            mask = classes == class_id
            class_tokens[class_id] = int(mask.sum())
            if mask.any():
                counts[class_id] = np.bincount(ids[mask].reshape(-1), minlength=256)
                np.add.at(mass[class_id], ids[mask].reshape(-1), weights[mask].reshape(-1))
        lookup = iteration_lookup[(request, original_block, iteration, nfe)]
        record: dict[str, Any] = {
            "request_id": request, "block_id": block_maps[request][original_block],
            "iteration_id": iteration, "nfe": nfe, "layer_id": layer,
            "prompt_tokens": prompt, "physical_rows": physical,
            "masked_current_block": int(np.count_nonzero(classes[current_start:] == 2)),
            "decoded_current_block": int(np.count_nonzero(classes[current_start:] == 4)),
            "newly_accepted_current_block": int(np.count_nonzero(classes[current_start:] == 3)),
            "accepted_this_iteration": int(trace.iterations["accepted"][lookup]),
            "remaining_mask_after": int(trace.iterations["masked_after"][lookup]),
            "normalized_block_progress": iteration / 31.0,
            "confidence_mean": float(trace.iterations["confidence_mean"][lookup]),
            "confidence_min": float(trace.iterations["confidence_min"][lookup]),
            "confidence_max": float(trace.iterations["confidence_max"][lookup]),
            "terminated": bool(trace.iterations["terminated"][lookup]),
            "class_token_counts": class_tokens, "expert_counts_by_class": counts,
            "router_mass_by_class": mass.astype(np.float16),
            "current_expert_ids": ids[current_start:],
            "current_router_weights": weights[current_start:].astype(np.float16),
            "current_position_class": classes[current_start:],
        }
        for ep in (2, 4, 8):
            record.update(_traffic_fields(ids, hidden, ep))
            record[f"current_source_rank_ep{ep}"] = _balanced_sources(physical, ep)[current_start:].astype(np.int8)
        records.append(record)
    metadata = dict(trace.metadata)
    metadata["trace_source"] = "converted_full_row_v1"
    metadata["requests"] = sorted(request_lengths)
    return _records_to_trace(records, metadata)
