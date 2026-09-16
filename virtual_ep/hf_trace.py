"""Semantics-preserving heavy trace hooks for official HF LLaDA2 generation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any
import numpy as np
import torch

from .schema import ITERATION_FIELDS, ROW_FIELDS, SCHEMA_VERSION, TraceBundle


@dataclass
class _Context:
    request_id: int
    block_id: int
    iteration_id: int
    nfe: int
    input_ids: torch.Tensor
    masked_before: int
    confidence_mean: float = float("nan")
    confidence_min: float = float("nan")
    confidence_max: float = float("nan")


class HFTraceCollector:
    """Observe the official model without changing decoder decisions.

    The collector wraps ``forward`` and the sampling helper only to observe
    inputs/outputs, and attaches hooks to every router.  A parity run with the
    same model, prompt, and RNG seed is required before accepting an artifact.
    """

    def __init__(self, model, mask_id: int, block_length: int, metadata: dict[str, Any]):
        self.model = model
        self.mask_id = int(mask_id)
        self.block_length = int(block_length)
        self.metadata = dict(metadata)
        self.metadata.update(
            schema_version=SCHEMA_VERSION,
            physical_row_semantics="vanilla_full_rows",
            trace_source="official_hf_hooks",
            source_partition_key="flattened_batch_sequence_row",
        )
        self._row_lists = {name: [] for name in ROW_FIELDS}
        self._iterations: list[dict[str, Any]] = []
        self._context: _Context | None = None
        self._last_context: _Context | None = None
        self._block_iterations: dict[int, int] = {}
        self._request_id = -1
        self._handles = []
        self._original_forward = None
        self._original_sampler = None

    def _finalize_previous(self, next_ids: torch.Tensor | None) -> None:
        previous = self._last_context
        if previous is None:
            return
        if next_ids is None:
            masked_after = 0
            terminated = True
        else:
            next_block = next_ids.shape[1] // self.block_length - 1
            if next_block == previous.block_id:
                masked_after = int((next_ids[:, -self.block_length :] == self.mask_id).sum())
            else:
                masked_after = 0
            terminated = False
        self._iterations.append(
            {
                "request_id": previous.request_id,
                "block_id": previous.block_id,
                "iteration_id": previous.iteration_id,
                "nfe": previous.nfe,
                "masked_before": previous.masked_before,
                "accepted": previous.masked_before - masked_after,
                "masked_after": masked_after,
                "confidence_mean": previous.confidence_mean,
                "confidence_min": previous.confidence_min,
                "confidence_max": previous.confidence_max,
                "terminated": terminated,
            }
        )
        self._last_context = None

    def _router_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if self._context is None:
                raise RuntimeError("router executed outside traced model forward")
            topk_ids, topk_weights, _router_logits = output
            context = self._context
            ids = topk_ids.detach().reshape(-1, topk_ids.shape[-1]).cpu().numpy()
            weights = (
                topk_weights.detach().reshape(-1, topk_weights.shape[-1]).float().cpu().numpy()
            )
            flat_input = context.input_ids.reshape(-1).cpu().numpy()
            row_count = len(flat_input)
            if ids.shape[0] != row_count:
                raise RuntimeError("router row count differs from physical input rows")
            self._row_lists["request_id"].append(
                np.full(row_count, context.request_id, dtype=np.int32)
            )
            self._row_lists["block_id"].append(
                np.full(row_count, context.block_id, dtype=np.int16)
            )
            self._row_lists["iteration_id"].append(
                np.full(row_count, context.iteration_id, dtype=np.int16)
            )
            self._row_lists["nfe"].append(
                np.full(row_count, context.nfe, dtype=np.int32)
            )
            self._row_lists["layer_id"].append(
                np.full(row_count, layer_id, dtype=np.int16)
            )
            positions = np.tile(np.arange(context.input_ids.shape[1]), context.input_ids.shape[0])
            self._row_lists["token_position"].append(positions.astype(np.int32, copy=False))
            self._row_lists["source_partition_key"].append(
                np.arange(row_count, dtype=np.int32)
            )
            self._row_lists["is_masked"].append(flat_input == self.mask_id)
            self._row_lists["expert_ids"].append(ids.astype(np.int16, copy=False))
            self._row_lists["router_weights"].append(weights.astype(np.float16, copy=False))

        return hook

    def start(self, request_id: int) -> None:
        if self._original_forward is not None:
            raise RuntimeError("collector is already installed")
        self._request_id = int(request_id)
        self._block_iterations.clear()
        self._last_context = None
        self._original_forward = self.model.forward
        collector = self

        def observed_forward(model_self, input_ids, *args, **kwargs):
            collector._finalize_previous(input_ids.detach())
            block_id = input_ids.shape[1] // collector.block_length - 1
            iteration_id = collector._block_iterations.get(block_id, 0)
            collector._block_iterations[block_id] = iteration_id + 1
            active = input_ids[:, -collector.block_length :] == collector.mask_id
            context = _Context(
                request_id=collector._request_id,
                block_id=block_id,
                iteration_id=iteration_id,
                nfe=sum(collector._block_iterations.values()) - 1,
                input_ids=input_ids.detach().cpu(),
                masked_before=int(active.sum()),
            )
            collector._context = context
            result = collector._original_forward(input_ids, *args, **kwargs)
            collector._context = None
            collector._last_context = context
            return result

        self.model.forward = MethodType(observed_forward, self.model)
        self._original_sampler = self.model._sample_with_temperature_topk_topp

        def observed_sampler(*args, **kwargs):
            result = collector._original_sampler(*args, **kwargs)
            if collector._last_context is None:
                raise RuntimeError("sampling executed without a completed forward")
            probabilities = result[1].detach().float()
            active = collector._last_context.input_ids[:, -collector.block_length :] == collector.mask_id
            active_probabilities = probabilities.cpu()[active]
            if active_probabilities.numel():
                collector._last_context.confidence_mean = float(active_probabilities.mean())
                collector._last_context.confidence_min = float(active_probabilities.min())
                collector._last_context.confidence_max = float(active_probabilities.max())
            return result

        self.model._sample_with_temperature_topk_topp = observed_sampler
        for layer_id, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, "gate"):
                self._handles.append(
                    layer.mlp.gate.register_forward_hook(self._router_hook(layer_id))
                )

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

    def bundle(self) -> TraceBundle:
        if self._original_forward is not None:
            raise RuntimeError("stop the collector before materializing the trace")
        dtypes = {
            "request_id": np.int32,
            "block_id": np.int16,
            "iteration_id": np.int16,
            "nfe": np.int32,
            "layer_id": np.int16,
            "token_position": np.int32,
            "source_partition_key": np.int32,
            "is_masked": bool,
            "expert_ids": np.int16,
            "router_weights": np.float16,
        }
        rows = {
            name: (
                np.concatenate(self._row_lists[name], axis=0).astype(dtype, copy=False)
                if self._row_lists[name]
                else np.empty((0, int(self.metadata["top_k"])), dtype=dtype)
                if name in ("expert_ids", "router_weights")
                else np.empty(0, dtype=dtype)
            )
            for name, dtype in dtypes.items()
        }
        iterations = {
            name: np.asarray(
                [row[name] for row in self._iterations],
                dtype=(bool if name == "terminated" else np.float32 if name.startswith("confidence") else np.int32),
            )
            for name in ITERATION_FIELDS
        }
        result = TraceBundle(rows=rows, iterations=iterations, metadata=self.metadata)
        result.validate()
        return result
