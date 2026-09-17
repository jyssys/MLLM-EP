"""Compact, versioned routing/refinement trace schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np


SCHEMA_VERSION = 1
METHOD_SCHEMA_VERSION = 2


def invocation_slices(rows: dict[str, np.ndarray]):
    """Yield contiguous invocation keys/slices without O(N*invocations) masks.

    The trace writer appends each router invocation atomically. Repeated,
    non-contiguous keys indicate a corrupt or externally reordered trace and
    are rejected rather than silently coalesced.
    """

    names = ("request_id", "block_id", "iteration_id", "nfe", "layer_id")
    keys = np.stack([rows[name] for name in names], axis=1)
    if not len(keys):
        return
    boundaries = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(keys)]))
    seen = set()
    for start, end in zip(starts, ends):
        key = tuple(int(value) for value in keys[start])
        if key in seen:
            raise ValueError(f"non-contiguous repeated invocation key: {key}")
        seen.add(key)
        yield key, slice(int(start), int(end))


ROW_FIELDS = (
    "request_id",
    "block_id",
    "iteration_id",
    "nfe",
    "layer_id",
    "token_position",
    "source_partition_key",
    "is_masked",
    "expert_ids",
    "router_weights",
)

METHOD_ROW_FIELDS = (
    "semantic_state",
    "execution_state",
    "method",
    "is_fresh_route",
    "is_reused_route",
    "reuse_source_refinement",
    "freeze_age",
)

ITERATION_FIELDS = (
    "request_id",
    "block_id",
    "iteration_id",
    "nfe",
    "masked_before",
    "accepted",
    "masked_after",
    "confidence_mean",
    "confidence_min",
    "confidence_max",
    "terminated",
)


@dataclass(frozen=True)
class TraceBundle:
    """Structural trace plus provenance.

    One row entry represents one physical token row at one routed-MoE layer
    and refinement iteration.  ``expert_ids`` and ``router_weights`` have
    shape ``[num_rows, top_k]``.  The schema records vanilla physical rows;
    it does not infer fresh-row compaction from mask counts.
    """

    rows: dict[str, np.ndarray]
    iterations: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def validate(self) -> None:
        missing_rows = sorted(set(ROW_FIELDS) - self.rows.keys())
        missing_iterations = sorted(set(ITERATION_FIELDS) - self.iterations.keys())
        if missing_rows or missing_iterations:
            raise ValueError(
                f"trace fields missing: rows={missing_rows}, iterations={missing_iterations}"
            )
        row_count = len(self.rows["request_id"])
        for key in ROW_FIELDS:
            if len(self.rows[key]) != row_count:
                raise ValueError(f"row field {key!r} has inconsistent length")
        iteration_count = len(self.iterations["request_id"])
        for key in ITERATION_FIELDS:
            if len(self.iterations[key]) != iteration_count:
                raise ValueError(f"iteration field {key!r} has inconsistent length")
        expert_ids = np.asarray(self.rows["expert_ids"])
        weights = np.asarray(self.rows["router_weights"])
        if expert_ids.ndim != 2 or weights.shape != expert_ids.shape:
            raise ValueError("expert_ids/router_weights must be matching [N, top_k] arrays")
        num_experts = int(self.metadata["num_routed_experts"])
        if expert_ids.size and (expert_ids.min() < -1 or expert_ids.max() >= num_experts):
            raise ValueError("expert id is outside configured routed-expert range")
        if np.any((expert_ids == -1) & (weights != 0)):
            raise ValueError("padded variable-k branches must have zero router weight")
        schema_version = int(self.metadata.get("schema_version", -1))
        if schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported trace schema version")
        if int(self.metadata.get("method_schema_version", 0)) >= METHOD_SCHEMA_VERSION:
            missing_method = sorted(set(METHOD_ROW_FIELDS) - self.rows.keys())
            if missing_method:
                raise ValueError(f"method trace fields missing: {missing_method}")
        if self.metadata.get("physical_row_semantics") != "vanilla_full_rows":
            raise ValueError("baseline simulator requires vanilla_full_rows semantics")

    @property
    def top_k(self) -> int:
        return int(np.asarray(self.rows["expert_ids"]).shape[1])

    def save(self, path: Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            f"row__{key}": np.asarray(value) for key, value in self.rows.items()
        }
        payload.update(
            {f"iteration__{key}": np.asarray(value) for key, value in self.iterations.items()}
        )
        payload["metadata_json"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True), dtype=np.str_
        )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "TraceBundle":
        with np.load(path, allow_pickle=False) as data:
            rows = {
                key.removeprefix("row__"): data[key]
                for key in data.files
                if key.startswith("row__")
            }
            iterations = {
                key.removeprefix("iteration__"): data[key]
                for key in data.files
                if key.startswith("iteration__")
            }
            metadata = json.loads(str(data["metadata_json"]))
        result = cls(rows=rows, iterations=iterations, metadata=metadata)
        result.validate()
        return result

    def select_requests(self, request_ids: set[int]) -> "TraceBundle":
        row_mask = np.isin(self.rows["request_id"], list(request_ids))
        iteration_mask = np.isin(self.iterations["request_id"], list(request_ids))
        return TraceBundle(
            rows={key: value[row_mask] for key, value in self.rows.items()},
            iterations={key: value[iteration_mask] for key, value in self.iterations.items()},
            metadata=dict(self.metadata),
        )
