"""Build logical assignment and deduplicated activation traffic matrices."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mapping import ExpertOwnership, SourcePartition


@dataclass(frozen=True)
class TrafficSummary:
    assignment_matrix: np.ndarray
    unique_activation_matrix: np.ndarray
    rank_expert_assignments: np.ndarray
    expert_assignments: np.ndarray
    token_fanout: np.ndarray
    logical_dispatch_bytes: int
    logical_combine_bytes: int

    @property
    def remote_assignments(self) -> int:
        matrix = self.assignment_matrix
        return int(matrix.sum() - np.trace(matrix))

    @property
    def remote_unique_activations(self) -> int:
        matrix = self.unique_activation_matrix
        return int(matrix.sum() - np.trace(matrix))


def build_traffic(
    expert_ids: np.ndarray,
    ordered_row_keys: np.ndarray,
    ownership: ExpertOwnership,
    source_partition: SourcePartition,
    hidden_size: int,
    element_bytes: int = 2,
) -> TrafficSummary:
    """Build traffic for one layer/refinement invocation.

    DeepEP can send one hidden activation per source-token/destination-rank
    even when multiple selected experts share that destination.  Therefore
    assignment counts ``A`` and unique activation counts ``U`` are retained
    separately.
    """

    expert_ids = np.asarray(expert_ids, dtype=np.int64)
    if expert_ids.ndim != 2:
        raise ValueError("expert_ids must have shape [physical_rows, top_k]")
    if len(ordered_row_keys) != expert_ids.shape[0]:
        raise ValueError("source keys and route rows differ")
    if ownership.ep_size != source_partition.ep_size:
        raise ValueError("ownership/source EP sizes differ")
    ep = ownership.ep_size
    sources = source_partition.ranks(ordered_row_keys)
    owners = ownership.owner(expert_ids)
    assignment = np.zeros((ep, ep), dtype=np.int64)
    unique = np.zeros((ep, ep), dtype=np.int64)
    expert_hist = np.bincount(
        expert_ids.reshape(-1), minlength=ownership.num_experts
    ).astype(np.int64)
    np.add.at(
        assignment,
        (np.repeat(sources, expert_ids.shape[1]), owners.reshape(-1)),
        1,
    )
    for destination in range(ep):
        tokens = np.any(owners == destination, axis=1)
        unique[:, destination] = np.bincount(
            sources[tokens], minlength=ep
        )
    if expert_ids.shape[0]:
        sorted_owners = np.sort(owners, axis=1)
        fanout = (
            1 + np.count_nonzero(sorted_owners[:, 1:] != sorted_owners[:, :-1], axis=1)
        ).astype(np.int16)
    else:
        fanout = np.empty(0, dtype=np.int16)
    rank_expert = np.bincount(
        owners.reshape(-1), minlength=ep
    ).astype(np.int64)
    remote_unique = int(unique.sum() - np.trace(unique))
    payload = remote_unique * hidden_size * element_bytes
    return TrafficSummary(
        assignment_matrix=assignment,
        unique_activation_matrix=unique,
        rank_expert_assignments=rank_expert,
        expert_assignments=expert_hist,
        token_fanout=fanout,
        logical_dispatch_bytes=payload,
        logical_combine_bytes=payload,
    )
