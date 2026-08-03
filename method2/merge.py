"""Expert-aware redundant-token merge.

The key Phase 1 invariant is routing preservation: merging happens per expert
contribution, not by deleting whole tokens from all of their top-k routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MergeCluster:
    expert: int
    token_indices: torch.Tensor
    weights: torch.Tensor
    representative: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.token_indices.numel())


@dataclass(frozen=True)
class MergeResult:
    clusters_by_expert: dict[int, list[MergeCluster]]
    candidate_load_before: dict[int, int]
    candidate_load_after: dict[int, int]


def _candidate_indices(candidate_mask_or_indices: torch.Tensor, n_tokens: int, device: torch.device) -> torch.Tensor:
    x = candidate_mask_or_indices.to(device)
    if x.dtype == torch.bool:
        if x.shape != (n_tokens,):
            raise ValueError(f"candidate mask must have shape ({n_tokens},)")
        return torch.nonzero(x, as_tuple=False).flatten()
    return x.to(dtype=torch.long).flatten()


def _greedy_hidden_clusters(
    token_indices: torch.Tensor,
    hidden_states: torch.Tensor,
    similarity_threshold: float,
) -> list[torch.Tensor]:
    if token_indices.numel() == 0:
        return []
    if similarity_threshold <= -1.0:
        return [token_indices]

    normalized = F.normalize(hidden_states[token_indices], dim=-1)
    clusters: list[list[int]] = []
    centroids: list[torch.Tensor] = []

    for local_idx, token_idx in enumerate(token_indices.tolist()):
        vector = normalized[local_idx]
        best_cluster = None
        best_score = -float("inf")
        for cluster_idx, centroid in enumerate(centroids):
            score = float(torch.dot(vector, centroid))
            if score >= similarity_threshold and score > best_score:
                best_cluster = cluster_idx
                best_score = score
        if best_cluster is None:
            clusters.append([token_idx])
            centroids.append(vector.clone())
        else:
            clusters[best_cluster].append(token_idx)
            cluster_tensor = torch.tensor(clusters[best_cluster], dtype=torch.long, device=hidden_states.device)
            centroids[best_cluster] = F.normalize(hidden_states[cluster_tensor].mean(dim=0), dim=0)

    return [torch.tensor(cluster, dtype=torch.long, device=hidden_states.device) for cluster in clusters]


def _importance_weights(importance: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    raw = importance[token_indices].to(dtype=torch.float32)
    total = raw.sum()
    if total <= 0:
        return torch.full_like(raw, 1.0 / max(1, raw.numel()))
    return raw / total


def expert_aware_merge(
    candidate_mask_or_indices: torch.Tensor,
    expert_assignment: torch.Tensor,
    hidden_states: torch.Tensor,
    importance: torch.Tensor,
    *,
    target_experts: Iterable[int] | torch.Tensor | None = None,
    similarity_threshold: float = 0.9,
    similarity: str = "hidden_cosine",
) -> MergeResult:
    """Cluster and merge candidate token contributions independently per expert."""

    if similarity != "hidden_cosine":
        # TODO(Phase2): add routing-distribution KL or other similarity metrics.
        raise NotImplementedError("Only hidden_cosine clustering is implemented in Phase 1")
    if expert_assignment.ndim != 2:
        raise ValueError("expert_assignment must have shape [num_tokens, top_k]")
    n_tokens = expert_assignment.shape[0]
    if hidden_states.ndim != 2 or hidden_states.shape[0] != n_tokens:
        raise ValueError("hidden_states must have shape [num_tokens, hidden_dim]")
    if importance.shape != (n_tokens,):
        raise ValueError("importance must have shape [num_tokens]")

    device = hidden_states.device
    expert_assignment = expert_assignment.to(device=device, dtype=torch.long)
    importance = importance.to(device=device)
    candidates = _candidate_indices(candidate_mask_or_indices, n_tokens, device)

    if target_experts is None:
        if candidates.numel() == 0:
            target_experts_t = torch.empty(0, dtype=torch.long, device=device)
        else:
            target_experts_t = torch.unique(expert_assignment[candidates].reshape(-1))
    else:
        target_experts_t = torch.as_tensor(list(target_experts), dtype=torch.long, device=device)

    clusters_by_expert: dict[int, list[MergeCluster]] = {}
    before: dict[int, int] = {}
    after: dict[int, int] = {}

    for expert in target_experts_t.tolist():
        routed_to_expert = (expert_assignment[candidates] == expert).any(dim=1)
        cand_e = candidates[routed_to_expert]
        before[expert] = int(cand_e.numel())
        clusters = _greedy_hidden_clusters(cand_e, hidden_states, similarity_threshold)
        after[expert] = len(clusters)
        expert_clusters: list[MergeCluster] = []
        for token_indices in clusters:
            weights = _importance_weights(importance, token_indices)
            representative = torch.sum(hidden_states[token_indices] * weights[:, None].to(hidden_states.dtype), dim=0)
            expert_clusters.append(
                MergeCluster(
                    expert=expert,
                    token_indices=token_indices.detach().cpu(),
                    weights=weights.detach().cpu(),
                    representative=representative.detach().cpu(),
                )
            )
        clusters_by_expert[expert] = expert_clusters

    return MergeResult(
        clusters_by_expert=clusters_by_expert,
        candidate_load_before=before,
        candidate_load_after=after,
    )


def simulate_identity_expert_combine(
    hidden_states: torch.Tensor,
    expert_assignment: torch.Tensor,
    merge_result: MergeResult | None = None,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simulate expert outputs using identity experts and optional merge clusters.

    This is a Phase 1-only helper for pipeline tests. A token's contribution to
    expert ``e`` is replaced by the expert-specific cluster representative only
    for that route; all other routes remain untouched.
    """

    if expert_assignment.ndim != 2:
        raise ValueError("expert_assignment must have shape [num_tokens, top_k]")
    n_tokens, top_k = expert_assignment.shape
    if hidden_states.shape[0] != n_tokens:
        raise ValueError("hidden_states and expert_assignment token counts differ")
    if routing_weights is None:
        routing_weights = torch.full((n_tokens, top_k), 1.0 / top_k, dtype=hidden_states.dtype, device=hidden_states.device)
    else:
        routing_weights = routing_weights.to(device=hidden_states.device, dtype=hidden_states.dtype)

    replacement: dict[tuple[int, int], torch.Tensor] = {}
    if merge_result is not None:
        for expert, clusters in merge_result.clusters_by_expert.items():
            for cluster in clusters:
                if cluster.size <= 1:
                    continue
                rep = cluster.representative.to(device=hidden_states.device, dtype=hidden_states.dtype)
                for token_idx in cluster.token_indices.tolist():
                    replacement[(token_idx, expert)] = rep

    out = torch.zeros_like(hidden_states)
    expert_assignment = expert_assignment.to(device=hidden_states.device)
    for pos in range(top_k):
        experts = expert_assignment[:, pos].tolist()
        contrib = hidden_states.clone()
        for token_idx, expert in enumerate(experts):
            rep = replacement.get((token_idx, int(expert)))
            if rep is not None:
                contrib[token_idx] = rep
        out += contrib * routing_weights[:, pos, None]
    return out

