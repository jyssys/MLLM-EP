"""Build synthetic batch scaling from one real captured request."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RepeatedWorkload:
    batch_equivalent: int
    hidden: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    source_token_indices: torch.Tensor
    request_indices: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def assignment_count(self) -> int:
        return int(self.topk_ids.numel())


def build_repeated_workload(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    batch_equivalent: int,
) -> RepeatedWorkload:
    if batch_equivalent <= 0:
        raise ValueError("batch_equivalent must be positive")
    if hidden.shape[0] != topk_ids.shape[0] or topk_ids.shape != topk_weights.shape:
        raise ValueError("hidden/routes have inconsistent token dimensions")
    tokens = int(hidden.shape[0])
    return RepeatedWorkload(
        batch_equivalent=batch_equivalent,
        hidden=hidden.repeat((batch_equivalent, 1)).contiguous(),
        topk_ids=topk_ids.repeat((batch_equivalent, 1)).contiguous(),
        topk_weights=topk_weights.repeat((batch_equivalent, 1)).contiguous(),
        source_token_indices=torch.arange(tokens, device=hidden.device).repeat(
            batch_equivalent
        ),
        request_indices=torch.arange(batch_equivalent, device=hidden.device).repeat_interleave(
            tokens
        ),
    )


def rank_slice(total_tokens: int, ep_size: int, rank: int) -> slice:
    if total_tokens % ep_size:
        raise ValueError(
            f"equal NCCL replay requires {total_tokens} tokens divisible by EP {ep_size}"
        )
    tokens_per_rank = total_tokens // ep_size
    return slice(rank * tokens_per_rank, (rank + 1) * tokens_per_rank)


def workload_metrics(
    workload: RepeatedWorkload,
    *,
    vision_tokens_per_request: int,
    ep_size: int,
    local_experts_per_rank: int,
) -> dict[str, object]:
    ids = workload.topk_ids.to(torch.int64)
    destination = torch.div(ids, local_experts_per_rank, rounding_mode="floor")
    rank_assignments = [
        int((destination == rank).sum().item()) for rank in range(ep_size)
    ]
    local_counts: list[list[int]] = []
    for rank in range(ep_size):
        start = rank * local_experts_per_rank
        counts = torch.bincount(
            ids[(ids >= start) & (ids < start + local_experts_per_rank)] - start,
            minlength=local_experts_per_rank,
        )
        local_counts.append([int(value) for value in counts.tolist()])
    return {
        "scaling_source": "synthetic batch scaling from real captured request",
        "batch_equivalent": workload.batch_equivalent,
        "real_tokens": workload.token_count,
        "vision_tokens": vision_tokens_per_request * workload.batch_equivalent,
        "total_routed_assignments": workload.assignment_count,
        "rank_routed_assignments": rank_assignments,
        "critical_rank_assignments": max(rank_assignments),
        "critical_assignment_rank": rank_assignments.index(max(rank_assignments)),
        "max_local_expert_token_count": max(max(row) for row in local_counts),
        "active_local_experts_per_rank": [sum(value > 0 for value in row) for row in local_counts],
    }
