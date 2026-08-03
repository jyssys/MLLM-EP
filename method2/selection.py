"""Load-targeted redundant-token selection."""

from __future__ import annotations

import math

import torch


def identify_straggler_gpus(loads: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    """Return GPU ids whose load is greater than ``threshold``."""

    loads = loads.detach().cpu().to(dtype=torch.float32)
    threshold_t = torch.as_tensor(threshold, dtype=torch.float32)
    return torch.nonzero(loads > threshold_t, as_tuple=False).flatten()


def tokens_routed_to_gpus(
    expert_assignment: torch.Tensor,
    expert_to_gpu: torch.Tensor,
    *,
    any_topk: bool = True,
) -> torch.Tensor:
    """Map token top-k expert assignments to GPU ids.

    If ``any_topk`` is true, the returned tensor has shape ``[N, K]``. If false,
    only the primary route is returned as ``[N]``.
    """

    if expert_assignment.ndim != 2:
        raise ValueError("expert_assignment must have shape [num_tokens, top_k]")
    expert_to_gpu = expert_to_gpu.to(device=expert_assignment.device, dtype=torch.long)
    routed = expert_to_gpu[expert_assignment.to(torch.long)]
    return routed if any_topk else routed[:, 0]


def select_merge_candidates(
    loads: torch.Tensor,
    expert_assignment: torch.Tensor,
    importance: torch.Tensor,
    *,
    threshold: float | torch.Tensor,
    rho: float = 0.3,
    expert_to_gpu: torch.Tensor | None = None,
    token_gpu: torch.Tensor | None = None,
    straggler_gpus: torch.Tensor | None = None,
    any_topk: bool = True,
) -> dict[str, torch.Tensor]:
    """Select low-importance vision tokens that route to straggler GPUs."""

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if importance.ndim != 1:
        raise ValueError("importance must be a 1D tensor")
    n_tokens = importance.numel()
    if expert_assignment.shape[0] != n_tokens:
        raise ValueError("expert_assignment and importance must have the same token count")

    if straggler_gpus is None:
        straggler_gpus = identify_straggler_gpus(loads, threshold)
    else:
        straggler_gpus = straggler_gpus.detach().cpu().to(dtype=torch.long)
    straggler_gpus_dev = straggler_gpus.to(device=importance.device)

    if token_gpu is None:
        if expert_to_gpu is None:
            raise ValueError("either expert_to_gpu or token_gpu must be provided")
        routed_gpu = tokens_routed_to_gpus(expert_assignment, expert_to_gpu.to(importance.device), any_topk=any_topk)
    else:
        routed_gpu = token_gpu.to(device=importance.device, dtype=torch.long)

    if routed_gpu.ndim == 2:
        straggler_token_mask = torch.isin(routed_gpu, straggler_gpus_dev).any(dim=1)
    else:
        straggler_token_mask = torch.isin(routed_gpu, straggler_gpus_dev)

    straggler_indices = torch.nonzero(straggler_token_mask, as_tuple=False).flatten()
    candidate_mask = torch.zeros(n_tokens, dtype=torch.bool, device=importance.device)
    if straggler_indices.numel() == 0 or rho == 0.0:
        return {
            "candidate_mask": candidate_mask.cpu(),
            "candidate_indices": torch.empty(0, dtype=torch.long),
            "straggler_gpus": straggler_gpus.cpu(),
            "straggler_token_mask": straggler_token_mask.cpu(),
            "cutoff_value": torch.tensor(float("nan")),
        }

    # Use floor for the lower-rho set so a high-importance tail remains
    # protected in small dummy cases. Keep one candidate when rho > 0.
    k = min(straggler_indices.numel(), max(1, math.floor(float(rho) * straggler_indices.numel())))
    local_importance = importance[straggler_indices]
    low_local = torch.topk(local_importance, k=k, largest=False, sorted=True).indices
    selected = straggler_indices[low_local]
    candidate_mask[selected] = True
    cutoff_value = importance[selected].max() if selected.numel() > 0 else torch.tensor(float("nan"))

    return {
        "candidate_mask": candidate_mask.cpu(),
        "candidate_indices": selected.detach().cpu(),
        "straggler_gpus": straggler_gpus.cpu(),
        "straggler_token_mask": straggler_token_mask.detach().cpu(),
        "cutoff_value": cutoff_value.detach().cpu(),
    }
