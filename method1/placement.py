"""Method 1: modality-balanced expert placement with LPT greedy."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlacementResult:
    expert_to_gpu: torch.Tensor
    gpu_loads: torch.Tensor
    expert_counts: torch.Tensor


def compute_loads(weights: torch.Tensor, expert_to_gpu: torch.Tensor, num_gpus: int) -> torch.Tensor:
    """Compute per-GPU load from expert weights and an expert->GPU map."""

    weights = weights.to(dtype=torch.float32)
    expert_to_gpu = expert_to_gpu.to(dtype=torch.long)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D tensor")
    if expert_to_gpu.shape != weights.shape:
        raise ValueError("expert_to_gpu must have the same shape as weights")
    if expert_to_gpu.min() < 0 or expert_to_gpu.max() >= num_gpus:
        raise ValueError("expert_to_gpu contains GPU ids outside [0, num_gpus)")
    loads = torch.zeros(num_gpus, dtype=torch.float32)
    loads.scatter_add_(0, expert_to_gpu.cpu(), weights.cpu())
    return loads


def contiguous_placement(num_experts: int, num_gpus: int) -> torch.Tensor:
    """Vanilla contiguous placement used as a simple baseline in tests."""

    if num_experts <= 0 or num_gpus <= 0:
        raise ValueError("num_experts and num_gpus must be positive")
    per_gpu = (num_experts + num_gpus - 1) // num_gpus
    mapping = torch.arange(num_experts, dtype=torch.long) // per_gpu
    return mapping.clamp_max(num_gpus - 1)


def round_robin_placement(num_experts: int, num_gpus: int) -> torch.Tensor:
    """Round-robin baseline for comparison."""

    if num_experts <= 0 or num_gpus <= 0:
        raise ValueError("num_experts and num_gpus must be positive")
    return torch.arange(num_experts, dtype=torch.long) % num_gpus


def lpt_placement(
    weights: torch.Tensor,
    num_gpus: int = 8,
    max_experts_per_gpu: int | None = None,
) -> PlacementResult:
    """Assign experts to GPUs with Longest Processing Time greedy.

    Experts are sorted by descending weight and placed on the least-loaded GPU
    that still has capacity.
    """

    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    weights = weights.detach().cpu().to(dtype=torch.float32)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D tensor")
    num_experts = weights.numel()
    if max_experts_per_gpu is not None and max_experts_per_gpu * num_gpus < num_experts:
        raise ValueError("max_experts_per_gpu capacity is smaller than num_experts")

    loads = torch.zeros(num_gpus, dtype=torch.float32)
    counts = torch.zeros(num_gpus, dtype=torch.long)
    mapping = torch.full((num_experts,), -1, dtype=torch.long)
    sorted_experts = sorted(range(num_experts), key=lambda idx: (-float(weights[idx]), idx))

    for expert_idx in sorted_experts:
        candidates = [
            gpu
            for gpu in range(num_gpus)
            if max_experts_per_gpu is None or counts[gpu].item() < max_experts_per_gpu
        ]
        if not candidates:
            raise RuntimeError("no GPU has remaining expert capacity")
        gpu = min(candidates, key=lambda g: (float(loads[g]), int(counts[g]), g))
        mapping[expert_idx] = gpu
        loads[gpu] += weights[expert_idx]
        counts[gpu] += 1

    return PlacementResult(expert_to_gpu=mapping, gpu_loads=loads, expert_counts=counts)


def brute_force_optimal_max_load(
    weights: torch.Tensor,
    num_gpus: int,
    max_experts_per_gpu: int | None = None,
) -> tuple[float, torch.Tensor]:
    """Return the optimal max-load for small test cases by exhaustive search."""

    weights = weights.detach().cpu().to(dtype=torch.float32)
    if weights.numel() > 12:
        raise ValueError("brute_force_optimal_max_load is intended for small sanity tests")

    best_load = float("inf")
    best_mapping: torch.Tensor | None = None
    for assignment in itertools.product(range(num_gpus), repeat=weights.numel()):
        mapping = torch.tensor(assignment, dtype=torch.long)
        if max_experts_per_gpu is not None:
            counts = torch.bincount(mapping, minlength=num_gpus)
            if torch.any(counts > max_experts_per_gpu):
                continue
        loads = compute_loads(weights, mapping, num_gpus)
        max_load = float(loads.max())
        if max_load < best_load:
            best_load = max_load
            best_mapping = mapping
    if best_mapping is None:
        raise RuntimeError("no feasible assignment found")
    return best_load, best_mapping

