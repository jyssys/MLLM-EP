"""Cap logic for stopping merges at the second-place load."""

from __future__ import annotations

from collections.abc import Callable

import torch


def second_largest_other_load(loads: torch.Tensor, straggler_gpu: int) -> float:
    """Return max load among GPUs other than ``straggler_gpu``."""

    loads = loads.detach().cpu().to(dtype=torch.float32)
    if not 0 <= straggler_gpu < loads.numel():
        raise ValueError("straggler_gpu is out of range")
    others = torch.cat([loads[:straggler_gpu], loads[straggler_gpu + 1 :]])
    if others.numel() == 0:
        return 0.0
    return float(others.max())


def cap_merge_count(
    loads: torch.Tensor,
    straggler_gpu: int,
    candidate_count: int,
    max_merge_ratio: float,
    *,
    reduction_per_merge: float = 1.0,
) -> dict[str, float | int]:
    """Compute how many candidate merges to apply before hitting second place."""

    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")
    if not 0.0 <= max_merge_ratio <= 1.0:
        raise ValueError("max_merge_ratio must be in [0, 1]")
    if reduction_per_merge <= 0:
        raise ValueError("reduction_per_merge must be positive")

    loads_f = loads.detach().cpu().to(dtype=torch.float32)
    straggler_load = float(loads_f[straggler_gpu])
    second_load = second_largest_other_load(loads_f, straggler_gpu)
    max_merges = int(torch.ceil(torch.tensor(candidate_count * max_merge_ratio)).item())
    needed = max(0.0, straggler_load - second_load)
    needed_merges = int(torch.ceil(torch.tensor(needed / reduction_per_merge)).item())
    actual = min(candidate_count, max_merges, needed_merges)
    capped_load = straggler_load - actual * reduction_per_merge

    return {
        "actual_merge_count": actual,
        "candidate_count": candidate_count,
        "max_merge_count": max_merges,
        "actual_merge_ratio": 0.0 if candidate_count == 0 else actual / candidate_count,
        "second_load": second_load,
        "straggler_load_before": straggler_load,
        "straggler_load_after": capped_load,
    }


def find_capped_merge_ratio(
    loads: torch.Tensor,
    straggler_gpu: int,
    max_ratio: float,
    load_after_merge: Callable[[float], float],
    *,
    steps: int = 100,
) -> dict[str, float]:
    """Generic rho scan for interfaces where merge effects are externally modeled."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    second_load = second_largest_other_load(loads, straggler_gpu)
    best_ratio = max_ratio
    best_load = load_after_merge(max_ratio)
    for idx in range(steps + 1):
        rho = max_ratio * idx / steps
        load = load_after_merge(rho)
        if load <= second_load:
            best_ratio = rho
            best_load = load
            break
    return {"rho_eff": best_ratio, "load_after": best_load, "second_load": second_load}

