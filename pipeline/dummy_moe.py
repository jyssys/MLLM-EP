"""CPU-only dummy MoE pipeline for Phase 1 integration tests."""

from __future__ import annotations

import torch

from method2.cap import cap_merge_count
from method2.importance import compute_cross_attention_importance
from method2.merge import expert_aware_merge, simulate_identity_expert_combine
from method2.selection import select_merge_candidates


def token_count_by_gpu(
    expert_assignment: torch.Tensor,
    expert_to_gpu: torch.Tensor,
    num_gpus: int,
    *,
    any_topk: bool = True,
) -> torch.Tensor:
    """Count token routes per GPU from top-k expert assignments."""

    expert_to_gpu = expert_to_gpu.to(device=expert_assignment.device, dtype=torch.long)
    routed = expert_to_gpu[expert_assignment.to(torch.long)]
    if any_topk:
        flat = routed.reshape(-1)
    else:
        flat = routed[:, 0]
    loads = torch.zeros(num_gpus, dtype=torch.float32, device=expert_assignment.device)
    loads.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return loads.cpu()


def run_dummy_pipeline(
    *,
    attention: torch.Tensor,
    text_indices: torch.Tensor | list[int],
    vision_indices: torch.Tensor | list[int],
    hidden_states: torch.Tensor,
    expert_assignment: torch.Tensor,
    expert_to_gpu: torch.Tensor,
    straggler_threshold: float,
    rho: float = 0.3,
    merge_similarity_threshold: float = 0.9,
    apply_cap: bool = True,
    routing_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | object]:
    """Run the Phase 1 dummy flow from importance through identity combine."""

    # TODO(Phase2): replace this CPU identity-combine simulation with the
    # framework insertion point before DeepSpeed-MoE dispatch/combine.
    importance = compute_cross_attention_importance(
        attention,
        text_indices=text_indices,
        vision_indices=vision_indices,
        lambda_cls=0.0,
        derope=False,
    ).to(hidden_states.device)
    if importance.shape[0] != hidden_states.shape[0]:
        raise ValueError("vision importance length must match hidden_states token count")

    num_gpus = int(expert_to_gpu.max().item()) + 1
    gpu_loads = token_count_by_gpu(expert_assignment, expert_to_gpu, num_gpus)
    selection = select_merge_candidates(
        loads=gpu_loads,
        expert_assignment=expert_assignment.cpu(),
        importance=importance.cpu(),
        threshold=straggler_threshold,
        rho=rho,
        expert_to_gpu=expert_to_gpu.cpu(),
        any_topk=True,
    )

    candidate_indices = selection["candidate_indices"].clone()
    cap_info = None
    if apply_cap and candidate_indices.numel() > 0 and selection["straggler_gpus"].numel() > 0:
        # Phase 1 cap is per first straggler GPU in the dummy pipeline.
        straggler_gpu = int(selection["straggler_gpus"][0])
        cap_info = cap_merge_count(gpu_loads, straggler_gpu, int(candidate_indices.numel()), max_merge_ratio=1.0)
        candidate_indices = candidate_indices[: int(cap_info["actual_merge_count"])]

    if selection["straggler_gpus"].numel() > 0:
        target_expert_mask = torch.isin(expert_to_gpu.cpu(), selection["straggler_gpus"])
        target_experts = torch.nonzero(target_expert_mask, as_tuple=False).flatten()
    else:
        target_experts = torch.empty(0, dtype=torch.long)

    merge_result = expert_aware_merge(
        candidate_indices,
        expert_assignment.cpu(),
        hidden_states.cpu(),
        importance.cpu(),
        target_experts=target_experts,
        similarity_threshold=merge_similarity_threshold,
    )
    output_before = simulate_identity_expert_combine(hidden_states.cpu(), expert_assignment.cpu(), None, routing_weights)
    output_after = simulate_identity_expert_combine(hidden_states.cpu(), expert_assignment.cpu(), merge_result, routing_weights)

    return {
        "importance": importance.cpu(),
        "gpu_loads": gpu_loads,
        "selection": selection,
        "cap": cap_info,
        "merge": merge_result,
        "output_before": output_before,
        "output_after": output_after,
    }
