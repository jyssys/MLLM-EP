from __future__ import annotations

import numpy as np


def expert_owner_map(num_experts: int, target_ep: int) -> np.ndarray:
    if target_ep <= 0 or num_experts % target_ep:
        raise ValueError("target EP must be positive and divide the expert count")
    return np.arange(num_experts, dtype=np.int16) // (num_experts // target_ep)


def validate_topk(expert_ids: np.ndarray, num_experts: int = 128) -> None:
    if expert_ids.ndim != 2:
        raise ValueError(f"expert IDs must be [rows,k], got {expert_ids.shape}")
    if expert_ids.size and (expert_ids.min() < 0 or expert_ids.max() >= num_experts):
        raise ValueError("expert ID outside model contract")
    if expert_ids.size and np.any(np.diff(np.sort(expert_ids, axis=1), axis=1) == 0):
        raise ValueError("duplicate expert within a logical token route")


def source_rank(dp_rank: int, tp_rank: int, physical_ep: int = 4) -> int:
    rank = dp_rank * 2 + tp_rank
    if rank >= physical_ep:
        raise ValueError((dp_rank, tp_rank, physical_ep))
    return rank


def source_rank_for_target(measured_dp_rank: int, tp_rank: int,
                           target_ep: int, request_ordinal: int) -> int:
    """Map a TP source copy into the target TP2/DP{1,2,4} family.

    EP4 preserves the physically observed DP owner.  EP2 has one DP owner.
    EP8 keeps global arrivals fixed and deterministically distributes requests
    over four DP owners; it does not clone a request onto idle ranks.
    """
    if target_ep not in (2, 4, 8) or tp_rank not in (0, 1):
        raise ValueError((measured_dp_rank, tp_rank, target_ep, request_ordinal))
    if target_ep == 2:
        target_dp = 0
    elif target_ep == 4:
        if measured_dp_rank not in (0, 1):
            raise ValueError(measured_dp_rank)
        target_dp = measured_dp_rank
    else:
        target_dp = request_ordinal % 4
    return target_dp * 2 + tp_rank
