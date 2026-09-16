"""Explicit expert ownership and source-rank mappings."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


SUPPORTED_EP = (2, 4, 8)


@dataclass(frozen=True)
class ExpertOwnership:
    num_experts: int
    ep_size: int
    mapping: str = "contiguous"

    def __post_init__(self) -> None:
        if self.ep_size not in SUPPORTED_EP:
            raise ValueError(f"supported EP sizes are {SUPPORTED_EP}")
        if self.mapping != "contiguous":
            raise ValueError("only the verified contiguous backend mapping is supported")
        if self.num_experts % self.ep_size:
            raise ValueError("num_experts must be divisible by ep_size")

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.ep_size

    def owner(self, expert_ids: np.ndarray) -> np.ndarray:
        expert_ids = np.asarray(expert_ids)
        if expert_ids.size and (
            expert_ids.min() < 0 or expert_ids.max() >= self.num_experts
        ):
            raise ValueError("expert id outside ownership map")
        return expert_ids // self.experts_per_rank

    def table(self) -> np.ndarray:
        return self.owner(np.arange(self.num_experts, dtype=np.int32))


@dataclass(frozen=True)
class SourcePartition:
    """Contiguous balanced source-row partition for the EP-isolation family.

    EP2 is measured directly on equal-size source shards.  For virtual EP4/8,
    non-divisible row counts use the standard balanced contiguous extension:
    the first ``count % ep_size`` ranks receive one extra row.  This extension
    is simulated and is never described as a measured dInfer mapping.
    """

    ep_size: int
    strict_divisibility: bool = False

    def ranks(self, ordered_row_keys: np.ndarray) -> np.ndarray:
        """Map rows in one invocation to source ranks.

        ``ordered_row_keys`` must be in the exact flattened physical row order
        consumed by the runtime.  Keys are accepted rather than inferred token
        positions so callers must make the source-order contract explicit.
        """

        keys = np.asarray(ordered_row_keys)
        count = len(keys)
        if self.strict_divisibility and count % self.ep_size:
            raise ValueError(
                f"physical rows ({count}) are not divisible by EP={self.ep_size}"
            )
        if count == 0:
            return np.empty(0, dtype=np.int16)
        base, remainder = divmod(count, self.ep_size)
        sizes = np.full(self.ep_size, base, dtype=np.int64)
        sizes[:remainder] += 1
        return np.repeat(np.arange(self.ep_size, dtype=np.int16), sizes)
