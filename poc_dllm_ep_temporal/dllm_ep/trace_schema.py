from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TraceRecord:
    run_id: str
    request_id: str
    batch_id: int
    block_id: int
    iteration_id: int
    layer_id: int
    prompt_length: int
    generation_budget: int
    block_length: int
    nfe: int
    num_positions_total: int
    num_positions_active: int
    num_masked_before: int
    num_masked_after: int
    num_newly_accepted: int
    top_k: int
    num_experts: int
    ep_size: int
    expert_assignment_counts: tuple[int, ...]
    rank_assignment_counts: tuple[int, ...]
    max_rank_load: int
    mean_rank_load: float
    max_over_mean: float
    rank_load_cv: float
    expert_to_rank_hash: str
    dispatch_ms: float | None = None
    expert_ms: float | None = None
    combine_ms: float | None = None
    moe_total_ms: float | None = None
    iteration_wall_ms: float | None = None

    def validate(self) -> None:
        if len(self.expert_assignment_counts) != self.num_experts:
            raise ValueError("expert count vector length does not match num_experts")
        if len(self.rank_assignment_counts) != self.ep_size:
            raise ValueError("rank count vector length does not match ep_size")
        expert_sum = sum(self.expert_assignment_counts)
        rank_sum = sum(self.rank_assignment_counts)
        if expert_sum != rank_sum:
            raise ValueError(f"assignment conservation failed: experts={expert_sum}, ranks={rank_sum}")
        expected = self.num_positions_total * self.top_k
        if expert_sum != expected:
            raise ValueError(f"top-k conservation failed: got={expert_sum}, expected={expected}")
        if self.max_rank_load != max(self.rank_assignment_counts, default=0):
            raise ValueError("max_rank_load disagrees with rank vector")

    @classmethod
    def from_dict(cls, obj: dict) -> "TraceRecord":
        obj = dict(obj)
        obj["expert_assignment_counts"] = tuple(obj["expert_assignment_counts"])
        obj["rank_assignment_counts"] = tuple(obj["rank_assignment_counts"])
        record = cls(**obj)
        record.validate()
        return record


def write_jsonl(path: str | Path, records: Iterable[TraceRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            record.validate()
            stream.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> list[TraceRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        return [TraceRecord.from_dict(json.loads(line)) for line in stream if line.strip()]
