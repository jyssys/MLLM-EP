"""Measured rank-local expert replay model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass(frozen=True)
class ComputeSample:
    active_experts: int
    assignments: int
    max_rows_per_expert: int
    cv_rows_per_expert: float
    latency_ms: float
    split: str
    sample_id: str


class ComputeModel:
    """Nearest-neighbour model over *measured* target-kernel replays.

    It intentionally avoids EP2/P scaling assumptions.  Each prediction is
    tied to the closest measured local workload shape and exposes the distance
    so out-of-envelope EP4/EP8 predictions can be flagged.
    """

    FEATURES = (
        "active_experts",
        "assignments",
        "max_rows_per_expert",
        "cv_rows_per_expert",
    )

    def __init__(self, samples: list[ComputeSample]):
        if not samples:
            raise ValueError("compute model requires measured samples")
        self.samples = samples
        self._features = np.asarray(
            [
                [
                    sample.active_experts,
                    sample.assignments,
                    sample.max_rows_per_expert,
                    sample.cv_rows_per_expert,
                ]
                for sample in samples
            ],
            dtype=np.float64,
        )
        self._scale = np.maximum(np.ptp(self._features, axis=0), 1.0)

    @classmethod
    def from_csv(
        cls, path: Path, allowed_splits: tuple[str, ...] | None = None
    ) -> "ComputeModel":
        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if allowed_splits is not None:
            rows = [row for row in rows if row["split"] in allowed_splits]
        return cls(
            [
                ComputeSample(
                    active_experts=int(row["active_experts"]),
                    assignments=int(row["assignments"]),
                    max_rows_per_expert=int(row["max_rows_per_expert"]),
                    cv_rows_per_expert=float(row["cv_rows_per_expert"]),
                    latency_ms=float(row["latency_ms"]),
                    split=row["split"],
                    sample_id=row["sample_id"],
                )
                for row in rows
            ]
        )

    def predict(self, expert_histogram: np.ndarray) -> tuple[float, float, str]:
        counts = np.asarray(expert_histogram, dtype=np.float64)
        nonzero = counts[counts > 0]
        if nonzero.size:
            query = np.asarray(
                [len(nonzero), nonzero.sum(), nonzero.max(), nonzero.std() / nonzero.mean()]
            )
        else:
            query = np.zeros(4)
        distances = np.linalg.norm((self._features - query) / self._scale, axis=1)
        index = int(np.argmin(distances))
        sample = self.samples[index]
        return sample.latency_ms, float(distances[index]), sample.sample_id
