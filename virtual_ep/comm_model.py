"""Endpoint-aware communication models derived from true EP2 measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


SCENARIOS = ("endpoint_optimistic", "ep2_calibrated_base", "concurrency_stressed")


@dataclass(frozen=True)
class DirectionModel:
    startup_ms: float
    gb_per_s: float
    peer_ms: float
    sync_ms: float

    def latency(self, outgoing: np.ndarray, incoming: np.ndarray) -> float:
        outgoing = np.asarray(outgoing, dtype=np.float64)
        incoming = np.asarray(incoming, dtype=np.float64)
        endpoint_bytes = max(float(outgoing.max()), float(incoming.max()))
        active_peers = max(
            int(np.count_nonzero(outgoing)), int(np.count_nonzero(incoming))
        )
        payload_ms = endpoint_bytes / (self.gb_per_s * 1e9) * 1e3
        return self.startup_ms + payload_ms + self.peer_ms * max(0, active_peers - 1) + self.sync_ms


@dataclass(frozen=True)
class CommunicationScenario:
    name: str
    dispatch: DirectionModel
    combine: DirectionModel
    provenance: dict

    @property
    def validation_verdict(self) -> str:
        """Return the held-out EP2 validation state carried by the model."""

        return str(self.provenance.get("validation_verdict", "UNVALIDATED"))

    @classmethod
    def load(cls, path: Path, name: str) -> "CommunicationScenario":
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}")
        document = json.loads(Path(path).read_text())
        scenario = document["scenarios"][name]
        return cls(
            name=name,
            dispatch=DirectionModel(**scenario["dispatch"]),
            combine=DirectionModel(**scenario["combine"]),
            provenance=scenario["provenance"],
        )


def matrix_endpoint_bytes(unique_matrix: np.ndarray, hidden_size: int, element_bytes: int = 2):
    matrix = np.asarray(unique_matrix, dtype=np.int64)
    remote = matrix.copy()
    np.fill_diagonal(remote, 0)
    byte_matrix = remote * hidden_size * element_bytes
    return byte_matrix.sum(axis=1), byte_matrix.sum(axis=0)
