"""Small statistical helpers with no heavy analysis dependency."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def percentile(values: Iterable[float], q: float) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return float("nan")
    return float(np.quantile(data, q))


def summarize_latency(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"n": 0, "median": float("nan"), "p90": float("nan"), "p99": float("nan"), "iqr": float("nan")}
    return {
        "n": int(data.size),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p99": float(np.quantile(data, 0.99)),
        "iqr": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
    }


def improvement_pct(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return 100.0 * (baseline - candidate) / baseline
