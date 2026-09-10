"""Best-static, perfect dynamic, and Amdahl-adjusted oracle calculations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from .metrics import improvement_pct
from .trace_schema import DEFAULT_KEY, index_backend_rows


def _group_cost(
    indexed: dict[str, dict[tuple[object, ...], Mapping[str, object]]],
    latency_field: str,
    group_indices: tuple[int, ...],
) -> tuple[float, dict[tuple[object, ...], dict[str, float]]]:
    grouped: dict[tuple[object, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for backend, table in indexed.items():
        for key, row in table.items():
            group = tuple(key[i] for i in group_indices)
            grouped[group][backend] += float(row[latency_field])
    oracle = sum(min(costs.values()) for costs in grouped.values())
    return oracle, {key: dict(value) for key, value in grouped.items()}


def compute_oracles(
    rows: Iterable[Mapping[str, object]],
    *,
    latency_field: str = "moe_total_ms",
    key_fields: tuple[str, ...] = DEFAULT_KEY,
) -> dict[str, object]:
    indexed = index_backend_rows(rows, key_fields=key_fields)
    backends = sorted(indexed)
    totals = {
        backend: sum(float(row[latency_field]) for row in table.values())
        for backend, table in indexed.items()
    }
    best_backend = min(totals, key=totals.get)
    best_static = totals[best_backend]
    per_invocation = sum(
        min(float(indexed[backend][key][latency_field]) for backend in backends)
        for key in next(iter(indexed.values()))
    )
    # DEFAULT_KEY: workload, request, phase, step, layer, M.
    per_step, step_groups = _group_cost(indexed, latency_field, (0, 1, 2, 3))
    per_request, request_groups = _group_cost(indexed, latency_field, (0, 1))
    return {
        "latency_field": latency_field,
        "backends": backends,
        "backend_totals_ms": totals,
        "best_static_backend": best_backend,
        "best_static_ms": best_static,
        "per_invocation_oracle_ms": per_invocation,
        "per_invocation_improvement_pct": improvement_pct(best_static, per_invocation),
        "per_step_oracle_ms": per_step,
        "per_step_improvement_pct": improvement_pct(best_static, per_step),
        "per_request_oracle_ms": per_request,
        "per_request_improvement_pct": improvement_pct(best_static, per_request),
        "step_groups": step_groups,
        "request_groups": request_groups,
    }


def amdahl_ttft_oracle(
    *,
    ttft_static_ms: float,
    replaceable_static_ms: float,
    replaceable_oracle_ms: float,
) -> dict[str, float]:
    if min(ttft_static_ms, replaceable_static_ms, replaceable_oracle_ms) < 0:
        raise ValueError("latencies must be non-negative")
    if replaceable_static_ms > ttft_static_ms:
        raise ValueError("replaceable boundary cannot exceed TTFT")
    if replaceable_oracle_ms > replaceable_static_ms:
        raise ValueError("oracle must not be slower than best static")
    projected = ttft_static_ms - replaceable_static_ms + replaceable_oracle_ms
    return {
        "ttft_static_ms": ttft_static_ms,
        "replaceable_static_ms": replaceable_static_ms,
        "replaceable_oracle_ms": replaceable_oracle_ms,
        "projected_ttft_oracle_ms": projected,
        "projected_ttft_improvement_pct": improvement_pct(ttft_static_ms, projected),
    }
