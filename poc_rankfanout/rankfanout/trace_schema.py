"""Strict invocation alignment for cross-backend oracle analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


DEFAULT_KEY = ("workload_id", "request_id", "phase", "step_id", "layer_id", "num_tokens")


class AlignmentError(ValueError):
    """Raised when an oracle comparison would silently compare unlike work."""


def index_backend_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    key_fields: tuple[str, ...] = DEFAULT_KEY,
) -> dict[str, dict[tuple[object, ...], Mapping[str, object]]]:
    indexed: dict[str, dict[tuple[object, ...], Mapping[str, object]]] = {}
    for row in rows:
        backend = str(row["backend"])
        key = tuple(row[field] for field in key_fields)
        table = indexed.setdefault(backend, {})
        if key in table:
            raise AlignmentError(f"duplicate key for {backend}: {key}")
        table[key] = row
    if len(indexed) < 2:
        raise AlignmentError("at least two backends are required")
    key_sets = [set(table) for table in indexed.values()]
    common = set.intersection(*key_sets)
    union = set.union(*key_sets)
    if common != union:
        missing = {backend: len(union - set(table)) for backend, table in indexed.items()}
        raise AlignmentError(f"unmatched invocation keys: {missing}")
    for key in sorted(common, key=str):
        values = [table[key] for table in indexed.values()]
        hashes = {row.get("expert_placement_hash") for row in values}
        if len(hashes) > 1:
            raise AlignmentError(f"expert placement mismatch for {key}")
        route_hashes = {row.get("route_hash") for row in values if row.get("route_hash") is not None}
        if len(route_hashes) > 1:
            raise AlignmentError(f"routing mismatch for {key}")
    return indexed
