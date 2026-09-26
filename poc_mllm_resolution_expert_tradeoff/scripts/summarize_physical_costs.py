from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


def _mapping(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        result[name] = Path(path)
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wall(root: Path | None, config: str | None = None) -> float | None:
    if root is None:
        return None
    by_wave: dict[int, list[float]] = defaultdict(list)
    for name in glob.glob(str(root / "timings.dp*.jsonl")):
        for row in _jsonl(Path(name)):
            row_config = (
                None
                if row.get("resolution_label") is None or row.get("k_visual") is None
                else f"{row['resolution_label']}_K{int(row['k_visual'])}"
            )
            if bool(row.get("anchor_measured", True)) and (
                config is None or row_config == config
            ):
                by_wave[int(row["batch_id"])].append(float(row["wall_ms"]))
    values = [max(rows) for rows in by_wave.values() if rows]
    return float(np.median(values)) if values else None


def _routes(root: Path | None) -> dict[str, Any]:
    rows = []
    if root is not None:
        for name in glob.glob(str(root / "route_summary*.jsonl")):
            rows.extend(_jsonl(Path(name)))
    if not rows:
        return {
            "actual_valid_pairs": None,
            "invalid_visual_slots": None,
            "max_mean": None,
            "logical_remote_dispatch_bf16_bytes": None,
            "physical_padded_rows": None,
            "physical_padding_status": "NOT_IN_ROUTE_SUMMARY",
        }
    ratios = []
    for row in rows:
        counts = np.asarray(row.get("all_rank_pair_histogram", []), dtype=float)
        if counts.size and counts.mean() > 0:
            ratios.append(float(counts.max() / counts.mean()))
    return {
        "actual_valid_pairs": int(sum(int(row["all_valid_pairs"]) for row in rows)),
        "valid_visual_pairs": int(sum(int(row["valid_visual_pairs"]) for row in rows)),
        "invalid_visual_slots": int(sum(int(row["invalid_visual_slots"]) for row in rows)),
        "max_mean": float(np.median(ratios)) if ratios else None,
        "logical_remote_dispatch_bf16_bytes": int(sum(
            int(row.get("logical_remote_dispatch_bf16_bytes", 0)) for row in rows
        )),
        "physical_padded_rows": None,
        "physical_padding_status": "REQUIRES_BACKEND_PROOF_NOT_IN_LOGICAL_ROUTE_SUMMARY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-analysis", action="append", default=[], required=True,
                        metavar="CONFIG=PATH")
    parser.add_argument("--route-root", action="append", default=[], metavar="CONFIG=PATH")
    parser.add_argument("--natural-root", action="append", default=[], metavar="CONFIG=PATH")
    parser.add_argument("--fixed32-root", action="append", default=[], metavar="CONFIG=PATH")
    parser.add_argument("--fixed128-root", action="append", default=[], metavar="CONFIG=PATH")
    parser.add_argument("--batch4-root", action="append", default=[], metavar="CONFIG=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyses = _mapping(args.profile_analysis)
    routes = _mapping(args.route_root)
    natural = _mapping(args.natural_root)
    fixed32 = _mapping(args.fixed32_root)
    fixed128 = _mapping(args.fixed128_root)
    batch4 = _mapping(args.batch4_root)
    configs = {}
    for name, path in sorted(analyses.items()):
        source = json.loads(path.read_text())
        aggregate = source["aggregate_median_ms"]
        route = _routes(routes.get(name))
        configs[name] = {
            "evidence": "MEASURED_TRUE_EP4_REPRESENTATIVE_COHORT",
            "ep_critical_ms": aggregate["routed_moe_total_ms"],
            "ttft_ms": aggregate["clean_wall_ms"],
            "natural_e2e_ms": _wall(natural.get(name), name),
            "fixed32_ms": _wall(fixed32.get(name), name),
            "fixed128_ms": _wall(fixed128.get(name), name),
            "distinct_global_batch4_ttft_ms": _wall(batch4.get(name), name),
            "dispatch_ms": aggregate["dispatch_critical_ms"],
            "expert_ms": aggregate["expert_critical_ms"],
            "combine_ms": aggregate["combine_critical_ms"],
            "profiling_overhead": source["profiling_overhead"],
            **route,
        }
    result = {
        "schema_version": "mllm_resolution_expert_tradeoff.physical_costs.v1",
        "scope": "REPRESENTATIVE_LOW_MID_MIDHIGH_HIGH_VISUAL_LENGTH_TRUE_EP4",
        "configs": configs,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
