from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


def _mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, path = value.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate prediction key: {name}")
        result[name] = Path(path)
    return result


def _summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request_values = [float(value) for value in payload["request_routed_ms"].values()]
    rows = payload["rows"]
    in_domain = int(payload["in_domain_rows"])
    total = int(payload["total_rows"])
    return {
        "target_ep": int(payload["target_ep"]),
        "target_topology": payload["target_topology"],
        "evidence": payload["evidence"],
        "request_groups": len(request_values),
        "mean_predicted_routed_ms": float(np.mean(request_values)),
        "median_predicted_routed_ms": float(np.median(request_values)),
        "in_domain_rows": in_domain,
        "total_rows": total,
        "in_domain_fraction": float(in_domain / total) if total else None,
        "all_rows_in_domain": in_domain == total,
        "ood_row_count": int(sum(not bool(row["in_domain"]) for row in rows)),
        "warning": payload.get("ep8_warning"),
        "source": str(path.resolve()),
    }


def _structural(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    max_mean = [float(row["rank_load_stats"]["max_mean"]) for row in rows]
    cv = [float(row["rank_load_stats"]["cv"]) for row in rows]
    max_second = [
        float(row["rank_load_stats"]["max"])
        / max(float(row["rank_load_stats"]["second_max"]), 1.0)
        for row in rows
    ]
    return {
        "evidence": payload["status"],
        "target_ep": int(payload["target_ep"]),
        "owner_rule": payload["owner_rule"],
        "source_rule": payload["source_rule"],
        "invocations": len(rows),
        "total_assignments": int(sum(row["assignment_invariant"] for row in rows)),
        "total_logical_dispatch_bytes": int(sum(
            row["logical_dispatch_bytes"] for row in rows
        )),
        "total_logical_combine_bytes": int(sum(
            row["logical_combine_bytes"] for row in rows
        )),
        "median_max_mean": float(np.median(max_mean)),
        "p90_max_mean": float(np.quantile(max_mean, 0.9)),
        "median_max_second": float(np.median(max_second)),
        "median_cv": float(np.median(cv)),
        "median_max_fanout": float(np.median([
            row["max_fanout"] for row in rows
        ])),
        "source": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True,
                        metavar="CONFIG_EP=PATH")
    parser.add_argument("--replay", action="append", default=[],
                        metavar="CONFIG_EP=PATH")
    parser.add_argument("--physical-costs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = _mapping(args.prediction)
    configs: dict[str, dict[str, Any]] = {}
    for key, path in sorted(sources.items()):
        config, ep_text = key.rsplit("_EP", 1)
        ep = int(ep_text)
        value = _summarize(path)
        if value["target_ep"] != ep:
            raise RuntimeError(f"target mismatch for {key}: {value['target_ep']}")
        configs.setdefault(config, {})[f"EP{ep}"] = value

    for key, path in sorted(_mapping(args.replay).items()):
        config, ep_text = key.rsplit("_EP", 1)
        ep = int(ep_text)
        value = configs.setdefault(config, {}).setdefault(f"EP{ep}", {})
        structural = _structural(path)
        if structural["target_ep"] != ep:
            raise RuntimeError(f"structural target mismatch for {key}")
        value["structural"] = structural

    baseline = configs.get("R0_K8")
    if baseline is None:
        raise RuntimeError("R0_K8 virtual baseline missing")
    for config, values in configs.items():
        for ep in (4, 8):
            name = f"EP{ep}"
            current = values.get(name)
            base = baseline.get(name)
            if current is None or base is None:
                continue
            current["point_gain_vs_r0_k8_pct"] = float(
                100
                * (1 - current["mean_predicted_routed_ms"]
                   / base["mean_predicted_routed_ms"])
            )
            current["gain_decision_status"] = (
                "UNVALIDATED_EXTRAPOLATION"
                if ep == 8
                else "UNUSABLE_OOD_POINT_ESTIMATE"
                if not current["all_rows_in_domain"]
                else "PART1_MODEL_STILL_FAILED_GLOBAL_ACCEPTANCE"
            )

    if args.physical_costs is not None:
        measured = json.loads(
            args.physical_costs.read_text(encoding="utf-8")
        )["configs"]
        for config, values in configs.items():
            ep4 = values.get("EP4")
            actual = measured.get(config, {}).get("ep_critical_ms")
            if ep4 is None or actual is None:
                continue
            predicted = float(ep4["mean_predicted_routed_ms"])
            actual = float(actual)
            ep4["actual_true_ep4_routed_ms"] = actual
            ep4["signed_error_ms"] = predicted - actual
            ep4["ape_pct"] = abs(predicted - actual) / actual * 100

    atomic_json(args.output, {
        "schema_version": "mllm_resolution_expert_tradeoff.virtual_summary.v1",
        "configs": configs,
        "interpretation": (
            "EP4 values are frozen Part-1 model point estimates and cannot decide "
            "candidate gains because the Part-1 simulator failed its global gate; "
            "rows outside its frozen domain are explicitly counted. EP8 is an "
            "unvalidated sensitivity extrapolation. Physical EP4 is primary."
        ),
    })


if __name__ == "__main__":
    main()
