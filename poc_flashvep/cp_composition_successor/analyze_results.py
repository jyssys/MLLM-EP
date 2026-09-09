#!/usr/bin/env python3
"""Generate reproducible aggregate tables for the CP composition PoC."""

from __future__ import annotations

import csv
import itertools
import json
import statistics
from pathlib import Path


ROOT = Path(
    "poc_flashvep/deepep_revalidation/results/"
    "cp_composition_successor_poc_20260909_174613"
)
OUT = ROOT / "analysis"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def common_prefix(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def track_a() -> tuple[list[dict], dict]:
    names = {
        "vanilla": "vanilla_r0_retry",
        "dcp_only": "dcp_r0_retry",
        "spec_only": "spec_r0_retry",
        "combined": "combined_r0_retry",
    }
    data = {name: read_json(ROOT / "track_a_v026" / directory / "summary.json")
            for name, directory in names.items()}
    rows = []
    for name, summary in data.items():
        rows.append(
            {
                "configuration": name,
                "e2e_p50_ms": 1000 * summary["e2e_s"]["p50"],
                "ttft_p50_ms": 1000 * summary["ttft_s"]["p50"],
                "tpot_p50_ms": 1000 * summary["tpot_s"]["p50"],
                "throughput_tokens_s": summary["throughput_tokens_s"],
            }
        )
    agreements = {}
    for lhs, rhs in (("vanilla", "dcp_only"), ("spec_only", "combined"),
                     ("vanilla", "spec_only"), ("dcp_only", "combined")):
        left = data[lhs]["output_ids"]
        right = data[rhs]["output_ids"]
        agreements[f"{lhs}_vs_{rhs}"] = {
            "exact": sum(a == b for a, b in zip(left, right, strict=True)),
            "count": len(left),
            "prefix_lengths": [common_prefix(a, b) for a, b in zip(left, right, strict=True)],
        }
    metrics = {
        "combined_vs_dcp_e2e_improvement_pct": 100 * (
            1 - data["combined"]["e2e_s"]["p50"] / data["dcp_only"]["e2e_s"]["p50"]
        ),
        "combined_vs_spec_e2e_improvement_pct": 100 * (
            1 - data["combined"]["e2e_s"]["p50"] / data["spec_only"]["e2e_s"]["p50"]
        ),
        "combined_vs_vanilla_e2e_change_pct": 100 * (
            data["combined"]["e2e_s"]["p50"] / data["vanilla"]["e2e_s"]["p50"] - 1
        ),
        "agreements": agreements,
    }
    return rows, metrics


def track_b() -> tuple[list[dict], dict]:
    rows = []
    outputs = {}
    for name in ("pcp_only", "dcp_only"):
        data = read_json(ROOT / "track_b" / f"{name}_r0" / "summary.json")
        med = data["median"]
        rows.append({"configuration": name, **med})
        outputs[name] = data["measured_output_ids"]
    metrics = {
        "pcp_vs_dcp_outputs_exact": outputs["pcp_only"] == outputs["dcp_only"],
        "pcp_dcp_combined": "ENVIRONMENT_BLOCKED_DURING_WARMUP",
        "current_source_implements_canonical_global_slot_ownership": True,
    }
    return rows, metrics


def track_c() -> tuple[list[dict], dict]:
    records = []
    patterns = ("*_screen_full_r0", "*_32k_r0", "*_content_matched_r0")
    for pattern in patterns:
        for directory in sorted((ROOT / "track_c").glob(pattern)):
            summary = read_json(directory / "summary.json")
            source = directory.name
            for row in summary["iterations"]:
                records.append({"source": source, "kind": "natural", **row})

    grouped = []
    key_fields = ("source", "length", "concurrency", "kind", "pcp", "replicas")
    records.sort(key=lambda row: tuple(row[field] for field in key_fields))
    for key, group in itertools.groupby(records, key=lambda row: tuple(row[field] for field in key_fields)):
        group = list(group)
        entry = dict(zip(key_fields, key, strict=True))
        for field in ("ttft_p50_ms", "ttft_p90_ms", "e2e_p50_ms", "e2e_p90_ms",
                      "fleet_wall_ms", "prompt_throughput_tok_s"):
            entry[field] = statistics.median(float(row[field]) for row in group)
        entry["repetitions"] = len(group)
        grouped.append(entry)

    base = [row for row in grouped if "content_matched" not in row["source"]]
    regimes = sorted({(row["length"], row["concurrency"]) for row in base})
    per_regime = {}
    for regime in regimes:
        candidates = [row for row in base if (row["length"], row["concurrency"]) == regime]
        winner = min(candidates, key=lambda row: row["fleet_wall_ms"])
        per_regime[f"{regime[0]}_c{regime[1]}"] = {
            "pcp": winner["pcp"],
            "fleet_wall_ms": winner["fleet_wall_ms"],
        }

    totals = {
        pcp: sum(next(row["fleet_wall_ms"] for row in base
                      if row["pcp"] == pcp and (row["length"], row["concurrency"]) == regime)
                 for regime in regimes)
        for pcp in (1, 2, 4)
    }
    best_static_pcp = min(totals, key=totals.get)
    oracle = sum(value["fleet_wall_ms"] for value in per_regime.values())
    # A length-only rule is allowed to choose one PCP degree per context length.
    length_only = 0.0
    length_choices = {}
    for length in sorted({length for length, _ in regimes}):
        subset = [regime for regime in regimes if regime[0] == length]
        options = {
            pcp: sum(next(row["fleet_wall_ms"] for row in base
                          if row["pcp"] == pcp and (row["length"], row["concurrency"]) == regime)
                     for regime in subset)
            for pcp in (1, 2, 4)
        }
        choice = min(options, key=options.get)
        length_choices[str(length)] = choice
        length_only += options[choice]
    # Obvious fixed-fleet control: use all four GPUs per request only at c=1,
    # otherwise preserve independent replicas. It happens to hit every winner.
    concurrency_rule = sum(next(row["fleet_wall_ms"] for row in base
                                if row["pcp"] == (4 if regime[1] == 1 else 1)
                                and (row["length"], row["concurrency"]) == regime)
                           for regime in regimes)

    content = [row for row in grouped if "content_matched" in row["source"]]
    content_winners = {}
    for key in sorted({(row["length"], row["concurrency"], row["kind"]) for row in content}):
        candidates = [row for row in content
                      if (row["length"], row["concurrency"], row["kind"]) == key]
        winner = min(candidates, key=lambda row: row["fleet_wall_ms"])
        ordered = sorted(candidates, key=lambda row: row["fleet_wall_ms"])
        content_winners[f"{key[0]}_c{key[1]}_{key[2]}"] = {
            "pcp": winner["pcp"],
            "fleet_wall_ms": winner["fleet_wall_ms"],
            "margin_vs_second_pct": 100 * (ordered[1]["fleet_wall_ms"] / winner["fleet_wall_ms"] - 1),
        }

    metrics = {
        "regimes": per_regime,
        "best_static_pcp": best_static_pcp,
        "best_static_total_ms": totals[best_static_pcp],
        "oracle_total_ms": oracle,
        "oracle_vs_best_static_pct": 100 * (1 - oracle / totals[best_static_pcp]),
        "length_only_choices": length_choices,
        "length_only_total_ms": length_only,
        "oracle_vs_length_only_pct": 100 * (1 - oracle / length_only),
        "concurrency_rule_total_ms": concurrency_rule,
        "concurrency_rule_recovers_oracle_pct": 100 * (totals[best_static_pcp] - concurrency_rule)
        / (totals[best_static_pcp] - oracle),
        "content_winners": content_winners,
        "content_changes_winner": len({value["pcp"] for value in content_winners.values()}) > 1,
    }
    return grouped, metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a_rows, a_metrics = track_a()
    b_rows, b_metrics = track_b()
    c_rows, c_metrics = track_c()
    write_csv(OUT / "track_a_request_matrix.csv", a_rows)
    write_csv(OUT / "track_b_transition.csv", b_rows)
    write_csv(OUT / "track_c_regimes.csv", c_rows)
    summary = {"track_a": a_metrics, "track_b": b_metrics, "track_c": c_metrics}
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
