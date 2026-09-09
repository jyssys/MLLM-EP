#!/usr/bin/env python3
"""Build the auditable tables and conservative hierarchy oracles for this PoC.

The script never treats nested CUDA stage sums as request latency.  It uses
them only to compute the maximum module-mixing saving within each observer
restart, and divides that saving by that restart's cleanly recorded GPU BCT.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics


POLICIES = [
    "P0_global_fixed",
    "P1_vision_bucket",
    "P2_lm_length_bucket",
    "RANDOM",
    "single_multi_split",
    "text_image_split",
]


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cv(values: list[float]) -> float:
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0


def consolidate_clean(root: Path, out: Path) -> tuple[list[dict], list[dict]]:
    datasets = [
        ("max1_balanced_grid", "analysis/pool128_balanced_r0", 1, True, "screen"),
        ("max1_b128_restart_control", "analysis/pool128_b128_balanced_3r", 1, True, "caution_no_full_shape_warmup"),
        ("max1_b128_warmed", "analysis/pool128_b128_warm_r0", 1, True, "primary"),
        ("fixed16_b128", "analysis/pool128_b128_fixed16_r0", 16, True, "diagnostic_output_mismatch"),
        ("natural32_b128", "analysis/pool128_b128_natural32_r0", 32, False, "diagnostic_output_mismatch"),
    ]
    bct_rows: list[dict] = []
    sct_rows: list[dict] = []
    for dataset, relative, budget, ignore_eos, evidence in datasets:
        directory = root / relative
        if not directory.exists():
            continue
        for row in read_csv(directory / "SCHEDULE_BASELINES.csv"):
            bct_rows.append({
                "dataset": dataset,
                "evidence_class": evidence,
                "output_budget": budget,
                "ignore_eos": ignore_eos,
                **row,
            })
        for row in read_csv(directory / "SCT_RESULTS.csv"):
            sct_rows.append({
                "dataset": dataset,
                "evidence_class": evidence,
                "output_budget": budget,
                "ignore_eos": ignore_eos,
                **row,
            })
    write_csv(out / "BCT_RESULTS.csv", bct_rows)
    write_csv(out / "SCT_RESULTS.csv", sct_rows)

    summaries = []
    grouped: dict[tuple, list[float]] = collections.defaultdict(list)
    for row in bct_rows:
        grouped[(row["dataset"], int(row["batch_size"]), row["policy"])].append(float(row["gpu_bct_s"]))
    for (dataset, batch_size, policy), values in sorted(grouped.items()):
        summaries.append({
            "dataset": dataset,
            "batch_size": batch_size,
            "policy": policy,
            "repetitions": len(values),
            "bct_median_s": statistics.median(values),
            "bct_mean_s": statistics.mean(values),
            "bct_min_s": min(values),
            "bct_max_s": max(values),
            "bct_cv": cv(values),
        })
    write_csv(out / "SCHEDULE_BASELINES.csv", summaries)
    return bct_rows, sct_rows


def build_atlas(root: Path, out: Path) -> list[dict]:
    rows = []
    for restart, path in enumerate(sorted((root / "analysis").glob("pool128_b128_warm_observer_r*/MODULE_COST_ATLAS.csv"))):
        for row in read_csv(path):
            rows.append({"observer_restart": restart, "source": str(path.relative_to(root)), **row})
    write_csv(out / "MODULE_COST_ATLAS.csv", rows)
    return rows


def build_matched_controls(root: Path, out: Path) -> None:
    pool_path = root / "manifests/pool128.jsonl"
    plan_path = root / "manifests/screen_plan128_balanced_r1.json"
    pool = {row["request_id"]: row for row in (json.loads(line) for line in pool_path.read_text().splitlines())}
    plan = json.loads(plan_path.read_text())
    grouped: dict[tuple[int, str], list[dict]] = collections.defaultdict(list)
    for wave in plan["waves"]:
        grouped[(wave["batch_size"], wave["policy"])].append(wave)
    rows = []
    expected_by_batch = {}
    roles = {
        "P0_global_fixed": "NATURAL_source_order",
        "P1_vision_bucket": "ALIGNED_vision",
        "P2_lm_length_bucket": "ALIGNED_attention__conflict_screen_for_vision_moe",
        "RANDOM": "RANDOM",
        "single_multi_split": "ALIGNED_coarse_image_count",
        "text_image_split": "CONFLICTED_coarse_modality_separation",
    }
    for (batch_size, policy), waves in sorted(grouped.items()):
        ids = [request_id for wave in waves for request_id in wave["request_ids"]]
        assert len(ids) == len(pool) and set(ids) == set(pool), (batch_size, policy)
        summary = {
            "batch_size": batch_size,
            "policy": policy,
            "experimental_role": roles[policy],
            "requests": len(ids),
            "total_pixels": sum(pool[key]["total_pixels"] for key in ids),
            "total_images": sum(pool[key]["image_count"] for key in ids),
            "total_question_words": sum(pool[key]["question_words"] for key in ids),
            "family_histogram": json.dumps(dict(sorted(collections.Counter(pool[key]["family"] for key in ids).items()))),
            "max_dp_prompt_token_max_mean": max(float(wave["dp_token_max_mean"]) for wave in waves),
            "median_dp_prompt_token_max_mean": statistics.median(float(wave["dp_token_max_mean"]) for wave in waves),
            "pool_sha256": plan["pool_sha256"],
            "tokens_sha256": plan["tokens_sha256"],
        }
        invariant = tuple(summary[key] for key in ["requests", "total_pixels", "total_images", "total_question_words", "family_histogram"])
        expected_by_batch.setdefault(batch_size, invariant)
        assert expected_by_batch[batch_size] == invariant
        summary["matched_global_marginals"] = True
        rows.append(summary)
    write_csv(out / "MATCHED_POOL_CONTROLS.csv", rows)


def module_oracles(atlas_rows: list[dict], out: Path) -> list[dict]:
    by_restart: dict[int, list[dict]] = collections.defaultdict(list)
    for raw in atlas_rows:
        row = dict(raw)
        for key in ["gpu_bct_s", "vision_ms", "attention_ms", "moe_ms", "llm_forward_ms"]:
            row[key] = float(row[key])
        row["attn_residual_ms"] = max(0.0, row["llm_forward_ms"] - row["attention_ms"] - row["moe_ms"])
        row["attention_unit_ms"] = row["attention_ms"] + row["attn_residual_ms"]
        row["module_total_ms"] = row["vision_ms"] + row["llm_forward_ms"]
        by_restart[int(row["observer_restart"])].append(row)

    results = []
    for restart, rows in sorted(by_restart.items()):
        static = min(rows, key=lambda row: row["module_total_ms"])
        best_vision = min(rows, key=lambda row: row["vision_ms"])
        best_llm = min(rows, key=lambda row: row["llm_forward_ms"])
        best_attention_unit = min(rows, key=lambda row: row["attention_unit_ms"])
        best_moe = min(rows, key=lambda row: row["moe_ms"])
        # BatchGen-style LM-only: no vision boundary; choose a common
        # vision+attention unit and allow an attention->MoE regroup.
        best_encoder_attention = min(rows, key=lambda row: row["vision_ms"] + row["attention_unit_ms"])
        p3 = best_encoder_attention["vision_ms"] + best_encoder_attention["attention_unit_ms"] + best_moe["moe_ms"]
        # Stage-only MLLM: encoder may regroup once, but attention and MoE use
        # one LM grouping.
        p4 = best_vision["vision_ms"] + best_llm["llm_forward_ms"]
        # Full hierarchy: encoder and attention->MoE are both independent.
        p5 = best_vision["vision_ms"] + best_attention_unit["attention_unit_ms"] + best_moe["moe_ms"]
        # One BF16 full-hidden checkpoint for each boundary.  This is a lower
        # bound on transfer time at 1.5 TB/s, not a measured production cost.
        hidden_bytes = 45893 * 2048 * 2
        checkpoint_ms = hidden_bytes / 1.5e12 * 1e3
        p7 = p5 + 2 * checkpoint_ms
        denominator = static["gpu_bct_s"] * 1000.0
        def saving(value: float) -> float:
            return max(0.0, static["module_total_ms"] - value)
        results.append({
            "observer_restart": restart,
            "static_reference_policy": static["policy"],
            "static_module_ms": static["module_total_ms"],
            "static_gpu_bct_ms": denominator,
            "best_vision_policy": best_vision["policy"],
            "best_attention_unit_policy": best_attention_unit["policy"],
            "best_moe_policy": best_moe["policy"],
            "p3_batchgen_lm_only_ms": p3,
            "p4_independent_stage_ms": p4,
            "p5_full_hierarchy_ms": p5,
            "p7_feasible_lower_bound_ms": p7,
            "p3_direct_bct_oracle_pct": 100 * saving(p3) / denominator,
            "p4_direct_bct_oracle_pct": 100 * saving(p4) / denominator,
            "p5_direct_bct_oracle_pct": 100 * saving(p5) / denominator,
            "p7_direct_bct_oracle_pct": 100 * saving(p7) / denominator,
            "vision_increment_over_p3_pct": 100 * max(0.0, p3 - p5) / denominator,
            "moe_increment_over_p4_pct": 100 * max(0.0, p4 - p5) / denominator,
            "hidden_checkpoint_bytes_per_boundary": hidden_bytes,
            "checkpoint_copy_lower_bound_ms_per_boundary": checkpoint_ms,
        })
    write_csv(out / "HIERARCHICAL_ORACLES.csv", results)
    return results


def write_memory(out: Path) -> None:
    hidden_bytes = 45893 * 2048 * 2
    rows = [
        {
            "item": "model_weights_per_gpu_runtime_log",
            "bytes": int(15.81 * 1024**3),
            "gib": 15.81,
            "basis": "vLLM worker log; Qwen3-VL TP2/DP2/EP4 load",
        },
        {
            "item": "configured_kv_cache_per_gpu",
            "bytes": 2 * 1024**3,
            "gib": 2.0,
            "basis": "run_offline_batches.py kv_cache_memory_bytes",
        },
        {
            "item": "one_global_bf16_hidden_yield",
            "bytes": hidden_bytes,
            "gib": hidden_bytes / 1024**3,
            "basis": "45,893 prompt tokens x hidden 2,048 x BF16",
        },
        {
            "item": "two_global_bf16_hidden_yields",
            "bytes": 2 * hidden_bytes,
            "gib": 2 * hidden_bytes / 1024**3,
            "basis": "encoder and attention-to-MoE boundaries; no host spill",
        },
    ]
    write_csv(out / "MEMORY_RESULTS.csv", rows)


def build_gpu_time_log(root: Path, out: Path) -> None:
    rows = []
    for manifest_path in sorted(list((root / "profile").glob("*/manifest.json")) + list((root / "screen").glob("*/manifest.json"))):
        manifest = json.loads(manifest_path.read_text())
        start = float(manifest.get("start_unix", 0.0))
        end = float(manifest.get("end_unix", start))
        duration = max(0.0, end - start)
        exit_codes = manifest.get("exit_codes", [])
        successful = bool(exit_codes) and all(code == 0 for code in exit_codes)
        name = manifest_path.parent.name
        # The first r0 smoke was the wrong Python environment and did not
        # execute the target engine. Keep it auditable but exclude it from
        # meaningful measurement time.
        meaningful = successful and name != "smoke_clean_r0"
        arguments = manifest.get("arguments", {})
        rows.append({
            "run": name,
            "start_unix": start,
            "end_unix": end,
            "wall_seconds": duration,
            "physical_gpus": "4;5;6;7",
            "four_gpu_hours": duration * 4 / 3600,
            "instrumented": arguments.get("instrument", False),
            "output_tokens": arguments.get("output_tokens", ""),
            "ignore_eos": arguments.get("ignore_eos", ""),
            "successful": successful,
            "meaningful_measurement": meaningful,
            "notes": "wrong interpreter; target runtime not reached" if name == "smoke_clean_r0" else "",
        })
    write_csv(out / "GPU_TIME_LOG.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    consolidate_clean(args.results, args.out)
    atlas = build_atlas(args.results, args.out)
    build_matched_controls(args.results, args.out)
    oracle = module_oracles(atlas, args.out)
    write_memory(args.out)
    build_gpu_time_log(args.results, args.out)
    print(json.dumps({
        "oracle_restarts": len(oracle),
        "p5_median_pct": statistics.median(row["p5_direct_bct_oracle_pct"] for row in oracle),
        "p5_max_pct": max(row["p5_direct_bct_oracle_pct"] for row in oracle),
        "p7_median_pct": statistics.median(row["p7_direct_bct_oracle_pct"] for row in oracle),
        "vision_increment_median_pct": statistics.median(row["vision_increment_over_p3_pct"] for row in oracle),
        "moe_increment_median_pct": statistics.median(row["moe_increment_over_p4_pct"] for row in oracle),
    }, indent=2))


if __name__ == "__main__":
    main()
