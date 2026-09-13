#!/usr/bin/env python3
"""Collapse observer-heavy EP4 traces into logical block/workload rows.

The runtime emits one JSONL row per physical rank.  Latencies are collapsed
with a critical-rank maximum, while source-side routing histograms are summed.
The script never treats four rank rows as four independent request latencies.
Clean BCT remains in STATIC_BLOCK_SWEEP.csv; trace timings describe components
and shape only.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_RE = re.compile(
    r"trace_(?P<task>gsm8k|humaneval)_n(?P<n>\d+)_B(?P<block>\d+)_"
    r"m(?P<mini>\d+)_g(?P<gen>\d+)_t(?P<threshold>[0-9.]+)"
    r"(?:_L(?P<target_total>\d+))?_r(?P<repeat>\d+)\.log$"
)
FORWARD_RE = re.compile(r"^Forward:\s*(?P<nfe>\d+),\s*Time:\s*(?P<wall>[0-9.]+)", re.MULTILINE)
PREFIX = "[LLADA_DENOISE]"
LAYERS = (1, 8, 16, 24, 31)
EP_STAGES = ("router", "dispatch", "expert", "combine", "shared", "gather")
BLOCK_STAGES = ("prepare_attn", "attention", "prepare_mlp", "mlp", "postprocess")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def phase(live: int, physical: int) -> str:
    ratio = live / physical if physical else 0.0
    return "early" if ratio > 2 / 3 else "middle" if ratio > 1 / 3 else "late"


def denoise_rows(log: Path) -> list[dict]:
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        offset = line.find(PREFIX)
        if offset < 0:
            continue
        row = json.loads(line[offset + len(PREFIX) :])
        if not str(row.get("request_id", "")).startswith("measured_"):
            continue
        physical = int(row["physical_rows"])
        live = int(sum(row["remaining_before"]))
        rows.append(
            {
                **row,
                "selected_sequences": len(row["sequence_ids"]),
                "live_positions": live,
                "live_ratio": live / physical,
                "phase": phase(live, physical),
            }
        )
    return rows


def load_rank_rows(directory: Path, prefix: str) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rank in range(4):
        path = directory / f"{prefix}_rank{rank}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("request_id", "")).startswith("measured_"):
                    grouped[(rank, int(row["layer"]))].append(row)
    return grouped


def align_tail(grouped: dict[tuple[int, int], list[dict]], count: int) -> dict[tuple[int, int, int], dict]:
    aligned = {}
    for rank in range(4):
        for layer in LAYERS:
            rows = sorted(grouped[(rank, layer)], key=lambda row: row["invocation"])
            if len(rows) < count:
                raise RuntimeError(f"short trace rank={rank} layer={layer}: {len(rows)} < {count}")
            for wave_index, row in enumerate(rows[-count:]):
                aligned[(rank, layer, wave_index)] = row
    return aligned


def collapse(log: Path, match: re.Match[str]) -> tuple[list[dict], list[dict], dict]:
    logical = denoise_rows(log)
    if not logical:
        raise RuntimeError(f"no measured denoising rows: {log}")
    tag = log.stem
    trace_root = ROOT / "results" / "instrumented" / tag / "trace"
    shape = align_tail(load_rank_rows(trace_root / "shape", "shape"), len(logical))
    ep = align_tail(load_rank_rows(trace_root / "ep", "ep"), len(logical))
    block = align_tail(load_rank_rows(trace_root / "block", "block"), len(logical))
    gd = match.groupdict()
    metadata = {
        "task": gd["task"],
        "sample_count": int(gd["n"]),
        "block_length": int(gd["block"]),
        "mini_batch_size": int(gd["mini"]),
        "generation": int(gd["gen"]),
        "threshold": float(gd["threshold"]),
        "target_total_length": int(gd["target_total"] or 0),
        "repeat": int(gd["repeat"]),
        "trace_tag": tag,
    }
    layer_rows, wave_rows = [], []
    for wave_index, state in enumerate(logical):
        sampled_layers = []
        for layer in LAYERS:
            shape_ranks = [shape[(rank, layer, wave_index)] for rank in range(4)]
            ep_ranks = [ep[(rank, layer, wave_index)] for rank in range(4)]
            block_ranks = [block[(rank, layer, wave_index)] for rank in range(4)]
            expert_histogram = np.sum(
                [np.asarray(row["expert_histogram_local_source"], dtype=float) for row in shape_ranks],
                axis=0,
            )
            rank_load = np.sum(
                [np.asarray(row["rank_counts_local_source"], dtype=float) for row in shape_ranks],
                axis=0,
            )
            positive = expert_histogram[expert_histogram > 0]
            remote_pairs = sum(
                sum(row["rank_counts_local_source"]) - row["rank_counts_local_source"][source_rank]
                for source_rank, row in enumerate(shape_ranks)
            )
            physical_rows = int(state["physical_rows"])
            row = {
                **metadata,
                "wave_index": wave_index,
                "phase": state["phase"],
                "layer": layer,
                "selected_sequences": state["selected_sequences"],
                "physical_rows": physical_rows,
                "live_positions": state["live_positions"],
                "live_ratio": state["live_ratio"],
                "accepted_positions": int(sum(state["accepted"])),
                "active_experts": int(np.sum(expert_histogram > 0)),
                "rows_per_active_expert_mean": float(np.mean(positive)),
                "rows_per_active_expert_p50": float(np.median(positive)),
                "rows_per_active_expert_p90": float(np.percentile(positive, 90)),
                "tiny_expert_fraction_le4": float(np.mean(positive <= 4)),
                "rank_load_max": float(np.max(rank_load)),
                "rank_load_mean": float(np.mean(rank_load)),
                "rank_load_cv": float(np.std(rank_load) / np.mean(rank_load)),
                "critical_rank": int(np.argmax(rank_load)),
                "rank_fanout_mean": float(np.mean([row["rank_fanout_mean"] for row in shape_ranks])),
                "remote_pairs": int(remote_pairs),
                "remote_fraction": float(remote_pairs / (physical_rows * 8)),
                "dispatch_bytes_bf16": int(remote_pairs * 4096 * 2),
                "combine_bytes_bf16": int(remote_pairs * 4096 * 2),
                "expert_histogram": json.dumps(expert_histogram.astype(int).tolist(), separators=(",", ":")),
                "rank_load": json.dumps(rank_load.astype(int).tolist(), separators=(",", ":")),
            }
            for stage in EP_STAGES:
                row[f"{stage}_critical_ms"] = max(float(rank_row[f"{stage}_ms"]) for rank_row in ep_ranks)
            for stage in BLOCK_STAGES:
                row[f"{stage}_critical_ms"] = max(float(rank_row[f"{stage}_ms"]) for rank_row in block_ranks)
            row["moe_component_sum_ms"] = sum(row[f"{stage}_critical_ms"] for stage in EP_STAGES)
            row["block_component_sum_ms"] = sum(row[f"{stage}_critical_ms"] for stage in BLOCK_STAGES)
            layer_rows.append(row)
            sampled_layers.append(row)
        numeric = (
            "active_experts",
            "rows_per_active_expert_mean",
            "rows_per_active_expert_p50",
            "rows_per_active_expert_p90",
            "tiny_expert_fraction_le4",
            "rank_load_max",
            "rank_load_cv",
            "rank_fanout_mean",
            "remote_pairs",
            "remote_fraction",
            "dispatch_bytes_bf16",
            "router_critical_ms",
            "dispatch_critical_ms",
            "expert_critical_ms",
            "combine_critical_ms",
            "shared_critical_ms",
            "attention_critical_ms",
            "mlp_critical_ms",
            "block_component_sum_ms",
        )
        wave_rows.append(
            {
                **metadata,
                "wave_index": wave_index,
                "phase": state["phase"],
                "selected_sequences": state["selected_sequences"],
                "physical_rows": state["physical_rows"],
                "live_positions": state["live_positions"],
                "live_ratio": state["live_ratio"],
                "accepted_positions": int(sum(state["accepted"])),
                "model_forward_ms": float(state["model_forward_ms"]),
                **{key: float(np.mean([row[key] for row in sampled_layers])) for key in numeric},
            }
        )
    timing = FORWARD_RE.search(log.read_text(errors="replace"))
    summary = {
        **metadata,
        "nfe": int(timing["nfe"]) if timing else None,
        "trace_wall_s": float(timing["wall"]) if timing else None,
        "waves": len(wave_rows),
        "model_forward_sum_s": sum(row["model_forward_ms"] for row in wave_rows) / 1000,
    }
    return layer_rows, wave_rows, summary


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    def ranks(values: list[float]) -> np.ndarray:
        values_array = np.asarray(values, dtype=float)
        order = np.argsort(values_array, kind="stable")
        result = np.empty(len(values_array), dtype=float)
        result[order] = np.arange(len(values_array), dtype=float)
        for value in np.unique(values_array):
            mask = values_array == value
            result[mask] = result[mask].mean()
        return result
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main() -> None:
    all_layers, all_waves, runs = [], [], []
    for log in sorted((ROOT / "logs").glob("trace_*.log")):
        match = LOG_RE.fullmatch(log.name)
        if not match or not FORWARD_RE.search(log.read_text(errors="replace")):
            continue
        layers, waves, summary = collapse(log, match)
        all_layers.extend(layers)
        all_waves.extend(waves)
        runs.append(summary)
    write_csv(ROOT / "BLOCK_EP_METRICS.csv", all_layers)
    write_csv(ROOT / "BLOCK_WAVE_METRICS.csv", all_waves)
    write_csv(ROOT / "TRACE_RUN_SUMMARY.csv", runs)

    phase_rows = []
    group_keys = sorted({(row["task"], row["block_length"], row["mini_batch_size"], row["phase"]) for row in all_waves})
    fields = (
        "physical_rows",
        "live_ratio",
        "model_forward_ms",
        "active_experts",
        "rows_per_active_expert_mean",
        "tiny_expert_fraction_le4",
        "rank_load_cv",
        "rank_fanout_mean",
        "remote_fraction",
        "dispatch_bytes_bf16",
        "attention_critical_ms",
        "dispatch_critical_ms",
        "expert_critical_ms",
        "combine_critical_ms",
    )
    for task, block_length, mini, phase_name in group_keys:
        subset = [
            row for row in all_waves
            if row["task"] == task and row["block_length"] == block_length
            and row["mini_batch_size"] == mini and row["phase"] == phase_name
        ]
        output = {
            "task": task,
            "block_length": block_length,
            "mini_batch_size": mini,
            "phase": phase_name,
            "waves": len(subset),
        }
        for field in fields:
            values = np.asarray([row[field] for row in subset], dtype=float)
            output[f"{field}_median"] = float(np.median(values))
            output[f"{field}_mean"] = float(np.mean(values))
        phase_rows.append(output)
    write_csv(ROOT / "BLOCK_PHASE_SUMMARY.csv", phase_rows)

    correlation_rows = []
    for task in sorted({row["task"] for row in all_waves}):
        for block_length in sorted({row["block_length"] for row in all_waves if row["task"] == task}):
            subset = [row for row in all_waves if row["task"] == task and row["block_length"] == block_length]
            for field in fields[2:]:
                correlation_rows.append(
                    {
                        "task": task,
                        "block_length": block_length,
                        "x": "live_ratio",
                        "y": field,
                        "spearman_rho": spearman([row["live_ratio"] for row in subset], [row[field] for row in subset]),
                        "observations": len(subset),
                    }
                )
    write_csv(ROOT / "BLOCK_PHASE_CORRELATIONS.csv", correlation_rows)
    print(f"wrote {len(all_layers)} layer rows and {len(all_waves)} wave rows from {len(runs)} traces")


if __name__ == "__main__":
    main()
