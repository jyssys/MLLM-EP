#!/usr/bin/env python3
"""Analyze logical/physical mismatches in LLaDA2-Flash true-EP4 traces.

Clean logs are the only source of request-level latency.  Shape, temporal,
and CUDA-event traces are observer-heavy and are used only for structure,
component attribution, and explicitly labelled counterfactual oracles.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
LAYERS = tuple(range(1, 32))
REP_LAYERS = (1, 16, 31)
STAGES = ("router", "dispatch", "expert", "combine", "shared", "gather")
FORWARD_RE = re.compile(r"Forward:\s*(?P<nfe>\d+), Time:\s*(?P<wall>[0-9.]+).*TPS:\s*(?P<tps>[0-9.]+)")
PREFIX = "[LLADA_DENOISE]"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def phase(live: int, physical: int) -> str:
    ratio = live / physical if physical else 0.0
    return "early" if ratio > 2 / 3 else "middle" if ratio > 1 / 3 else "late"


def parse_log(path: Path) -> tuple[list[dict], dict]:
    logical: list[dict] = []
    final = None
    with path.open(errors="replace") as handle:
        for line in handle:
            match = FORWARD_RE.search(line)
            if match:
                final = {
                    "nfe": int(match.group("nfe")),
                    "wall_seconds": float(match.group("wall")),
                    "tokens_per_second": float(match.group("tps")),
                }
            position = line.find(PREFIX)
            if position < 0:
                continue
            record = json.loads(line[position + len(PREFIX):])
            if not str(record.get("request_id", "")).startswith("measured_"):
                continue
            physical = int(record["physical_rows"])
            live = int(sum(record["remaining_before"]))
            record["wave"] = len(logical)
            record["live_rows"] = live
            record["live_ratio"] = live / physical
            record["phase"] = phase(live, physical)
            record["ready_requests"] = len(record["sequence_ids"])
            logical.append(record)
    if final is None:
        raise RuntimeError(f"No completed Forward timing in {path}")
    return logical, final


def clean_summary() -> tuple[list[dict], dict[str, float]]:
    rows = []
    pattern = re.compile(r"clean_(gsm8k|humaneval)_ep4_b32_mini32_r(\d+)_g32\.log")
    for path in sorted((ROOT / "logs").glob("clean_*.log")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        _, final = parse_log(path)
        rows.append({"dataset": match.group(1), "repeat": int(match.group(2)), **final})
    medians = {
        dataset: float(np.median([row["wall_seconds"] for row in rows if row["dataset"] == dataset]))
        for dataset in sorted({row["dataset"] for row in rows})
    }
    return rows, medians


def distribution(histogram: np.ndarray) -> dict:
    positive = histogram[histogram > 0]
    if not positive.size:
        return {
            "pairs": 0, "active_experts": 0, "rows_mean": 0.0,
            "rows_p50": 0.0, "rows_p90": 0.0, "rows_max": 0.0,
            "tiny_le1": 0.0, "tiny_le2": 0.0, "tiny_le4": 0.0,
            "tiny_le8": 0.0,
        }
    return {
        "pairs": int(histogram.sum()),
        "active_experts": int(positive.size),
        "rows_mean": float(positive.mean()),
        "rows_p50": float(np.median(positive)),
        "rows_p90": float(np.quantile(positive, 0.9)),
        "rows_max": float(positive.max()),
        "tiny_le1": float(np.mean(positive <= 1)),
        "tiny_le2": float(np.mean(positive <= 2)),
        "tiny_le4": float(np.mean(positive <= 4)),
        "tiny_le8": float(np.mean(positive <= 8)),
    }


def open_rank_files(directory: Path, prefix: str):
    handles = [(directory / f"{prefix}_rank{rank}.jsonl").open() for rank in range(4)]
    try:
        for lines in zip(*handles, strict=True):
            records = [json.loads(line) for line in lines]
            yield records
    finally:
        for handle in handles:
            handle.close()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def process_shape(dataset: str, result_dir: Path, log: Path):
    logical, _ = parse_log(log)
    layer_rows: list[dict] = []
    rank_rows: list[dict] = []
    route_rows: list[dict] = []
    counters = defaultdict(int)
    previous_route: dict[int, dict[tuple[int, int, int], tuple[tuple[int, ...], frozenset[int]]]] = defaultdict(dict)
    previous_rank_load: dict[int, np.ndarray] = {}

    for records in open_rank_files(result_dir, "shape"):
        if len({(row["layer"], row["invocation"], row["request_id"]) for row in records}) != 1:
            raise RuntimeError("rank shape traces are not aligned")
        if not str(records[0].get("request_id", "")).startswith("measured_"):
            continue
        layer = int(records[0]["layer"])
        wave = counters[layer]
        if wave >= len(logical):
            raise RuntimeError(f"extra shape record {dataset=} {layer=} {wave=}")
        state = logical[wave]
        physical = int(state["physical_rows"])
        # The measured section starts with one prompt-prefill forward.  NFE
        # includes it, whereas LLADA_DENOISE begins with the first refinement
        # wave.  Exclude that boundary event from refinement analysis.
        if wave == 0 and int(records[0]["physical_m_global"]) != physical:
            continue
        counters[layer] += 1
        if any(int(row["physical_m_global"]) != physical for row in records):
            raise RuntimeError(f"physical-M mismatch {dataset=} {layer=} {wave=}")
        if records[0]["sequence_ids"] != state["sequence_ids"] or records[0]["block_starts"] != state["block_starts"]:
            raise RuntimeError(f"logical/shape identity mismatch {dataset=} {layer=} {wave=}")

        full_hist = np.sum(
            [np.asarray(row["expert_histogram_local_source"], dtype=np.int64) for row in records], axis=0
        )
        live_hist = np.zeros(256, dtype=np.int64)
        mask = np.asarray(records[0]["mask_before"], dtype=bool).reshape(-1)
        if mask.size != physical:
            raise RuntimeError(f"mask size mismatch {mask.size=} {physical=}")
        confidence = np.asarray(state.get("token_confidence", np.full(mask.shape, np.nan)), dtype=float).reshape(-1)
        margin = np.asarray(state.get("confidence_margin", np.full(mask.shape, np.nan)), dtype=float).reshape(-1)
        router_entropy = np.empty(physical, dtype=float)
        router_support = np.empty(physical, dtype=float)
        top1_probability = np.empty(physical, dtype=float)
        topk_mass = np.empty(physical, dtype=float)
        topk_gap = np.empty(physical, dtype=float)
        remote_pairs = 0
        current_routes: dict[tuple[int, int, int], tuple[tuple[int, ...], frozenset[int]]] = {}
        matched = ordered_exact = set_exact = 0
        overlap_sum = rank_set_exact = 0
        rank_overlap_sum = 0.0
        matched_live = ordered_exact_live = set_exact_live = 0
        overlap_live_sum = 0

        sequence_ids = [int(value) for value in state["sequence_ids"]]
        block_starts = [int(value) for value in state["block_starts"]]
        for source, row in enumerate(records):
            start = int(row["global_row_start"])
            count = int(row["physical_m_local"])
            stop = start + count
            ids = np.asarray(row["topk_ids_local_source"], dtype=np.int64)
            if ids.shape != (count, 8):
                raise RuntimeError(f"route shape mismatch {ids.shape=} {(count, 8)=}")
            local_live = mask[start:stop]
            if local_live.any():
                live_hist += np.bincount(ids[local_live].reshape(-1), minlength=256)
            owners = ids // 64
            remote_pairs += int(np.sum(owners != source))
            for name, target in (
                ("router_entropy_local_source", router_entropy),
                ("router_effective_support_local_source", router_support),
                ("top1_probability_local_source", top1_probability),
                ("topk_probability_mass_local_source", topk_mass),
                ("topk_probability_gap_local_source", topk_gap),
            ):
                target[start:stop] = np.asarray(row[name], dtype=float)
            for local_index, route_array in enumerate(ids):
                global_index = start + local_index
                request_index, position = divmod(global_index, 32)
                key = (sequence_ids[request_index], block_starts[request_index], position)
                route = tuple(int(value) for value in route_array)
                rank_set = frozenset(int(value // 64) for value in route_array)
                current_routes[key] = (route, rank_set)
                previous = previous_route[layer].get(key)
                if previous is None:
                    continue
                matched += 1
                overlap = len(set(route) & set(previous[0]))
                overlap_sum += overlap
                ordered_exact += int(route == previous[0])
                set_exact += int(set(route) == set(previous[0]))
                rank_set_exact += int(rank_set == previous[1])
                rank_overlap_sum += len(rank_set & previous[1]) / max(1, len(rank_set | previous[1]))
                if mask[global_index]:
                    matched_live += 1
                    overlap_live_sum += overlap
                    ordered_exact_live += int(route == previous[0])
                    set_exact_live += int(set(route) == set(previous[0]))
        previous_route[layer].update(current_routes)

        full = distribution(full_hist)
        compact = distribution(live_hist)
        rank_load = full_hist.reshape(4, 64).sum(axis=1)
        compact_rank_load = live_hist.reshape(4, 64).sum(axis=1)
        full_cv = float(rank_load.std() / rank_load.mean()) if rank_load.mean() else 0.0
        compact_cv = float(compact_rank_load.std() / compact_rank_load.mean()) if compact_rank_load.mean() else 0.0
        rank_cosine = cosine(rank_load, previous_rank_load[layer]) if layer in previous_rank_load else math.nan
        previous_rank_load[layer] = rank_load.copy()
        record = {
            "dataset": dataset, "wave": wave, "layer": layer,
            "phase": state["phase"], "normalized_wave": wave / max(1, len(logical) - 1),
            "ready_requests": state["ready_requests"], "physical_rows": physical,
            "live_rows": state["live_rows"], "live_ratio": state["live_ratio"],
            "accepted_rows": int(sum(state["accepted"])),
            **{f"full_{key}": value for key, value in full.items()},
            **{f"compact_{key}": value for key, value in compact.items()},
            "rank_load_cv": full_cv,
            "compact_rank_load_cv": compact_cv,
            "critical_rank": int(rank_load.argmax()),
            "compact_critical_rank": int(compact_rank_load.argmax()) if compact_rank_load.sum() else -1,
            "rank_load_cosine_lag1": rank_cosine,
            "rank_fanout_mean": float(np.mean([row["rank_fanout_mean"] for row in records])),
            "remote_pairs": remote_pairs,
            "remote_fraction": remote_pairs / max(1, physical * 8),
            "remote_bytes_bf16_one_way": remote_pairs * 4096 * 2,
            "token_confidence_live_mean": float(np.nanmean(confidence[mask])) if mask.any() else math.nan,
            "token_confidence_live_p50": float(np.nanmedian(confidence[mask])) if mask.any() else math.nan,
            "token_margin_live_mean": float(np.nanmean(margin[mask])) if mask.any() else math.nan,
            "router_entropy_live_mean": float(router_entropy[mask].mean()) if mask.any() else math.nan,
            "router_support_live_mean": float(router_support[mask].mean()) if mask.any() else math.nan,
            "top1_probability_live_mean": float(top1_probability[mask].mean()) if mask.any() else math.nan,
            "topk_mass_live_mean": float(topk_mass[mask].mean()) if mask.any() else math.nan,
            "topk_gap_live_mean": float(topk_gap[mask].mean()) if mask.any() else math.nan,
            "router_entropy_all_mean": float(router_entropy.mean()),
            "topk_mass_all_mean": float(topk_mass.mean()),
            "full_histogram": json.dumps(full_hist.tolist(), separators=(",", ":")),
            "compact_histogram": json.dumps(live_hist.tolist(), separators=(",", ":")),
        }
        layer_rows.append(record)
        route_rows.append({
            "dataset": dataset, "wave": wave, "layer": layer, "phase": state["phase"],
            "matched_rows": matched,
            "topk_overlap_fraction_lag1": overlap_sum / max(1, matched * 8),
            "ordered_exact_fraction_lag1": ordered_exact / max(1, matched),
            "set_exact_fraction_lag1": set_exact / max(1, matched),
            "destination_set_exact_fraction_lag1": rank_set_exact / max(1, matched),
            "destination_set_jaccard_lag1": rank_overlap_sum / max(1, matched),
            "matched_live_rows": matched_live,
            "live_topk_overlap_fraction_lag1": overlap_live_sum / max(1, matched_live * 8),
            "live_ordered_exact_fraction_lag1": ordered_exact_live / max(1, matched_live),
            "live_set_exact_fraction_lag1": set_exact_live / max(1, matched_live),
            "rank_load_cosine_lag1": rank_cosine,
        })
        for owner_rank in range(4):
            owner_full = distribution(full_hist[owner_rank * 64:(owner_rank + 1) * 64])
            owner_compact = distribution(live_hist[owner_rank * 64:(owner_rank + 1) * 64])
            rank_rows.append({
                "dataset": dataset, "wave": wave, "layer": layer, "rank": owner_rank,
                "phase": state["phase"], "physical_rows": physical, "live_ratio": state["live_ratio"],
                **{f"full_{key}": value for key, value in owner_full.items()},
                **{f"compact_{key}": value for key, value in owner_compact.items()},
                "global_rank_load_cv": full_cv,
                "global_compact_rank_load_cv": compact_cv,
                "rank_fanout_mean": record["rank_fanout_mean"],
                "remote_fraction": record["remote_fraction"],
            })
    for layer in LAYERS:
        if counters[layer] != len(logical):
            raise RuntimeError(f"incomplete shape {dataset=} {layer=} {counters[layer]} != {len(logical)}")
    return layer_rows, rank_rows, route_rows, logical


def process_stage(dataset: str, result_dir: Path, log: Path):
    logical, final = parse_log(log)
    layer_rows: list[dict] = []
    rank_rows: list[dict] = []
    counters = defaultdict(int)
    for records in open_rank_files(result_dir, "ep"):
        if len({(row["layer"], row["invocation"], row["request_id"]) for row in records}) != 1:
            raise RuntimeError("rank stage traces are not aligned")
        if not str(records[0].get("request_id", "")).startswith("measured_"):
            continue
        layer = int(records[0]["layer"])
        wave = counters[layer]
        state = logical[wave]
        if wave == 0 and int(records[0]["physical_m_global"]) != int(state["physical_rows"]):
            continue
        counters[layer] += 1
        if records[0]["sequence_ids"] != state["sequence_ids"] or records[0]["block_starts"] != state["block_starts"]:
            raise RuntimeError(f"logical/stage identity mismatch {dataset=} {layer=} {wave=}")
        aggregate = {
            "dataset": dataset, "wave": wave, "layer": layer, "phase": state["phase"],
            "physical_rows": state["physical_rows"], "live_ratio": state["live_ratio"],
        }
        for stage in STAGES:
            aggregate[f"{stage}_critical_ms"] = max(float(row[f"{stage}_ms"]) for row in records)
            aggregate[f"{stage}_mean_rank_ms"] = float(np.mean([float(row[f"{stage}_ms"]) for row in records]))
        aggregate["moe_critical_ms"] = sum(aggregate[f"{stage}_critical_ms"] for stage in STAGES)
        layer_rows.append(aggregate)
        for rank, row in enumerate(records):
            rank_rows.append({
                "dataset": dataset, "wave": wave, "layer": layer, "rank": rank,
                "phase": state["phase"], "physical_rows": state["physical_rows"],
                **{f"{stage}_ms": float(row[f"{stage}_ms"]) for stage in STAGES},
            })
    for layer in LAYERS:
        if counters[layer] != len(logical):
            raise RuntimeError(f"incomplete stage {dataset=} {layer=} {counters[layer]} != {len(logical)}")
    return layer_rows, rank_rows, logical, final


def process_temporal(dataset: str, result_dir: Path, log: Path):
    logical, _ = parse_log(log)
    records_by_key = {}
    path = result_dir / "ep_rank0.jsonl"
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("request_id", "")).startswith("measured_"):
                records_by_key[(int(row["layer"]), int(row["invocation"]))] = row
    rows = []
    for layer in REP_LAYERS:
        records = sorted((row for (candidate, _), row in records_by_key.items() if candidate == layer), key=lambda row: row["invocation"])
        if len(records) < len(logical):
            raise RuntimeError(f"short temporal trace {dataset=} {layer=} {len(records)=}")
        records = records[-len(logical):]
        for wave, (state, row) in enumerate(zip(logical, records, strict=True)):
            out = {"dataset": dataset, "wave": wave, "layer": layer, "phase": state["phase"]}
            for key, value in row.items():
                if key.startswith("hidden_") or key.startswith("moe_output_"):
                    out[key] = value
            rows.append(out)
    return rows


def join_rows(shape_rows: list[dict], stage_rows: list[dict], shape_rank: list[dict], stage_rank: list[dict]):
    stage_map = {(row["dataset"], row["wave"], row["layer"]): row for row in stage_rows}
    joined_layers = []
    for row in shape_rows:
        key = (row["dataset"], row["wave"], row["layer"])
        joined_layers.append({**row, **{k: v for k, v in stage_map[key].items() if k.endswith("_ms")}})
    stage_rank_map = {(row["dataset"], row["wave"], row["layer"], row["rank"]): row for row in stage_rank}
    joined_ranks = []
    for row in shape_rank:
        key = (row["dataset"], row["wave"], row["layer"], row["rank"])
        joined_ranks.append({**row, **{k: v for k, v in stage_rank_map[key].items() if k.endswith("_ms")}})
    return joined_layers, joined_ranks


def phase_summary(layer_rows: list[dict]) -> list[dict]:
    metrics = (
        "live_ratio", "full_active_experts", "full_rows_p50", "full_tiny_le4",
        "compact_active_experts", "compact_rows_p50", "compact_tiny_le4",
        "router_entropy_live_mean", "router_support_live_mean", "token_confidence_live_mean",
        "topk_mass_live_mean", "rank_load_cv", "rank_fanout_mean", "remote_fraction",
        "expert_critical_ms", "dispatch_critical_ms", "combine_critical_ms", "moe_critical_ms",
    )
    rows = []
    for dataset in sorted({row["dataset"] for row in layer_rows}):
        for phase_name in ("early", "middle", "late"):
            subset = [row for row in layer_rows if row["dataset"] == dataset and row["phase"] == phase_name]
            out = {"dataset": dataset, "phase": phase_name, "observations": len(subset)}
            for metric in metrics:
                values = np.asarray([row[metric] for row in subset if not math.isnan(float(row[metric]))], dtype=float)
                out[f"{metric}_median"] = float(np.median(values)) if values.size else math.nan
                out[f"{metric}_mean"] = float(np.mean(values)) if values.size else math.nan
            rows.append(out)
    return rows


def fit_predictors(rank_rows: list[dict]):
    count_features = ["full_pairs", "layer"]
    shape_features = count_features + ["full_active_experts", "full_rows_p50", "full_rows_p90", "full_rows_max", "full_tiny_le1", "full_tiny_le4"]
    ep_features = shape_features + ["global_rank_load_cv", "rank_fanout_mean", "remote_fraction"]
    feature_sets = {"M0_count": count_features, "M1_shape": shape_features, "M2_shape_ep": ep_features}
    rows = []
    models = {}
    for target in ("expert_ms", "dispatch_ms", "combine_ms"):
        y_all = np.asarray([row[target] for row in rank_rows], dtype=float)
        group_values = defaultdict(list)
        for row, value in zip(rank_rows, y_all, strict=True):
            group_values[(row["dataset"], row["layer"], row["rank"])].append(float(value))
        group_p99 = {key: float(np.quantile(values, 0.99)) for key, values in group_values.items()}
        robust_keep = np.asarray([
            value <= group_p99[(row["dataset"], row["layer"], row["rank"])]
            for row, value in zip(rank_rows, y_all, strict=True)
        ])
        for sample_filter, keep in (("raw", np.ones(len(rank_rows), dtype=bool)), ("group_p99_trimmed", robust_keep)):
            for model_name, features in feature_sets.items():
                predictions = np.full_like(y_all, np.nan)
                x = np.asarray([[float(row[key]) for key in features] for row in rank_rows], dtype=float)
                for held_out in sorted({row["dataset"] for row in rank_rows}):
                    train = np.asarray([row["dataset"] != held_out for row in rank_rows]) & keep
                    test = np.asarray([row["dataset"] == held_out for row in rank_rows]) & keep
                    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=8, max_features=1.0, random_state=7, n_jobs=-1)
                    model.fit(x[train], y_all[train])
                    predictions[test] = model.predict(x[test])
                rmse = float(mean_squared_error(y_all[keep], predictions[keep]) ** 0.5)
                rows.append({
                    "target": target, "sample_filter": sample_filter,
                    "observations": int(keep.sum()), "model": model_name,
                    "features": "+".join(features),
                    "leave_dataset_out_rmse_ms": rmse,
                    "leave_dataset_out_r2": float(r2_score(y_all[keep], predictions[keep])),
                    "winner_accuracy_10pct": float(np.mean(np.abs(predictions[keep] - y_all[keep]) <= 0.1 * np.maximum(y_all[keep], 1e-6))),
                })
                model = ExtraTreesRegressor(n_estimators=240, min_samples_leaf=8, max_features=1.0, random_state=11, n_jobs=-1)
                model.fit(x[keep], y_all[keep])
                models[(target, model_name, sample_filter)] = (model, features)
    correlations = []
    candidates = [
        "full_pairs", "full_active_experts", "full_rows_p50", "full_tiny_le4",
        "global_rank_load_cv", "rank_fanout_mean", "remote_fraction",
    ]
    for target in ("expert_ms", "dispatch_ms", "combine_ms"):
        for feature in candidates:
            value = spearmanr([row[feature] for row in rank_rows], [row[target] for row in rank_rows]).statistic
            correlations.append({"target": target, "feature": feature, "spearman": float(value)})
    return rows, correlations, models


def matched_pairs(rank_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rank_rows:
        groups[(row["dataset"], row["layer"], row["rank"])].append(row)
    output = []
    for (dataset, layer, rank), values in groups.items():
        values.sort(key=lambda row: row["full_pairs"])
        for left, right in zip(values, values[1:]):
            denom = max(1, min(left["full_pairs"], right["full_pairs"]))
            if abs(left["full_pairs"] - right["full_pairs"]) / denom > 0.05:
                continue
            slow, fast = (left, right) if left["expert_ms"] >= right["expert_ms"] else (right, left)
            ratio = slow["expert_ms"] / max(1e-6, fast["expert_ms"])
            if ratio < 1.10:
                continue
            output.append({
                "dataset": dataset, "layer": layer, "rank": rank,
                "slow_wave": slow["wave"], "fast_wave": fast["wave"],
                "pairs_slow": slow["full_pairs"], "pairs_fast": fast["full_pairs"],
                "expert_ms_slow": slow["expert_ms"], "expert_ms_fast": fast["expert_ms"],
                "latency_ratio": ratio,
                "active_experts_slow": slow["full_active_experts"], "active_experts_fast": fast["full_active_experts"],
                "rows_p50_slow": slow["full_rows_p50"], "rows_p50_fast": fast["full_rows_p50"],
                "tiny_le4_slow": slow["full_tiny_le4"], "tiny_le4_fast": fast["full_tiny_le4"],
            })
    return sorted(output, key=lambda row: row["latency_ratio"], reverse=True)


def oracle_analysis(rank_rows: list[dict], models, clean_medians: dict[str, float]):
    model, features = models[("expert_ms", "M1_shape", "raw")]
    robust_model, robust_features = models[("expert_ms", "M1_shape", "group_p99_trimmed")]
    if features != robust_features:
        raise RuntimeError("raw and robust shape models use different features")
    envelope_groups = defaultdict(list)
    for row in rank_rows:
        envelope_groups[(row["dataset"], row["layer"], row["rank"])].append(
            (float(row["full_pairs"]), float(row["expert_ms"]))
        )
    envelope_arrays = {}
    robust_envelope_arrays = {}
    group_p99 = {}
    for key, values in envelope_groups.items():
        values.sort()
        envelope_arrays[key] = (
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
        )
        threshold = float(np.quantile([value[1] for value in values], 0.99))
        group_p99[key] = threshold
        robust_values = [value for value in values if value[1] <= threshold]
        robust_envelope_arrays[key] = (
            np.asarray([value[0] for value in robust_values]),
            np.asarray([value[1] for value in robust_values]),
        )

    def envelope(row: dict, pairs_value: float, robust: bool = False) -> float:
        arrays = robust_envelope_arrays if robust else envelope_arrays
        pairs, times = arrays[(row["dataset"], row["layer"], row["rank"])]
        position = int(np.searchsorted(pairs, pairs_value))
        lo = max(0, position - 12)
        hi = min(len(pairs), position + 12)
        if hi - lo < 8:
            lo = max(0, hi - 8)
            hi = min(len(pairs), lo + 8)
        return float(np.quantile(times[lo:hi], 0.10))

    compact_vectors = []
    for row in rank_rows:
        row["fragment_ideal_expert_ms"] = envelope(row, float(row["full_pairs"]))
        row["fragment_ideal_expert_ms_p99_trimmed"] = envelope(row, float(row["full_pairs"]), robust=True)
        row["expert_ms_p99_winsorized"] = min(
            float(row["expert_ms"]),
            group_p99[(row["dataset"], row["layer"], row["rank"])],
        )
        vector = []
        for feature in features:
            if feature.startswith("full_"):
                vector.append(float(row["compact_" + feature[5:]]))
            else:
                vector.append(float(row[feature]))
        compact_vectors.append(vector)
    compact_predictions = model.predict(np.asarray(compact_vectors, dtype=float))
    robust_compact_predictions = robust_model.predict(np.asarray(compact_vectors, dtype=float))
    for row, prediction, robust_prediction in zip(rank_rows, compact_predictions, robust_compact_predictions, strict=True):
        row["compact_predicted_expert_ms"] = 0.0 if row["compact_pairs"] == 0 else max(0.0, float(prediction))
        row["compact_ideal_expert_ms"] = 0.0 if row["compact_pairs"] == 0 else envelope(row, float(row["compact_pairs"]))
        row["compact_predicted_expert_ms_p99_trimmed"] = 0.0 if row["compact_pairs"] == 0 else max(0.0, float(robust_prediction))
        row["compact_ideal_expert_ms_p99_trimmed"] = 0.0 if row["compact_pairs"] == 0 else envelope(row, float(row["compact_pairs"]), robust=True)

    groups = defaultdict(list)
    for row in rank_rows:
        groups[(row["dataset"], row["wave"], row["layer"])].append(row)
    per_layer = []
    for (dataset, wave, layer), ranks in groups.items():
        actual = max(row["expert_ms"] for row in ranks)
        ideal = max(row["fragment_ideal_expert_ms"] for row in ranks)
        actual_robust = max(row["expert_ms_p99_winsorized"] for row in ranks)
        ideal_robust = max(row["fragment_ideal_expert_ms_p99_trimmed"] for row in ranks)
        compact = max(row["compact_predicted_expert_ms"] for row in ranks)
        compact_ideal = max(row["compact_ideal_expert_ms"] for row in ranks)
        compact_robust = max(row["compact_predicted_expert_ms_p99_trimmed"] for row in ranks)
        compact_ideal_robust = max(row["compact_ideal_expert_ms_p99_trimmed"] for row in ranks)
        per_layer.append({
            "dataset": dataset, "wave": wave, "layer": layer,
            "actual_expert_critical_ms": actual,
            "fragment_ideal_expert_critical_ms": ideal,
            "fragment_removable_ms": max(0.0, actual - ideal),
            "actual_expert_critical_ms_p99_winsorized": actual_robust,
            "fragment_ideal_expert_critical_ms_p99_trimmed": ideal_robust,
            "fragment_removable_ms_p99_trimmed": max(0.0, actual_robust - ideal_robust),
            "compact_predicted_expert_critical_ms": compact,
            "compact_ideal_expert_critical_ms": compact_ideal,
            "post_compaction_fragment_residual_ms": max(0.0, compact - compact_ideal),
            "compaction_expert_work_removal_ms": max(0.0, actual - compact),
            "compact_predicted_expert_critical_ms_p99_trimmed": compact_robust,
            "compact_ideal_expert_critical_ms_p99_trimmed": compact_ideal_robust,
            "post_compaction_fragment_residual_ms_p99_trimmed": max(0.0, compact_robust - compact_ideal_robust),
            "compaction_expert_work_removal_ms_p99_trimmed": max(0.0, actual_robust - compact_robust),
        })
    oracle_rows = []
    for dataset in sorted(clean_medians):
        subset = [row for row in per_layer if row["dataset"] == dataset]
        wall_ms = clean_medians[dataset] * 1000
        fragment = sum(row["fragment_removable_ms"] for row in subset)
        robust_fragment = sum(row["fragment_removable_ms_p99_trimmed"] for row in subset)
        compaction = sum(row["compaction_expert_work_removal_ms"] for row in subset)
        robust_compaction = sum(row["compaction_expert_work_removal_ms_p99_trimmed"] for row in subset)
        residual = sum(row["post_compaction_fragment_residual_ms"] for row in subset)
        robust_residual = sum(row["post_compaction_fragment_residual_ms_p99_trimmed"] for row in subset)
        oracle_rows.extend([
            {"dataset": dataset, "candidate": "perfect_current_fragmentation_removal", "oracle_ms": fragment, "clean_e2e_percent": 100 * fragment / wall_ms, "evidence": "measured expert latency lower-envelope; optimistic"},
            {"dataset": dataset, "candidate": "perfect_current_fragmentation_removal_p99_trimmed", "oracle_ms": robust_fragment, "clean_e2e_percent": 100 * robust_fragment / wall_ms, "evidence": "group-p99-winsorized expert latency and trimmed lower-envelope; robust optimistic"},
            {"dataset": dataset, "candidate": "hypothetical_live_row_compaction_expert_only", "oracle_ms": compaction, "clean_e2e_percent": 100 * compaction / wall_ms, "evidence": "model-based sensitivity; not measured Epoch"},
            {"dataset": dataset, "candidate": "hypothetical_live_row_compaction_expert_only_p99_trimmed", "oracle_ms": robust_compaction, "clean_e2e_percent": 100 * robust_compaction / wall_ms, "evidence": "p99-trimmed shape-model sensitivity; not measured Epoch"},
            {"dataset": dataset, "candidate": "post_compaction_fragmentation_residual", "oracle_ms": residual, "clean_e2e_percent": 100 * residual / wall_ms, "evidence": "shape-model minus count-matched lower envelope; optimistic sensitivity"},
            {"dataset": dataset, "candidate": "post_compaction_fragmentation_residual_p99_trimmed", "oracle_ms": robust_residual, "clean_e2e_percent": 100 * robust_residual / wall_ms, "evidence": "p99-trimmed shape-model minus trimmed lower envelope; robust optimistic sensitivity"},
            {"dataset": dataset, "candidate": "feasible_tiny_shape_specialization_50pct_capture", "oracle_ms": 0.5 * residual, "clean_e2e_percent": 50 * residual / wall_ms, "evidence": "bounded feasibility assumption, no implementation"},
            {"dataset": dataset, "candidate": "feasible_tiny_shape_specialization_50pct_capture_p99_trimmed", "oracle_ms": 0.5 * robust_residual, "clean_e2e_percent": 50 * robust_residual / wall_ms, "evidence": "50% capture of robust optimistic post-compaction residual; no implementation"},
        ])
    return per_layer, oracle_rows


def aggregate_temporal(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for layer in REP_LAYERS:
            for phase_name in ("early", "middle", "late"):
                subset = [row for row in rows if row["dataset"] == dataset and row["layer"] == layer and row["phase"] == phase_name and row.get("hidden_matched_rows")]
                out = {"dataset": dataset, "layer": layer, "phase": phase_name, "observations": len(subset)}
                for metric in (
                    "hidden_cosine_mean", "hidden_rel_l2_mean", "hidden_near_equal_fraction",
                    "hidden_live_cosine_mean", "hidden_live_rel_l2_mean",
                    "moe_output_cosine_mean", "moe_output_rel_l2_mean", "moe_output_near_equal_fraction",
                    "moe_output_live_cosine_mean", "moe_output_live_rel_l2_mean",
                ):
                    values = [float(row[metric]) for row in subset if row.get(metric) is not None]
                    out[f"{metric}_median"] = float(np.median(values)) if values else math.nan
                output.append(out)
    return output


def plot_results(layer_rows, route_rows, matched, oracle_rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_order = ("early", "middle", "late")

    def phase_plot(metric, ylabel, filename):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for dataset, marker in (("gsm8k", "o"), ("humaneval", "s")):
            values = [np.median([row[metric] for row in layer_rows if row["dataset"] == dataset and row["phase"] == p]) for p in phase_order]
            ax.plot(phase_order, values, marker=marker, label=dataset)
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / filename, dpi=180); plt.close(fig)

    phase_plot("live_ratio", "decision-live / physical rows", "01_live_fraction_vs_phase.png")
    phase_plot("full_active_experts", "active routed experts", "02_active_experts_vs_phase.png")
    phase_plot("full_rows_p50", "p50 rows / active expert", "03_rows_per_expert_vs_phase.png")
    phase_plot("full_tiny_le4", "fraction active experts <=4 rows", "04_tiny_expert_fraction_vs_phase.png")
    phase_plot("token_confidence_live_mean", "live-token confidence", "05_token_confidence_vs_phase.png")
    phase_plot("router_entropy_live_mean", "router entropy (live rows)", "06_router_entropy_vs_phase.png")
    phase_plot("topk_mass_live_mean", "top-k probability mass (live rows)", "07_topk_mass_vs_phase.png")
    phase_plot("compact_rows_p50", "post-compaction p50 rows / expert", "11_post_compaction_rows_per_expert.png")
    phase_plot("compact_tiny_le4", "post-compaction expert fraction <=4 rows", "12_post_compaction_tiny_fraction.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for dataset, color in (("gsm8k", "tab:blue"), ("humaneval", "tab:orange")):
        subset = [row for row in layer_rows if row["dataset"] == dataset]
        ax.scatter([row["full_pairs"] for row in subset], [row["expert_critical_ms"] for row in subset], s=4, alpha=.18, color=color, label=dataset)
    ax.set_xlabel("token-expert pairs"); ax.set_ylabel("critical-rank expert latency (ms)"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "08_expert_latency_vs_pairs.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for dataset, color in (("gsm8k", "tab:blue"), ("humaneval", "tab:orange")):
        subset = [row for row in layer_rows if row["dataset"] == dataset]
        ax.scatter([row["full_tiny_le4"] for row in subset], [row["expert_critical_ms"] for row in subset], s=4, alpha=.18, color=color, label=dataset)
    ax.set_xlabel("active experts <=4 rows"); ax.set_ylabel("critical-rank expert latency (ms)"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "09_expert_latency_vs_tiny_fraction.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ratios = [row["latency_ratio"] for row in matched[:100]]
    ax.hist(ratios, bins=25); ax.set_xlabel("same-work adjacent-pair expert latency ratio"); ax.set_ylabel("matched pairs"); fig.tight_layout(); fig.savefig(output_dir / "10_same_work_different_latency.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for dataset, marker in (("gsm8k", "o"), ("humaneval", "s")):
        subset = [row for row in route_rows if row["dataset"] == dataset and row["matched_live_rows"]]
        ax.scatter([row["live_topk_overlap_fraction_lag1"] for row in subset], [row["live_set_exact_fraction_lag1"] for row in subset], s=5, alpha=.25, marker=marker, label=dataset)
    ax.set_xlabel("lag-1 top-k overlap fraction"); ax.set_ylabel("lag-1 exact set fraction"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "13_route_overlap_vs_exact.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    candidates = [
        "perfect_current_fragmentation_removal_p99_trimmed",
        "hypothetical_live_row_compaction_expert_only_p99_trimmed",
        "post_compaction_fragmentation_residual_p99_trimmed",
        "feasible_tiny_shape_specialization_50pct_capture_p99_trimmed",
    ]
    x = np.arange(len(candidates)); width = .36
    for offset, dataset in ((-.18, "gsm8k"), (.18, "humaneval")):
        values = [next(row["clean_e2e_percent"] for row in oracle_rows if row["dataset"] == dataset and row["candidate"] == candidate) for candidate in candidates]
        ax.bar(x + offset, values, width, label=dataset)
    ax.set_xticks(x, [value.replace("_", "\n") for value in candidates], fontsize=7); ax.set_ylabel("clean request E2E oracle (%)"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "14_candidate_oracles.png", dpi=180); plt.close(fig)


def main() -> None:
    clean_rows, clean_medians = clean_summary()
    write_csv(ROOT / "BASELINE_CLEAN.csv", clean_rows)

    configs = {
        "gsm8k": {
            "shape_dir": ROOT / "results/raw/shape/gsm8k/r3",
            "shape_log": ROOT / "logs/trace_shape_gsm8k_ep4_b32_mini32_r3_g32.log",
            "stage_dir": ROOT / "results/raw/stage/gsm8k/r2",
            "stage_log": ROOT / "logs/trace_stage_gsm8k_ep4_b32_mini32_r2_g32.log",
            "temporal_dir": ROOT / "results/raw/temporal/gsm8k/r2",
            "temporal_log": ROOT / "logs/trace_temporal_gsm8k_ep4_b32_mini32_r2_g32.log",
        },
        "humaneval": {
            "shape_dir": ROOT / "results/raw/shape/humaneval/r2",
            "shape_log": ROOT / "logs/trace_shape_humaneval_ep4_b32_mini32_r2_g32.log",
            "stage_dir": ROOT / "results/raw/stage/humaneval/r2",
            "stage_log": ROOT / "logs/trace_stage_humaneval_ep4_b32_mini32_r2_g32.log",
            "temporal_dir": ROOT / "results/raw/temporal/humaneval/r1",
            "temporal_log": ROOT / "logs/trace_temporal_humaneval_ep4_b32_mini32_r1_g32.log",
        },
    }
    shape_layers, shape_ranks, route_rows = [], [], []
    stage_layers, stage_ranks, temporal_rows = [], [], []
    identity_checks = []
    for dataset, cfg in configs.items():
        layers, ranks, routes, shape_logical = process_shape(dataset, cfg["shape_dir"], cfg["shape_log"])
        stages, stage_rank, stage_logical, stage_final = process_stage(dataset, cfg["stage_dir"], cfg["stage_log"])
        identity = all(
            left["sequence_ids"] == right["sequence_ids"]
            and left["block_starts"] == right["block_starts"]
            and left["remaining_before"] == right["remaining_before"]
            for left, right in zip(shape_logical, stage_logical, strict=True)
        )
        identity_checks.append({"dataset": dataset, "waves": len(shape_logical), "shape_stage_trajectory_exact": identity, "stage_trace_wall_seconds_observer_heavy": stage_final["wall_seconds"]})
        if not identity:
            raise RuntimeError(f"shape and stage trajectories diverged for {dataset}")
        shape_layers += layers; shape_ranks += ranks; route_rows += routes
        stage_layers += stages; stage_ranks += stage_rank
        temporal_rows += process_temporal(dataset, cfg["temporal_dir"], cfg["temporal_log"])

    layer_rows, rank_rows = join_rows(shape_layers, stage_layers, shape_ranks, stage_ranks)
    summaries = phase_summary(layer_rows)
    predictor_rows, correlation_rows, models = fit_predictors(rank_rows)
    matches = matched_pairs(rank_rows)
    oracle_layer, oracle_rows = oracle_analysis(rank_rows, models, clean_medians)
    temporal_summary = aggregate_temporal(temporal_rows)

    write_csv(ROOT / "TRACE_IDENTITY_CHECKS.csv", identity_checks)
    write_csv(ROOT / "LAYER_WAVE_METRICS.csv", layer_rows)
    write_csv(ROOT / "RANK_LAYER_METRICS.csv", rank_rows)
    write_csv(ROOT / "PHASE_MISMATCH_SUMMARY.csv", summaries)
    write_csv(ROOT / "TEMPORAL_ROUTE_STABILITY.csv", route_rows)
    write_csv(ROOT / "TEMPORAL_HIDDEN_OUTPUT.csv", temporal_rows)
    write_csv(ROOT / "TEMPORAL_HIDDEN_OUTPUT_SUMMARY.csv", temporal_summary)
    write_csv(ROOT / "PREDICTOR_RESULTS.csv", predictor_rows)
    write_csv(ROOT / "PREDICTOR_CORRELATIONS.csv", correlation_rows)
    write_csv(ROOT / "MATCHED_SAME_WORK_PAIRS.csv", matches[:1000])
    write_csv(ROOT / "FRAGMENTATION_ORACLE_BY_LAYER.csv", oracle_layer)
    write_csv(ROOT / "CANDIDATE_ORACLES.csv", oracle_rows)
    plot_results(layer_rows, route_rows, matches, oracle_rows, ROOT / "figures")

    summary = {
        "clean_median_seconds": clean_medians,
        "trace_identity": identity_checks,
        "phase_summary": summaries,
        "predictors": predictor_rows,
        "oracle": oracle_rows,
        "matched_pair_count": len(matches),
    }
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "clean_medians": clean_medians,
        "shape_layer_observations": len(layer_rows),
        "rank_layer_observations": len(rank_rows),
        "matched_pairs": len(matches),
        "oracles": oracle_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
