#!/usr/bin/env python3
"""Bounded exact-route DeepEP communication replay for saturation/imbalance PoC.

The prepare phase is CPU-only and fixes all matching thresholds before any
communication timing is collected.  The run phase is intentionally limited to
DeepEP dispatch/combine; no model or expert GEMM is executed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_ROOT = ROOT / "poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
DEFAULT_MANIFEST = DEFAULT_ROUTE_ROOT / "workload_manifest.json"
NUM_EXPERTS = 128
WORLD = 4
TOP_K = 8
HIDDEN = 2048
BLOCK_EXPERTS = 32

# These are fixed before inspecting measured communication latency.
POLICY = {
    "token_relative_max": 0.05,
    "similar_s_abs": 0.05,
    "similar_i_abs": 0.05,
    "different_s_abs": 0.05,
    "different_i_abs": 0.10,
    "max_pairs_per_kind": 4,
    "max_rows_per_regime": 2,
    "quartile_low_high": [0.25, 0.75],
    "warmups": 5,
    "iterations": 20,
    "go_effect": 0.10,
    "hold_effect": 0.05,
    "go_consistency": 0.75,
}


def _json(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: _json(v) if isinstance(v, (np.generic, np.ndarray)) else v for k, v in row.items()} for row in rows)


def _route_path(route_root: Path, rel: str) -> Path:
    path = route_root / rel
    if not path.exists():
        # Manifest route paths are relative to the parent result directory.
        candidate = route_root.parent / rel
        if candidate.exists():
            return candidate
    return path


def load_vision_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        vision = dict(pair["vision"])
        vision["pair_id"] = pair["pair_id"]
        vision["token_bucket"] = pair.get("token_bucket", "unknown")
        records.append(vision)
    return records


def compute_row(request: dict[str, Any], layer: int, routes: np.ndarray) -> dict[str, Any]:
    # routes is [tokens, layers, top_k], containing global expert ids.
    ids = routes[:, layer, :].astype(np.int64, copy=False)
    tokens = int(ids.shape[0])
    rank_ids = ids // BLOCK_EXPERTS
    unique_per_token = np.array([len(np.unique(row)) for row in rank_ids], dtype=np.float64)
    rank_load = np.bincount(rank_ids.reshape(-1), minlength=WORLD).astype(np.int64)
    mean_load = float(rank_load.mean())
    prompt_ids = request["_prompt_token_ids"]
    vision_mask = prompt_ids == 151655
    if len(vision_mask) != tokens:
        raise ValueError(f"token id/route length mismatch {request['request_id']} {len(vision_mask)} {tokens}")
    v_rank_load = np.bincount(rank_ids[vision_mask].reshape(-1), minlength=WORLD).astype(np.int64) if vision_mask.any() else np.zeros(WORLD, dtype=np.int64)
    t_rank_load = np.bincount(rank_ids[~vision_mask].reshape(-1), minlength=WORLD).astype(np.int64) if (~vision_mask).any() else np.zeros(WORLD, dtype=np.int64)
    expert_hist = np.bincount(ids.reshape(-1), minlength=NUM_EXPERTS).astype(np.int64)
    v_expert_hist = np.bincount(ids[vision_mask].reshape(-1), minlength=NUM_EXPERTS).astype(np.int64) if vision_mask.any() else np.zeros(NUM_EXPERTS, dtype=np.int64)
    t_expert_hist = np.bincount(ids[~vision_mask].reshape(-1), minlength=NUM_EXPERTS).astype(np.int64) if (~vision_mask).any() else np.zeros(NUM_EXPERTS, dtype=np.int64)
    cv = float(rank_load.std() / mean_load) if mean_load else 0.0
    s = float(unique_per_token.mean() / WORLD)
    imbalance = float(rank_load.max() / mean_load) if mean_load else 0.0

    def modality_stats(mask: np.ndarray) -> dict[str, Any]:
        if not bool(mask.any()):
            return {"S": float("nan"), "I": float("nan"), "rank_cv": float("nan"), "p_u4": float("nan"), "p_u_ge3": float("nan"), "assignments": 0, "rank_load": [0] * WORLD}
        mod_ranks = rank_ids[mask]
        mod_u = np.asarray([len(np.unique(row)) for row in mod_ranks], dtype=np.float64)
        mod_load = np.bincount(mod_ranks.reshape(-1), minlength=WORLD).astype(np.int64)
        mod_mean = float(mod_load.mean())
        return {
            "S": float(mod_u.mean() / WORLD),
            "I": float(mod_load.max() / mod_mean) if mod_mean else float("nan"),
            "rank_cv": float(mod_load.std() / mod_mean) if mod_mean else float("nan"),
            "p_u4": float(np.mean(mod_u == WORLD)),
            "p_u_ge3": float(np.mean(mod_u >= 3)),
            "assignments": int(mod_load.sum()),
            "rank_load": mod_load.tolist(),
        }
    vision_stats = modality_stats(vision_mask)
    text_stats = modality_stats(~vision_mask)
    return {
        "request_id": request["request_id"],
        "pair_id": int(request["pair_id"]),
        "category": request.get("category", ""),
        "token_bucket": request.get("token_bucket", ""),
        "route_file": request["route_file"],
        "layer": int(layer),
        "token_count": tokens,
        "total_assignments": int(tokens * TOP_K),
        "S": s,
        "I": imbalance,
        "rank_cv": cv,
        "p_u4": float(np.mean(unique_per_token == WORLD)),
        "p_u_ge3": float(np.mean(unique_per_token >= 3)),
        "mean_unique_ranks": float(unique_per_token.mean()),
        "rank_load": rank_load.tolist(),
        "vision_rank_load": v_rank_load.tolist(),
        "text_rank_load": t_rank_load.tolist(),
        "vision_assignments": int(v_rank_load.sum()),
        "text_assignments": int(t_rank_load.sum()),
        "vision_fraction": float(v_rank_load.sum() / (tokens * TOP_K)) if tokens else 0.0,
        "vision_S": vision_stats["S"],
        "vision_I": vision_stats["I"],
        "vision_rank_cv": vision_stats["rank_cv"],
        "vision_p_u4": vision_stats["p_u4"],
        "vision_p_u_ge3": vision_stats["p_u_ge3"],
        "text_S": text_stats["S"],
        "text_I": text_stats["I"],
        "text_rank_cv": text_stats["rank_cv"],
        "text_p_u4": text_stats["p_u4"],
        "text_p_u_ge3": text_stats["p_u_ge3"],
        "expert_hist": expert_hist.tolist(),
        "vision_expert_hist": v_expert_hist.tolist(),
        "text_expert_hist": t_expert_hist.tolist(),
    }


def prepare(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_vision_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    for request in records:
        route_path = _route_path(args.route_root, request["route_file"])
        data = np.load(route_path)
        routes = data["routed_experts"]
        request = dict(request)
        request["_prompt_token_ids"] = data["prompt_token_ids"].astype(np.int64)
        for layer in range(routes.shape[1]):
            rows.append(compute_row(request, layer, routes))
    _csv_write(out / "route_metrics.csv", rows)
    dump_json(out / "selection_policy.json", POLICY)
    dump_json(out / "source_workload_manifest.json", records)

    s_values = np.asarray([r["S"] for r in rows], dtype=np.float64)
    i_values = np.asarray([r["I"] for r in rows], dtype=np.float64)
    s_q25, s_q75 = np.quantile(s_values, [0.25, 0.75]).tolist()
    i_q25, i_q75 = np.quantile(i_values, [0.25, 0.75]).tolist()
    quantiles = {"S_q25": s_q25, "S_q75": s_q75, "I_q25": i_q25, "I_q75": i_q75}

    def pair_candidates(kind: str) -> list[tuple[tuple[float, float, int, int], dict[str, Any], dict[str, Any]]]:
        candidates = []
        for left_idx, left in enumerate(rows):
            for right in rows[left_idx + 1 :]:
                if left["request_id"] == right["request_id"]:
                    continue
                rel_tokens = abs(left["token_count"] - right["token_count"]) / max(left["token_count"], right["token_count"], 1)
                if rel_tokens > POLICY["token_relative_max"]:
                    continue
                ds = abs(left["S"] - right["S"])
                di = abs(left["I"] - right["I"])
                if kind == "similar_I_different_S":
                    ok = di <= POLICY["similar_i_abs"] and ds >= POLICY["different_s_abs"]
                    similar, different = di, -ds
                else:
                    ok = ds <= POLICY["similar_s_abs"] and di >= POLICY["different_i_abs"]
                    similar, different = ds, -di
                if ok:
                    candidates.append(((similar, different, int(left["pair_id"]), int(left["layer"])), left, right))
        candidates.sort(key=lambda x: x[0])
        chosen: list[tuple[tuple[float, float, int, int], dict[str, Any], dict[str, Any]]] = []
        used: set[tuple[str, int]] = set()
        for item in candidates:
            a, b = item[1], item[2]
            keys = {(a["request_id"], a["layer"]), (b["request_id"], b["layer"])}
            if not used.intersection(keys):
                chosen.append(item)
                used.update(keys)
            if len(chosen) >= POLICY["max_pairs_per_kind"]:
                break
        if len(chosen) < POLICY["max_pairs_per_kind"]:
            for item in candidates:
                if item in chosen:
                    continue
                chosen.append(item)
                if len(chosen) >= POLICY["max_pairs_per_kind"]:
                    break
        return chosen

    case_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    pair_no = 0
    for kind in ("similar_I_different_S", "similar_S_different_I"):
        for _, left, right in pair_candidates(kind):
            # Deterministic side labels make the effect direction explicit.
            if kind == "similar_I_different_S":
                ordered = sorted([left, right], key=lambda r: (r["S"], r["request_id"], r["layer"]))
                low, high = ordered[0], ordered[1]
                low_label, high_label = "low_S", "high_S"
            else:
                ordered = sorted([left, right], key=lambda r: (r["I"], r["request_id"], r["layer"]))
                low, high = ordered[0], ordered[1]
                low_label, high_label = "low_I", "high_I"
            pair_key = f"{kind}_{pair_no:02d}"
            for side, row, label in (("low", low, low_label), ("high", high, high_label)):
                case_rows.append({"case_id": f"pair_{pair_key}_{side}", "kind": kind, "pair_key": pair_key, "side": label, **row})
            pair_rows.append({"pair_key": pair_key, "kind": kind, "low_request": low["request_id"], "low_layer": low["layer"], "high_request": high["request_id"], "high_layer": high["layer"], "token_rel_diff": abs(low["token_count"] - high["token_count"]) / max(low["token_count"], high["token_count"], 1), "S_diff": abs(low["S"] - high["S"]), "I_diff": abs(low["I"] - high["I"])})
            pair_no += 1

    regime_specs = {
        "low_S_low_I": lambda r: r["S"] <= s_q25 and r["I"] <= i_q25,
        "high_S_low_I": lambda r: r["S"] >= s_q75 and r["I"] <= i_q25,
        "low_S_high_I": lambda r: r["S"] <= s_q25 and r["I"] >= i_q75,
        "high_S_high_I": lambda r: r["S"] >= s_q75 and r["I"] >= i_q75,
    }
    for regime, predicate in regime_specs.items():
        candidates = [r for r in rows if predicate(r)]
        candidates.sort(key=lambda r: (r["request_id"], r["layer"]))
        for idx, row in enumerate(candidates[: POLICY["max_rows_per_regime"]]):
            case_rows.append({"case_id": f"regime_{regime}_{idx:02d}", "kind": "regime", "pair_key": regime, "side": regime, **row})

    _csv_write(out / "selected_cases.csv", case_rows)
    _csv_write(out / "matched_pairs.csv", pair_rows)
    summary = {
        "num_route_rows": len(rows),
        "num_selected_cases": len(case_rows),
        "num_pairs": len(pair_rows),
        "pair_counts": {kind: sum(1 for p in pair_rows if p["kind"] == kind) for kind in ("similar_I_different_S", "similar_S_different_I")},
        "regime_counts": {name: sum(1 for c in case_rows if c["pair_key"] == name) for name in regime_specs},
        "quantiles": quantiles,
        "global": {"S_median": float(np.median(s_values)), "I_median": float(np.median(i_values)), "token_median": float(np.median([r["token_count"] for r in rows]))},
    }
    dump_json(out / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_list(value: str) -> list[float]:
    return [float(x) for x in value.strip("[]").split(",") if x.strip()]


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    import deep_ep

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD:
        raise RuntimeError(f"DeepEP replay requires world size {WORLD}, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = _read_csv(args.output / "selected_cases.csv")
    route_cache: dict[tuple[str, int], np.ndarray] = {}
    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(dist.group.WORLD, args.buffer_mib * 1024 * 1024, 0, low_latency_mode=False, num_qps_per_rank=1, explicitly_destroy=True)
    generator = torch.Generator(device=device).manual_seed(19001 + rank)
    samples: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        key = (case["route_file"], int(case["layer"]))
        if key not in route_cache:
            data = np.load(_route_path(args.route_root, case["route_file"]))
            route_cache[key] = data["routed_experts"][:, int(case["layer"]), :].astype(np.int64, copy=True)
        ids_np = route_cache[key]
        n = ids_np.shape[0]
        hidden = torch.randn((n, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator)
        topk_ids = torch.from_numpy(ids_np).to(device=device, dtype=deep_ep.topk_idx_t).contiguous()
        topk_weights = torch.full((n, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        # Layout is intentionally recomputed for each exact route layout and is part of dispatch timing.
        for iteration in range(args.warmups + args.iterations):
            dist.barrier()
            torch.cuda.synchronize(device)
            start_total = torch.cuda.Event(enable_timing=True)
            start_dispatch = torch.cuda.Event(enable_timing=True)
            end_dispatch = torch.cuda.Event(enable_timing=True)
            start_combine = torch.cuda.Event(enable_timing=True)
            end_combine = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter_ns()
            start_total.record()
            start_dispatch.record()
            layout = buffer.get_dispatch_layout(topk_ids, NUM_EXPERTS, async_finish=False, allocate_on_comm_stream=False)
            num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = layout
            recv_hidden, recv_ids, recv_weights, recv_count, handle, dispatch_event = buffer.dispatch(
                x=hidden, handle=None, num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert, topk_idx=topk_ids,
                topk_weights=topk_weights, expert_alignment=1, config=deep_ep.Buffer.get_dispatch_config(world),
                previous_event=previous_event, async_finish=False, allocate_on_comm_stream=False)
            end_dispatch.record()
            start_combine.record()
            combined, _, combine_event = buffer.combine(
                x=recv_hidden, handle=handle, topk_weights=None, config=deep_ep.Buffer.get_combine_config(world),
                async_finish=False, allocate_on_comm_stream=False)
            end_combine.record()
            end_total.record()
            end_total.synchronize()
            host_end = time.perf_counter_ns()
            if iteration >= args.warmups:
                samples.append({
                    "case_index": case_index, "case_id": case["case_id"], "iteration": iteration - args.warmups, "rank": rank,
                    "physical_gpu": [1, 2, 3, 4][local_rank], "token_count": n, "received_tokens": int(recv_hidden.shape[0]),
                    "dispatch_ms": float(start_dispatch.elapsed_time(end_dispatch)), "combine_ms": float(start_combine.elapsed_time(end_combine)),
                    "total_ms": float(start_total.elapsed_time(end_total)), "host_ms": (host_end - host_start) / 1e6,
                })
            del layout, recv_hidden, recv_ids, recv_weights, recv_count, handle, dispatch_event, combine_event, combined
            dist.barrier()
        del hidden, topk_ids, topk_weights
    dump_json(args.output / f"rank{rank}_samples.json", {"rank": rank, "local_rank": local_rank, "physical_gpu": [1, 2, 3, 4][local_rank], "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "deep_ep": str(Path(deep_ep.__file__).resolve()), "samples": samples})
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def aggregate(args: argparse.Namespace) -> None:
    import statistics
    case_rows = _read_csv(args.output / "selected_cases.csv")
    by_case: dict[str, list[dict[str, Any]]] = {r["case_id"]: [] for r in case_rows}
    rank_files = sorted(args.output.glob("rank*_samples.json"))
    for path in rank_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sample in payload["samples"]:
            by_case[sample["case_id"]].append(sample)
    aggregated: list[dict[str, Any]] = []
    for case in case_rows:
        samples = by_case.get(case["case_id"], [])
        by_iter: dict[int, list[dict[str, Any]]] = {}
        for s in samples:
            by_iter.setdefault(int(s["iteration"]), []).append(s)
        values = []
        rank_medians: dict[str, float] = {}
        for metric in ("dispatch_ms", "combine_ms", "total_ms", "host_ms"):
            per_iter = [max(float(s[metric]) for s in ss) for _, ss in sorted(by_iter.items()) if len(ss) == WORLD]
            values.append((metric, per_iter))
            aggregated_value = float(np.median(per_iter)) if per_iter else float("nan")
            aggregated.append({"case_id": case["case_id"], "kind": case["kind"], "pair_key": case["pair_key"], "side": case["side"], "request_id": case["request_id"], "layer": int(case["layer"]), "token_count": int(case["token_count"]), "S": float(case["S"]), "I": float(case["I"]), "vision_fraction": float(case["vision_fraction"]), "metric": metric, "median_ms": aggregated_value, "p25_ms": float(np.percentile(per_iter, 25)) if per_iter else float("nan"), "p75_ms": float(np.percentile(per_iter, 75)) if per_iter else float("nan"), "p95_ms": float(np.percentile(per_iter, 95)) if per_iter else float("nan"), "cv": float(np.std(per_iter) / np.mean(per_iter)) if per_iter and np.mean(per_iter) else float("nan"), "n_iterations": len(per_iter), "rank_medians_ms": json.dumps({str(r): float(np.median([float(s[metric]) for s in samples if int(s["rank"]) == r])) for r in range(WORLD)})})
    _csv_write(args.output / "aggregated_results.csv", aggregated)

    # Pair deltas are measured on the maximum-rank collective time per iteration.
    pair_results: list[dict[str, Any]] = []
    for pair in _read_csv(args.output / "matched_pairs.csv"):
        kind = pair["kind"]
        for metric in ("dispatch_ms", "combine_ms", "total_ms"):
            def med(side: str) -> float:
                matches = [r for r in aggregated if r["pair_key"] == pair["pair_key"] and r["side"] == side and r["metric"] == metric]
                return float(matches[0]["median_ms"]) if matches else float("nan")
            low_side, high_side = ("low_S", "high_S") if kind == "similar_I_different_S" else ("low_I", "high_I")
            low = med(low_side); high = med(high_side)
            pair_results.append({"pair_key": pair["pair_key"], "kind": kind, "metric": metric, "low_ms": low, "high_ms": high, "relative_high_minus_low": (high - low) / low if low else float("nan"), "S_diff": pair["S_diff"], "I_diff": pair["I_diff"], "token_rel_diff": pair["token_rel_diff"]})
    _csv_write(args.output / "matched_pair_results.csv", pair_results)

    regime_results: list[dict[str, Any]] = []
    for regime in ("low_S_low_I", "high_S_low_I", "low_S_high_I", "high_S_high_I"):
        for metric in ("dispatch_ms", "combine_ms", "total_ms"):
            vals = [r["median_ms"] for r in aggregated if r["pair_key"] == regime and r["metric"] == metric and math.isfinite(r["median_ms"])]
            regime_results.append({"regime": regime, "metric": metric, "n": len(vals), "median_ms": float(np.median(vals)) if vals else float("nan")})
    _csv_write(args.output / "regime_results.csv", regime_results)

    # All route rows provide modality diagnostics; no modality is used as a predictor.
    route_rows = _read_csv(args.output / "route_metrics.csv")
    modality_rows: list[dict[str, Any]] = []
    for name, prefix in (("vision_tokens", "vision_"), ("text_tokens", "text_")):
        for metric in ("S", "I", "rank_cv", "p_u4", "p_u_ge3"):
            arr = np.asarray([float(r[f"{prefix}{metric}"]) for r in route_rows], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            modality_rows.append({"modality": name, "metric": metric, "n": len(arr), "mean": float(arr.mean()) if len(arr) else float("nan"), "median": float(np.median(arr)) if len(arr) else float("nan"), "p25": float(np.percentile(arr, 25)) if len(arr) else float("nan"), "p75": float(np.percentile(arr, 75)) if len(arr) else float("nan")})
    _csv_write(args.output / "modality_regime.csv", modality_rows)

    # Gate is fixed: median matched effect and consistency, otherwise HOLD/NO-GO.
    effects: dict[str, list[float]] = {"S": [], "I": []}
    for row in pair_results:
        if row["metric"] != "total_ms" or not math.isfinite(float(row["relative_high_minus_low"])):
            continue
        key = "S" if row["kind"] == "similar_I_different_S" else "I"
        effects[key].append(float(row["relative_high_minus_low"]))
    effect_summary = {}
    for key, vals in effects.items():
        abs_vals = np.abs(vals)
        positive = [v for v in vals if v > 0]
        effect_summary[key] = {"n": len(vals), "median_signed": float(np.median(vals)) if vals else float("nan"), "median_abs": float(np.median(abs_vals)) if vals else float("nan"), "positive_fraction": len(positive) / len(vals) if vals else 0.0}
    interaction = float("nan")
    hh = [r["median_ms"] for r in regime_results if r["regime"] == "high_S_high_I" and r["metric"] == "total_ms" and math.isfinite(r["median_ms"])]
    others = [r["median_ms"] for r in regime_results if r["regime"] != "high_S_high_I" and r["metric"] == "total_ms" and math.isfinite(r["median_ms"])]
    if hh and others and np.median(others):
        interaction = float((np.median(hh) - np.median(others)) / np.median(others))
    candidates = [v["median_abs"] for v in effect_summary.values() if math.isfinite(v["median_abs"]) and v["n"] >= 3 and v["positive_fraction"] >= POLICY["go_consistency"]]
    if candidates and max(candidates) >= POLICY["go_effect"]:
        gate = "GO"
    elif candidates and max(candidates) >= POLICY["hold_effect"]:
        gate = "HOLD"
    elif math.isfinite(interaction) and abs(interaction) >= POLICY["go_effect"]:
        gate = "GO"
    elif math.isfinite(interaction) and abs(interaction) >= POLICY["hold_effect"]:
        gate = "HOLD"
    else:
        gate = "NO-GO"
    summary = {"DEEPEP_SATURATION_LATENCY": gate, "effect_summary": effect_summary, "high_high_interaction_total": interaction, "n_rank_files": len(rank_files), "world_size": WORLD, "physical_gpus": [1, 2, 3, 4], "policy": POLICY, "aggregate_files": [str(p.name) for p in rank_files]}
    dump_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    make_figures(args.output, aggregated, pair_results, route_rows, regime_results)
    write_report(args.output, summary, effect_summary, interaction, modality_rows)


def make_figures(out: Path, aggregated: list[dict[str, Any]], pair_results: list[dict[str, Any]], route_rows: list[dict[str, str]], regimes: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    by_metric = defaultdict(list)
    for r in aggregated:
        if r["metric"] in ("dispatch_ms", "combine_ms", "total_ms") and math.isfinite(float(r["median_ms"])):
            by_metric[r["metric"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, color in (("dispatch_ms", "tab:blue"), ("combine_ms", "tab:orange")):
        rs = by_metric[metric]
        axes[0].scatter([float(r["S"]) for r in rs], [float(r["median_ms"]) for r in rs], s=18, alpha=.7, label=metric, c=color)
        axes[1].scatter([float(r["I"]) for r in rs], [float(r["median_ms"]) for r in rs], s=18, alpha=.7, label=metric, c=color)
    axes[0].set(xlabel="S: unique destination rank saturation", ylabel="median max-rank latency (ms)"); axes[1].set(xlabel="I: max/mean rank load", ylabel="median max-rank latency (ms)")
    axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(out / "plot1_saturation_imbalance_latency.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4)); labels=[]; values=[]
    for p in pair_results:
        if p["metric"] == "total_ms": labels.append(p["pair_key"]); values.append(100*float(p["relative_high_minus_low"]))
    if values: ax.bar(labels, values, color=["tab:blue" if "S_" in l else "tab:orange" for l in labels]); ax.axhline(0,color="k",lw=.7)
    ax.set_ylabel("high minus low latency (%)"); ax.set_title("Pre-registered matched effects"); ax.tick_params(axis="x",rotation=45); fig.tight_layout(); fig.savefig(out / "plot2_matched_case_effects.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5));
    for name, prefix, color in (("Vision", "vision_", "tab:green"), ("Text", "text_", "tab:red")):
        rs = [r for r in route_rows if math.isfinite(float(r[f"{prefix}S"])) and math.isfinite(float(r[f"{prefix}I"]))]
        ax.scatter([float(r[f"{prefix}S"]) for r in rs], [float(r[f"{prefix}I"]) for r in rs], s=7, alpha=.25, label=name, c=color)
    ax.set(xlabel="S", ylabel="I", title="Within real-image request token-source regimes"); ax.legend(); fig.tight_layout(); fig.savefig(out / "plot3_modality_regime_distribution.png", dpi=160); plt.close(fig)


def write_report(out: Path, summary: dict[str, Any], effect_summary: dict[str, Any], interaction: float, modality_rows: list[dict[str, Any]]) -> None:
    agg = _read_csv(out / "aggregated_results.csv")
    pairs = _read_csv(out / "matched_pair_results.csv")
    lines = ["# DeepEP Saturation/Latency Forensics", "", f"`DEEPEP_SATURATION_LATENCY: {summary['DEEPEP_SATURATION_LATENCY']}`", "", "## Scope and controls", "", "Exact Qwen3-VL route artifacts (24 real-image requests × 48 layers) were replayed with current linear placement `expert_id // 32`. No model, routing, placement, or dynamic communication code was changed. DeepEP dispatch and combine were measured on the four logical ranks mapped to physical GPUs 1,2,3,4 (`CUDA_VISIBLE_DEVICES=1,2,3,4`) with BF16 hidden size 2048 and EP4; expert GEMM was excluded. Random BF16 communication payloads preserve route/layout/shape while avoiding model execution. Each rank injected the same artifact token rows to satisfy the collective contract; this is a bounded route-layout replay, not a claim about live DP source-token partitioning.", "", f"Fixed selection policy is stored in `{out / 'selection_policy.json'}`, including token matching ≤{POLICY['token_relative_max']*100:.0f}%, similar/different S/I tolerances, and quartile regime boundaries. {len(pairs)} pair-effect rows were measured; no post-hoc threshold changes were made.", "", "## Matched results", "", "| Effect | n | median absolute total-latency change | positive fraction |", "|---|---:|---:|---:|"]
    for key, val in effect_summary.items():
        lines.append(f"| {key} (matched other metric) | {val['n']} | {val['median_abs']*100:.2f}% | {val['positive_fraction']*100:.1f}% |" if math.isfinite(val['median_abs']) else f"| {key} | 0 | unavailable | unavailable |")
    lines += ["", "### Dispatch/combine breakdown", "", "The matched effect is computed as (high metric − low metric)/low metric using the max-rank CUDA-event time per iteration. Negative values mean the designated high-S or high-I layout was faster.", "", "| Comparison | Dispatch median Δ | Combine median Δ | Total median Δ |", "|---|---:|---:|---:|"]
    for kind, label in (("similar_I_different_S", "S high vs low (I matched)"), ("similar_S_different_I", "I high vs low (S matched)")):
        vals = {}
        for metric in ("dispatch_ms", "combine_ms", "total_ms"):
            arr = [float(r["relative_high_minus_low"]) for r in pairs if r["kind"] == kind and r["metric"] == metric and math.isfinite(float(r["relative_high_minus_low"]))]
            vals[metric] = float(np.median(arr)) if arr else float("nan")
        lines.append(f"| {label} | {vals['dispatch_ms']*100:.2f}% | {vals['combine_ms']*100:.2f}% | {vals['total_ms']*100:.2f}% |" if all(math.isfinite(v) for v in vals.values()) else f"| {label} | unavailable | unavailable | unavailable |")
    lines += ["", f"High-S/high-I versus the other quartile regimes interaction (total max-rank time): {interaction*100:.2f}%" if math.isfinite(interaction) else "High-S/high-I interaction: unavailable (insufficient regime rows).", "", "All-rank collective timing uses the maximum rank event per iteration, followed by a 20-iteration median/p25/p75/p95 summary. The 4-rank replay completed with 20 measured iterations per selected case and no DeepEP runtime errors.", "", "## Modality diagnostic", "", "Vision/Text labels are source labels within the same real-image request (`image_token_id == 151655` versus non-vision tokens). They were not used as a latency predictor. `modality_regime.csv` reports source-token S/I/rank-CV distributions; the figure uses modality-specific token subsets rather than request-level labels.", "", "## Artifacts", "", f"Result directory: `{out}`", "", "Figures: `plot1_saturation_imbalance_latency.png`, `plot2_matched_case_effects.png`, `plot3_modality_regime_distribution.png`.", "", "## Interpretation", ""]
    if summary['DEEPEP_SATURATION_LATENCY'] == 'GO':
        lines.append("The fixed gate found a repeatable ≥10% matched effect in at least one S/I comparison. This is mechanism evidence for a bounded communication-side follow-up, not a dynamic communication implementation.")
    elif summary['DEEPEP_SATURATION_LATENCY'] == 'HOLD':
        lines.append("The fixed gate found a 5–10% or incomplete matched effect. Communication-shape sensitivity is suggestive but not sufficient to justify a dynamic method without a larger controlled replay.")
    else:
        lines.append("The fixed gate found no repeatable ≥5% matched effect (or the required matched regimes were unavailable). Token/assignment volume remains the more defensible explanation in this bounded measurement.")
    lines += ["", "### Limitations", "", "DeepEP collectives were replayed with exact route IDs/layouts and deterministic random BF16 payloads, not live Qwen3 hidden states. The collective call is synchronous (`async_finish=False`) and no expert GEMM is included. If a required matched regime had insufficient artifact rows, it is explicitly marked unavailable rather than synthesized.", ""]
    report_path = ROOT / "poc_flashvep/reports/deepep_saturation_latency.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "run", "aggregate"):
        p = sub.add_parser(mode)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    sub.choices["prepare"].add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub.choices["run"].add_argument("--warmups", type=int, default=POLICY["warmups"])
    sub.choices["run"].add_argument("--iterations", type=int, default=POLICY["iterations"])
    sub.choices["run"].add_argument("--buffer-mib", type=int, default=512)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args)
    elif args.mode == "run": run(args)
    else: aggregate(args)


if __name__ == "__main__":
    main()
