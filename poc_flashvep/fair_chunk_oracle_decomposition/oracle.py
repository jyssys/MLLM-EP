"""Fair fixed/balanced/route-aware chunk decomposition.

The only mutable object in this PoC is the contiguous cut list.  Routes and
token order are loaded from the previously captured Qwen3-VL artifacts.  The
Numba kernels below implement exact dynamic programs over the same visual
expert-tile cost used by the preceding route-oracle experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / "poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
LONG = ROOT / "poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000"
OUT = ROOT / "poc_flashvep/deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_230000"
BUDGETS = (128, 256, 512, 1024)
STRATEGIES = ("fixed", "balanced", "same_count", "strict", "relaxed")
SHORT_IDS = ("coins", "cat", "logo", "coffee", "coffee_rocket", "model_card", "retina", "method")
LONG_IDS = ("long_6img_natural_fine", "long_8img_mixed", "long_10img_chart_mixed", "long_12img_broad")


def block_m(n: int) -> int:
    return 16 if n <= 32 else 32 if n <= 96 else 64 if n <= 512 else 128


def fixed_cuts(n: int, budget: int) -> list[int]:
    cuts = list(range(0, n, budget))
    if not cuts or cuts[-1] != n:
        cuts.append(n)
    return cuts


def balanced_cuts(n: int, budget: int) -> list[int]:
    k = (n + budget - 1) // budget
    q, r = divmod(n, k)
    sizes = [q + (i < r) for i in range(k)]
    return [int(x) for x in ([0] + list(np.cumsum(sizes, dtype=np.int64)))]


@njit(cache=True)
def _variable_prev(routes: np.ndarray, mask: np.ndarray, budget: int,
                   min_len: int, max_len: int) -> np.ndarray:
    """Variable-count exact DP; cost is visual assignment tile count."""
    n = routes.shape[0]
    inf = 1.0e18
    dp = np.full(n + 1, inf)
    prev = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0
    for start in range(n):
        if dp[start] >= inf:
            continue
        lo = start + min_len
        hi = min(n, start + max_len)
        counts = np.zeros((48, 128), dtype=np.int32)
        costs = np.zeros(4, dtype=np.int64)  # BLOCK_M 16, 32, 64, 128
        for end in range(start + 1, hi + 1):
            if mask[end - 1]:
                for layer in range(48):
                    for k in range(8):
                        expert = routes[end - 1, layer, k]
                        old = counts[layer, expert]
                        counts[layer, expert] = old + 1
                        if old % 16 == 0:
                            costs[0] += 1
                        if old % 32 == 0:
                            costs[1] += 1
                        if old % 64 == 0:
                            costs[2] += 1
                        if old % 128 == 0:
                            costs[3] += 1
            if end < lo:
                continue
            length = end - start
            ci = 0 if length <= 32 else 1 if length <= 96 else 2 if length <= 512 else 3
            value = dp[start] + costs[ci]
            if value < dp[end] - 1e-9:
                dp[end] = value
                prev[end] = start
    return prev


@njit(cache=True)
def _interval_costs(routes: np.ndarray, mask: np.ndarray, budget: int) -> np.ndarray:
    """Cost for every start and length up to budget, in one incremental pass."""
    n = routes.shape[0]
    out = np.full((n, budget + 1), -1, dtype=np.int64)
    for start in range(n):
        counts = np.zeros((48, 128), dtype=np.int32)
        costs = np.zeros(4, dtype=np.int64)
        hi = min(n, start + budget)
        for end in range(start + 1, hi + 1):
            if mask[end - 1]:
                for layer in range(48):
                    for k in range(8):
                        expert = routes[end - 1, layer, k]
                        old = counts[layer, expert]
                        counts[layer, expert] = old + 1
                        if old % 16 == 0:
                            costs[0] += 1
                        if old % 32 == 0:
                            costs[1] += 1
                        if old % 64 == 0:
                            costs[2] += 1
                        if old % 128 == 0:
                            costs[3] += 1
            length = end - start
            ci = 0 if length <= 32 else 1 if length <= 96 else 2 if length <= 512 else 3
            out[start, length] = costs[ci]
    return out


@njit(cache=True)
def _same_count_prev(routes: np.ndarray, mask: np.ndarray, budget: int,
                     chunk_count: int) -> np.ndarray:
    """Exact DP with exactly K chunks and each chunk <= budget."""
    n = routes.shape[0]
    inf = 1.0e18
    intervals = _interval_costs(routes, mask, budget)
    dp_prev = np.full(n + 1, inf)
    dp_prev[0] = 0.0
    back = np.full((chunk_count + 1, n + 1), -1, dtype=np.int64)
    for part in range(1, chunk_count + 1):
        dp_cur = np.full(n + 1, inf)
        for end in range(part, n - (chunk_count - part) + 1):
            lo = max(part - 1, end - budget)
            hi = end - 1
            for start in range(lo, hi + 1):
                if dp_prev[start] >= inf:
                    continue
                value = dp_prev[start] + intervals[start, end - start]
                if value < dp_cur[end] - 1e-9:
                    dp_cur[end] = value
                    back[part, end] = start
        dp_prev = dp_cur
    return back


def _recover(prev: np.ndarray, n: int, fallback: list[int]) -> list[int]:
    if prev[n] < 0:
        return fallback
    cuts = [n]
    cur = n
    while cur > 0:
        cur = int(prev[cur])
        if cur < 0:
            return fallback
        cuts.append(cur)
    return list(reversed(cuts))


def same_count_cuts(routes: np.ndarray, mask: np.ndarray, budget: int) -> list[int]:
    k = len(fixed_cuts(len(mask), budget)) - 1
    back = _same_count_prev(routes, mask, budget, k)
    n = len(mask)
    if back[k, n] < 0:
        return fixed_cuts(n, budget)
    cuts = [n]
    cur = n
    for part in range(k, 0, -1):
        cur = int(back[part, cur])
        if cur < 0:
            return fixed_cuts(n, budget)
        cuts.append(cur)
    return list(reversed(cuts))


def strict_cuts(routes: np.ndarray, mask: np.ndarray, budget: int) -> list[int]:
    prev = _variable_prev(routes, mask, budget, 1, budget)
    return _recover(prev, len(mask), fixed_cuts(len(mask), budget))


def relaxed_cuts(routes: np.ndarray, mask: np.ndarray, budget: int) -> list[int]:
    lo = max(1, int(np.ceil(.75 * budget)))
    hi = int(np.floor(1.25 * budget))
    prev = _variable_prev(routes, mask, budget, lo, hi)
    return _recover(prev, len(mask), fixed_cuts(len(mask), budget))


def load_routes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = json.loads((SHORT / "workload_manifest.json").read_text())
    by_id = {p["vision"]["request_id"]: p["vision"] for p in manifest["pairs"]}
    for request_id in SHORT_IDS:
        item = by_id[request_id]
        with np.load(SHORT / item["route_file"]) as z:
            routes = z["routed_experts"].astype(np.int64)
            tokens = z["prompt_token_ids"].astype(np.int64)
        rows.append({"request_id": request_id, "category": item["category"],
                     "routes": routes, "token_ids": tokens, "source": "short"})
    long_manifest = json.loads((LONG / "sample_manifest.json").read_text())
    long_by_id = {x["sample_id"]: x for x in long_manifest["samples"]}
    for request_id in LONG_IDS:
        item = long_by_id[request_id]
        with np.load(LONG / f"routing.{request_id}.npz") as z:
            routes = z["routed_experts"].astype(np.int64)
            tokens = z["prompt_token_ids"].astype(np.int64)
        rows.append({"request_id": request_id, "category": item["category"],
                     "routes": routes, "token_ids": tokens, "source": "long"})
    return rows


def chunk_metrics(route: np.ndarray, mask: np.ndarray, cuts: list[int], scope: str) -> dict[str, float]:
    tiles = pad = assignments = active = 0.0
    expert_batches: list[int] = []
    for st, en in zip(cuts[:-1], cuts[1:]):
        ids = route[st:en]
        if scope == "vision":
            ids = ids[mask[st:en]]
        flat = ids.reshape(-1)
        counts = np.bincount(flat, minlength=128) if len(flat) else np.zeros(128, dtype=np.int64)
        bm = block_m(en - st)
        q = (counts + bm - 1) // bm
        tiles += float(q.sum()); pad += float((q * bm - counts).sum())
        assignments += float(len(flat)); active += float(np.count_nonzero(counts))
        expert_batches.extend(counts[counts > 0].tolist())
    arr = np.asarray(expert_batches, dtype=np.float64)
    return {"tile_sum": tiles, "padded_rows": pad, "assignments": assignments,
            "active_expert_sum": active,
            "median_expert_batch": float(np.median(arr)) if len(arr) else 0.0,
            "p10_expert_batch": float(np.quantile(arr, .1)) if len(arr) else 0.0,
            "tiny_le_1_fraction": float(np.mean(arr <= 1)) if len(arr) else 0.0,
            "tiny_le_4_fraction": float(np.mean(arr <= 4)) if len(arr) else 0.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cut_manifest: dict[str, Any] = {"budgets": list(BUDGETS), "strategies": list(STRATEGIES), "samples": {}}
    for sample in load_routes():
        routes, tokens = sample["routes"], sample["token_ids"]
        mask = tokens == 151655
        cut_manifest["samples"][sample["request_id"]] = {}
        for budget in BUDGETS:
            cuts = {
                "fixed": fixed_cuts(len(routes), budget),
                "balanced": balanced_cuts(len(routes), budget),
                "same_count": same_count_cuts(routes, mask, budget),
                "strict": strict_cuts(routes, mask, budget),
                "relaxed": relaxed_cuts(routes, mask, budget),
            }
            cut_manifest["samples"][sample["request_id"]][str(budget)] = cuts
            for strategy, ends in cuts.items():
                sizes = np.diff(ends)
                for layer in range(48):
                    layer_route = routes[:, layer, :]
                    all_m = chunk_metrics(layer_route, mask, ends, "all")
                    vis_m = chunk_metrics(layer_route, mask, ends, "vision")
                    rows.append({"request_id": sample["request_id"], "source": sample["source"],
                                 "category": sample["category"], "budget": budget, "layer": layer,
                                 "strategy": strategy, "total_tokens": len(routes),
                                 "vision_tokens": int(mask.sum()), "vision_ratio": float(mask.mean()),
                                 "chunks": len(sizes), "min_chunk": int(sizes.min()),
                                 "max_chunk": int(sizes.max()), "chunk_size_cv": float(sizes.std() / sizes.mean()),
                                 "all_tile_sum": all_m["tile_sum"], "all_padded_rows": all_m["padded_rows"],
                                 "all_assignments": all_m["assignments"], "all_active_expert_sum": all_m["active_expert_sum"],
                                 "all_median_expert_batch": all_m["median_expert_batch"], "all_p10_expert_batch": all_m["p10_expert_batch"],
                                 "all_tiny_le_1_fraction": all_m["tiny_le_1_fraction"], "all_tiny_le_4_fraction": all_m["tiny_le_4_fraction"],
                                 "vision_tile_sum": vis_m["tile_sum"], "vision_padded_rows": vis_m["padded_rows"],
                                 "vision_assignments": vis_m["assignments"], "vision_active_expert_sum": vis_m["active_expert_sum"],
                                 "vision_median_expert_batch": vis_m["median_expert_batch"], "vision_p10_expert_batch": vis_m["p10_expert_batch"],
                                 "vision_tiny_le_1_fraction": vis_m["tiny_le_1_fraction"], "vision_tiny_le_4_fraction": vis_m["tiny_le_4_fraction"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "offline_per_request_layer.csv", index=False)
    (OUT / "strategy_cuts.json").write_text(json.dumps(cut_manifest, indent=2) + "\n")
    summaries = []
    for budget in BUDGETS:
        base = df[(df.budget == budget) & (df.strategy == "fixed")].set_index(["request_id", "layer"])
        for strategy in STRATEGIES:
            cur = df[(df.budget == budget) & (df.strategy == strategy)].set_index(["request_id", "layer"])
            idx = base.index.intersection(cur.index)
            row = {"budget": budget, "strategy": strategy, "observations": len(idx),
                   "median_chunks": float(cur.loc[idx, "chunks"].median()),
                   "median_chunk_cv": float(cur.loc[idx, "chunk_size_cv"].median()),
                   "median_min_chunk": float(cur.loc[idx, "min_chunk"].median()),
                   "median_max_chunk": float(cur.loc[idx, "max_chunk"].median())}
            for scope in ("all", "vision"):
                ratio = cur.loc[idx, f"{scope}_tile_sum"].to_numpy() / np.maximum(base.loc[idx, f"{scope}_tile_sum"].to_numpy(), 1e-12)
                row[f"{scope}_tile_reduction_vs_fixed"] = float(np.median(1 - ratio))
                row[f"{scope}_tile_ratio_vs_fixed"] = float(np.median(1 / np.maximum(ratio, 1e-12)))
                row[f"{scope}_padded_reduction_vs_fixed"] = float(np.median(1 - cur.loc[idx, f"{scope}_padded_rows"].to_numpy() / np.maximum(base.loc[idx, f"{scope}_padded_rows"].to_numpy(), 1e-12)))
            summaries.append(row)
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(OUT / "offline_strategy_summary.csv", index=False)
    # Component decomposition uses paired all-token tile reductions.
    # Components are computed from absolute proxy ratios below, so every
    # component is a paired difference between adjacent strategies.
    comp_rows = []
    for budget in BUDGETS:
        d = sdf[sdf.budget == budget].set_index("strategy")
        abs_ratio = {s: 1.0 - float(d.loc[s, "all_tile_reduction_vs_fixed"]) for s in STRATEGIES}
        comp_rows.append({"budget": budget,
                          "tail_balancing": abs_ratio["fixed"] - abs_ratio["balanced"],
                          "routing_only": abs_ratio["balanced"] - abs_ratio["same_count"],
                          "chunk_count_flexibility": abs_ratio["same_count"] - abs_ratio["strict"],
                          "relaxed_gt_budget": abs_ratio["strict"] - abs_ratio["relaxed"],
                          "total_fixed_to_relaxed": abs_ratio["fixed"] - abs_ratio["relaxed"]})
    pd.DataFrame(comp_rows).to_csv(OUT / "offline_decomposition.csv", index=False)
    # A compact report-friendly figure: proxy reduction relative to fixed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BUDGETS)); width = .16
    for j, strategy in enumerate(STRATEGIES):
        vals = [100 * float(sdf[(sdf.budget == b) & (sdf.strategy == strategy)]["all_tile_reduction_vs_fixed"].iloc[0]) for b in BUDGETS]
        ax.bar(x + (j - 2) * width, vals, width, label=strategy)
    ax.axhline(0, color="k", lw=.7); ax.set_xticks(x, [str(b) for b in BUDGETS]); ax.set_xlabel("Chunk budget"); ax.set_ylabel("All-token tile proxy reduction vs fixed (%)"); ax.set_title("Fair oracle decomposition (offline proxy)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fair_decomposition_proxy.png", dpi=180); plt.close(fig)
    summary = {"status": "ok", "result_dir": str(OUT), "budgets": list(BUDGETS), "requests": [x["request_id"] for x in load_routes()], "strategies": list(STRATEGIES), "exact_routes_unchanged": True, "gpu_mapping": [1, 2, 3, 4]}
    (OUT / "offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
