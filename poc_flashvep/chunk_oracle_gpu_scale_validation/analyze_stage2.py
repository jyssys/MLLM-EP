"""Scale analysis for the bounded long-request route capture.

This is intentionally an offline, route-preserving analysis.  It reuses the
same fixed-cut and bounded exact visual-route DP as the preceding spatial
chunk experiment; no route or token order is changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000"
ANALYSIS = RESULT / "analysis"
FIG = RESULT / "figures"
ANALYSIS.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
BUDGETS = (128, 256, 512, 1024)
LAYERS = tuple(range(48))


def block_m(n: int) -> int:
    # Source-audited vLLM BF16 TritonExperts proxy used by the prior PoC.
    return 16 if n <= 32 else 32 if n <= 96 else 64 if n <= 512 else 128


def fixed_cuts(n: int, budget: int) -> list[int]:
    cuts = list(range(0, n, budget))
    if not cuts or cuts[-1] != n:
        cuts.append(n)
    return cuts


def valid_range(start: int, n: int, budget: int) -> tuple[int, int]:
    lo = start + max(1, int(np.ceil(.75 * budget)))
    hi = min(n, start + int(np.floor(1.25 * budget)))
    return lo, hi


def visual_prefix(routes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, layers, _ = routes.shape
    out = np.zeros((layers, 128, n + 1), dtype=np.int32)
    for pos in range(n):
        out[:, :, pos + 1] = out[:, :, pos]
        if bool(mask[pos]):
            for layer in range(layers):
                out[layer, routes[pos, layer], pos + 1] += 1
    return out


@njit(cache=True)
def _oracle_prev_numba(routes: np.ndarray, mask: np.ndarray, budget: int) -> np.ndarray:
    """Exact bounded DP, with incremental tile costs (avoids O(48*128) per cut)."""
    n = routes.shape[0]
    inf = 1.0e18
    dp = np.full(n + 1, inf)
    prev = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0
    for start in range(n):
        if dp[start] >= inf:
            continue
        lo = start + (3 * budget + 3) // 4
        hi = min(n, start + (5 * budget) // 4)
        counts = np.zeros((48, 128), dtype=np.int32)
        # Tile costs for the only BLOCK_M values reachable by the bounded
        # intervals (32, 64, 128).  A ceil count increases at each multiple.
        costs = np.zeros(3, dtype=np.int64)
        for end in range(start + 1, hi + 1):
            if mask[end - 1]:
                for layer in range(48):
                    for k in range(8):
                        expert = routes[end - 1, layer, k]
                        old = counts[layer, expert]
                        counts[layer, expert] = old + 1
                        if old % 32 == 0:
                            costs[0] += 1
                        if old % 64 == 0:
                            costs[1] += 1
                        if old % 128 == 0:
                            costs[2] += 1
            if end < lo:
                continue
            length = end - start
            ci = 0 if length <= 32 else 1 if length <= 96 else 2
            value = dp[start] + costs[ci]
            if value < dp[end] - 1e-9:
                dp[end] = value
                prev[end] = start
    return prev


def oracle_cuts(routes: np.ndarray, mask: np.ndarray, budget: int) -> list[int]:
    prev = _oracle_prev_numba(routes, mask, budget)
    n = len(mask)
    if prev[n] < 0:
        return fixed_cuts(n, budget)
    cuts = [n]
    cur = n
    while cur > 0:
        cur = int(prev[cur])
        cuts.append(cur)
    return list(reversed(cuts))


def scope_metrics(route: np.ndarray, mask: np.ndarray, cuts: list[int],
                  scope: str) -> tuple[float, float, float, float]:
    """Return tile sum, padded rows, assignment count, and tail count."""
    tile_sum = padded = assignments = tails = 0.0
    for st, en in zip(cuts[:-1], cuts[1:]):
        ids = route[st:en]
        if scope == "vision":
            ids = ids[mask[st:en]]
        flat = ids.reshape(-1)
        counts = np.bincount(flat, minlength=128) if len(flat) else np.zeros(128, dtype=np.int64)
        m = block_m(en - st)
        q = (counts + m - 1) // m
        tile_sum += float(q.sum())
        assignments += float(len(flat))
        padded += float((q * m - counts).sum())
        tails += float(np.count_nonzero(counts % m))
    return tile_sum, padded, assignments, tails


def load() -> tuple[dict, list[dict]]:
    manifest = json.loads((RESULT / "sample_manifest.json").read_text())
    items = []
    for sample in manifest["samples"]:
        route_path = RESULT / f"routing.{sample['sample_id']}.npz"
        with np.load(route_path) as z:
            route = z["routed_experts"].astype(np.int64)
            token_ids = z["prompt_token_ids"].astype(np.int64)
        if route.shape[1:] != (48, 8) or len(token_ids) != len(route):
            raise AssertionError((sample["sample_id"], route.shape, len(token_ids)))
        items.append({"sample": sample, "route": route,
                      "token_ids": token_ids,
                      "mask": token_ids == int(sample.get("image_token_id", 151655))})
    return manifest, items


def main() -> None:
    manifest, items = load()
    rows = []
    for item in items:
        sample = item["sample"]
        route, mask = item["route"], item["mask"]
        for budget in BUDGETS:
            cuts_by = {"fixed": fixed_cuts(len(route), budget),
                       "oracle": oracle_cuts(route, mask, budget)}
            for strategy, cuts in cuts_by.items():
                for layer in LAYERS:
                    layer_route = route[:, layer, :]
                    all_m = scope_metrics(layer_route, mask, cuts, "all")
                    vis_m = scope_metrics(layer_route, mask, cuts, "vision")
                    rows.append({
                        "sample_id": sample["sample_id"],
                        "category": sample["category"],
                        "budget": budget,
                        "layer": layer,
                        "strategy": strategy,
                        "total_tokens": len(route),
                        "vision_tokens": int(mask.sum()),
                        "vision_ratio": float(mask.mean()),
                        "chunks": len(cuts) - 1,
                        "chunk_sizes": json.dumps([int(b-a) for a, b in zip(cuts[:-1], cuts[1:])]),
                        "boundaries": json.dumps(cuts),
                        "all_tile_sum": all_m[0], "all_padded_rows": all_m[1],
                        "all_assignments": all_m[2], "all_tail_count": all_m[3],
                        "vision_tile_sum": vis_m[0], "vision_padded_rows": vis_m[1],
                        "vision_assignments": vis_m[2], "vision_tail_count": vis_m[3],
                    })
    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS / "stage2_per_observation.csv", index=False)
    # Paired fixed/oracle summaries.  Tile ratio is a reduction proxy: >1 is
    # better than fixed, and is reported separately for all and visual scope.
    summary = []
    for budget in BUDGETS:
        f = df[(df.budget == budget) & (df.strategy == "fixed")].set_index(["sample_id", "layer"])
        o = df[(df.budget == budget) & (df.strategy == "oracle")].set_index(["sample_id", "layer"])
        idx = f.index.intersection(o.index)
        row = {"budget": budget, "observations": len(idx),
               "sample_count": int(df[df.budget == budget].sample_id.nunique())}
        for scope in ("all", "vision"):
            ftiles = f.loc[idx, f"{scope}_tile_sum"]
            otiles = o.loc[idx, f"{scope}_tile_sum"]
            fpads = f.loc[idx, f"{scope}_padded_rows"]
            opads = o.loc[idx, f"{scope}_padded_rows"]
            red = 1.0 - otiles.to_numpy() / np.maximum(ftiles.to_numpy(), 1e-12)
            row[f"{scope}_tile_ratio_median"] = float(np.median(ftiles.to_numpy() / np.maximum(otiles.to_numpy(), 1e-12)))
            row[f"{scope}_reduction_median"] = float(np.median(red))
            row[f"{scope}_reduction_mean"] = float(np.mean(red))
            row[f"{scope}_positive_fraction"] = float(np.mean(red > 0))
            row[f"{scope}_padded_reduction_median"] = float(np.median(1.0 - opads.to_numpy() / np.maximum(fpads.to_numpy(), 1e-12)))
        row["fixed_median_chunks"] = float(f.loc[idx, "chunks"].median())
        row["oracle_median_chunks"] = float(o.loc[idx, "chunks"].median())
        row["max_tokens"] = int(df[df.budget == budget].total_tokens.max())
        summary.append(row)
    summ = pd.DataFrame(summary)
    summ.to_csv(ANALYSIS / "stage2_summary.csv", index=False)

    # Token length manifest table and a compact per-sample headroom table.
    manifest_rows = []
    for item in items:
        s = item["sample"]
        manifest_rows.append({"sample_id": s["sample_id"], "category": s["category"],
                              "prompt_tokens": s["processor_prompt_tokens"],
                              "vision_tokens": s["processor_vision_tokens"],
                              "image_count": len(s["images"]),
                              "question": s["question"]})
    pd.DataFrame(manifest_rows).to_csv(ANALYSIS / "stage2_workload_manifest.csv", index=False)

    # Figures use all-layer paired observations; medians are explicitly marked.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    med = []
    for budget in BUDGETS:
        f = df[(df.budget == budget) & (df.strategy == "fixed")].set_index(["sample_id", "layer"])
        o = df[(df.budget == budget) & (df.strategy == "oracle")].set_index(["sample_id", "layer"])
        idx = f.index.intersection(o.index)
        red = (1.0 - o.loc[idx, "all_tile_sum"].to_numpy() / np.maximum(f.loc[idx, "all_tile_sum"].to_numpy(), 1e-12)) * 100
        ax.scatter(np.full(len(red), budget), red, s=16, alpha=.35)
        med.append(float(np.median(red)))
    ax.plot(BUDGETS, med, "o-", color="#b91c1c", label="median all-token proxy")
    ax.axhline(10, ls="--", color="#15803d", lw=.9, label="10% reference")
    ax.set_xlabel("Chunk budget"); ax.set_ylabel("Fixed → route-oracle reduction (%)")
    ax.set_title("Long multimodal routes: oracle proxy headroom by budget")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "plot4_stage2_oracle_headroom_by_budget.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = [s["prompt_tokens"] for s in manifest_rows]
    y = [s["vision_tokens"] for s in manifest_rows]
    labels = [s["sample_id"].replace("long_", "") for s in manifest_rows]
    ax.scatter(x, y, s=50, color="#2563eb")
    for a, b, label in zip(x, y, labels): ax.annotate(label, (a, b), fontsize=8, xytext=(3, 3), textcoords="offset points")
    lim = [0, max(x) * 1.1]; ax.plot(lim, lim, "k--", lw=.7, alpha=.5)
    ax.set_xlabel("Decoder input tokens"); ax.set_ylabel("Visual tokens"); ax.set_title("Bounded long multimodal workload lengths")
    fig.tight_layout(); fig.savefig(FIG / "plot5_stage2_token_length_distribution.png", dpi=180); plt.close(fig)

    out = {"status": "ok", "run_id": RESULT.name.rsplit("_", 2)[-2] + "_" + RESULT.name.rsplit("_", 1)[-1],
           "samples": manifest_rows, "budgets": list(BUDGETS), "layers": 48,
           "summary": summ.to_dict(orient="records"),
           "oracle_definition": "same bounded [0.75,1.25] exact visual-route DP as prior offline PoC",
           "stage2_scale_gate": "STRONG_POSITIVE_AT_512_1024" if bool((summ[summ.budget.isin([512, 1024])]["all_reduction_median"] >= .10).all()) else "NOT_REPEATED_AT_512_1024"}
    (RESULT / "stage2_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
