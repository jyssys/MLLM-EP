#!/usr/bin/env python3
"""Create CPU-only temporal tables and the required diagnostic figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--copy", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    frame = pd.read_json(args.trace, lines=True)
    copy = pd.read_csv(args.copy)
    oracle = json.loads(Path(args.oracle).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for row in frame.itertuples(index=False):
        groups[(row.request_id, row.block_id, row.layer_id)].append(row)
    persistence = []
    by_layer = defaultdict(list)
    for horizon in (1, 2, 4, 8):
        metrics = defaultdict(list)
        for (_request, _block, layer), rows in groups.items():
            indexed = {row.iteration_id: row for row in rows}
            for current in rows:
                future = indexed.get(current.iteration_id + horizon)
                if future is None:
                    continue
                left = np.asarray(current.expert_assignment_counts, dtype=float)
                right = np.asarray(future.expert_assignment_counts, dtype=float)
                rank_left = np.asarray(current.rank_assignment_counts, dtype=float)
                rank_right = np.asarray(future.rank_assignment_counts, dtype=float)
                metrics["expert_cosine"].append(cosine(left, right))
                metrics["rank_cosine"].append(cosine(rank_left, rank_right))
                metrics["rank_argmax_same"].append(int(rank_left.argmax() == rank_right.argmax()))
                metrics["top1_same"].append(int(left.argmax() == right.argmax()))
                top_count = max(1, int(np.ceil(.05 * len(left))))
                left_top = set(np.argpartition(left, -top_count)[-top_count:])
                right_top = set(np.argpartition(right, -top_count)[-top_count:])
                metrics["top5pct_retention"].append(len(left_top & right_top) / top_count)
                hot_left = left >= 1.5 * left.mean()
                hot_right = right >= 1.5 * right.mean()
                if hot_left.any():
                    metrics["hot_1p5_retention"].append(float(np.logical_and(hot_left, hot_right).sum() / hot_left.sum()))
                if horizon == 1:
                    by_layer[layer].append(cosine(left, right))
        persistence.append({"horizon": horizon, "pairs": len(metrics["expert_cosine"]),
                            **{name: float(np.mean(values)) for name, values in metrics.items()}})
    persistence_frame = pd.DataFrame(persistence)
    persistence_frame.to_csv(output / "temporal_persistence.csv", index=False)
    pd.DataFrame([{"layer": layer, "expert_cosine_t1": np.mean(values), "pairs": len(values)}
                  for layer, values in sorted(by_layer.items())]).to_csv(output / "layer_autocorrelation.csv", index=False)

    # Per-request aggregate diversity.
    request_rows = []
    request_vectors = {}
    for request, rows in frame.groupby("request_id"):
        ranks = np.stack(rows.rank_assignment_counts).sum(axis=0)
        request_vectors[request] = ranks
        request_rows.append({"request_id": request, **{f"r{i}": ranks[i] for i in range(4)},
                             "dominant_rank": int(ranks.argmax()),
                             "max_over_mean": float(ranks.max() / ranks.mean())})
    pd.DataFrame(request_rows).to_csv(output / "request_rank_profiles.csv", index=False)
    similarities = [cosine(request_vectors[a], request_vectors[b])
                    for i, a in enumerate(request_vectors) for b in list(request_vectors)[i + 1:]]
    diversity = {"pair_count": len(similarities), "cosine_p10": float(np.quantile(similarities, .1)),
                 "cosine_p50": float(np.quantile(similarities, .5)),
                 "cosine_p90": float(np.quantile(similarities, .9)),
                 "dominant_rank_counts": pd.Series([row["dominant_rank"] for row in request_rows]).value_counts().to_dict()}
    (output / "request_diversity.json").write_text(json.dumps(diversity, indent=2), encoding="utf-8")

    first = frame[frame.request_id == frame.request_id.iloc[0]].sort_values(["iteration_id", "layer_id"])
    iteration = first.groupby("iteration_id", as_index=False).agg(
        masked_before=("num_masked_before", "first"), masked_after=("num_masked_after", "first"),
        accepted=("num_newly_accepted", "first"), dispatch=("dispatch_ms", "sum"),
        expert=("expert_ms", "sum"), combine=("combine_ms", "sum"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(iteration["iteration_id"], iteration["dispatch"],
                 iteration["expert"], iteration["combine"],
                 labels=["dispatch", "expert", "combine"])
    ax.set(xlabel="denoising iteration", ylabel="summed CUDA ms over 16 layers", title="dLLM EP stage timeline (representative request)")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "01_iteration_timeline.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iteration.iteration_id, iteration.masked_before, marker="o", label="masked before")
    ax.plot(iteration.iteration_id, iteration.masked_after, marker="o", label="masked after")
    ax.bar(iteration.iteration_id, iteration.accepted, alpha=.3, label="accepted")
    ax.set(xlabel="denoising iteration", ylabel="positions", title="Mask progress")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "02_mask_progress.png", dpi=160); plt.close(fig)

    representative = first
    rank_matrix = np.stack(representative.rank_assignment_counts)
    expert_matrix = np.stack(representative.expert_assignment_counts)
    fig, ax = plt.subplots(figsize=(7, 5)); im=ax.imshow(rank_matrix, aspect="auto", cmap="magma")
    ax.set(xlabel="EP rank", ylabel="iteration × layer", title="Rank-load heatmap"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(output / "03_rank_heatmap.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); im=ax.imshow(expert_matrix, aspect="auto", cmap="viridis")
    ax.set(xlabel="expert", ylabel="iteration × layer", title="Expert-load heatmap"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(output / "04_expert_heatmap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(persistence_frame.horizon, persistence_frame.top1_same, marker="o", label="top-1 same")
    ax.plot(persistence_frame.horizon, persistence_frame.top5pct_retention, marker="o", label="top-5% retention")
    ax.plot(persistence_frame.horizon, persistence_frame.hot_1p5_retention, marker="o", label=">=1.5x mean retention")
    ax.set(xlabel="future iteration horizon", ylabel="conditional persistence", ylim=(0, 1), title="Hot-expert persistence")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "05_hot_persistence.png", dpi=160); plt.close(fig)

    layer = pd.read_csv(output / "layer_autocorrelation.csv")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(layer.layer, layer.expert_cosine_t1)
    ax.set(xlabel="layer", ylabel="t→t+1 cosine", ylim=(0, 1), title="Expert-load autocorrelation by layer")
    fig.tight_layout(); fig.savefig(output / "06_layer_autocorrelation.png", dpi=160); plt.close(fig)

    matrix = np.full((4, 4), np.nan)
    visible = np.full((4, 4), np.nan)
    for row in copy.itertuples(index=False):
        matrix[row.physical_src - 4, row.physical_dst - 4] = row.copy_p50_ms
        visible[row.physical_src - 4, row.physical_dst - 4] = row.visible_copy_p50_ms
    for number, values, title in ((7, matrix, "12 MiB expert P2P copy p50 (ms)"),
                                  (8, visible, "Visible copy cost under compute overlap (ms)")):
        fig, ax = plt.subplots(figsize=(5, 4)); im=ax.imshow(values, cmap="magma")
        ax.set_xticks(range(4), labels=range(4, 8)); ax.set_yticks(range(4), labels=range(4, 8))
        ax.set(xlabel="destination GPU", ylabel="source GPU", title=title); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(output / f"{number:02d}_{'copy' if number == 7 else 'visible_copy'}_heatmap.png", dpi=160); plt.close(fig)

    replica = pd.DataFrame(oracle["replication"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(replica.horizon, replica.oracle_gross_expert_ms, marker="o", label="perfect gross")
    ax.plot(replica.horizon, replica.oracle_net_expert_ms, marker="o", label="perfect net")
    ax.plot(replica.horizon, replica.causal_net_expert_ms, marker="o", label="current-state net")
    ax.set(xlabel="replica lifetime H", ylabel="aggregate optimistic ms", title="Replication benefit after measured copy cost")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "09_replica_lifetime.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); width=.35; x=np.arange(len(replica))
    ax.bar(x-width/2, replica.oracle_projected_request_reduction_pct, width, label="perfect")
    ax.bar(x+width/2, replica.causal_projected_request_reduction_pct, width, label="current-state")
    ax.set_xticks(x, labels=replica.horizon); ax.set(xlabel="H", ylabel="optimistic request reduction (%)", title="Perfect vs causal replication")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "10_replica_policy.png", dpi=160); plt.close(fig)

    # One replica per layer means at most 16 concurrent replicas: 1.56% of
    # global expert storage.  Below that, scale linearly as an optimistic cap.
    max_benefit = float(replica.causal_projected_request_reduction_pct.max())
    budgets = np.array([1., 2., 5., 10.]); benefits = max_benefit * np.minimum(1., budgets / 1.5625)
    pd.DataFrame({"budget_pct_global_expert_storage": budgets, "optimistic_causal_e2e_pct": benefits}).to_csv(output / "replica_memory_budget.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(budgets, benefits, marker="o")
    ax.set(xlabel="replica budget (% global expert storage)", ylabel="optimistic request reduction (%)", title="Memory budget upper bound")
    fig.tight_layout(); fig.savefig(output / "11_replica_memory.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(similarities, bins=20)
    ax.set(xlabel="pairwise aggregate rank-vector cosine", ylabel="request pairs", title="Request rank-load diversity")
    fig.tight_layout(); fig.savefig(output / "12_request_similarity.png", dpi=160); plt.close(fig)

    current_max, future_max = [], []
    for rows in groups.values():
        indexed = {row.iteration_id: row for row in rows}
        for current in rows:
            future = indexed.get(current.iteration_id + 1)
            if future is not None:
                current_max.append(max(current.rank_assignment_counts))
                future_max.append(max(future.rank_assignment_counts))
    fig, ax = plt.subplots(figsize=(5, 5)); ax.hexbin(current_max, future_max, gridsize=35, mincnt=1, cmap="viridis")
    low = min(current_max + future_max); high = max(current_max + future_max)
    ax.plot([low, high], [low, high], "r--", linewidth=1)
    ax.set(xlabel="current max-rank load", ylabel="next max-rank load",
           title="Current-iteration predictor")
    fig.tight_layout(); fig.savefig(output / "13_current_vs_next_load.png", dpi=160); plt.close(fig)

    batching = oracle["batching"]
    names = ["fcfs", "random", "perfect", "current", "ema"]
    values = [batching["assignment_cost"][name] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(names, values)
    ax.set(ylabel="aggregate max-rank assignment cost", title="Complementary batching policies")
    fig.tight_layout(); fig.savefig(output / "14_batching_policies.png", dpi=160); plt.close(fig)

    iteration_cost = pd.DataFrame(batching["iterations"])
    iteration_cost.to_csv(output / "batching_iteration_costs.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("fcfs", "perfect", "current", "ema"):
        ax.plot(iteration_cost.iteration, iteration_cost[name], label=name)
    ax.set(xlabel="denoising iteration", ylabel="max-rank assignment cost", title="Batch cost by iteration")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "15_batch_cost_distribution.png", dpi=160); plt.close(fig)

    names = ["random", "perfect", "current", "ema"]
    gains = [batching[f"{name}_optimistic_e2e_projection_pct"] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(names, gains)
    ax.set(ylabel="optimistic E2E reduction (%)", title="Load-to-request upper-bound mapping")
    fig.tight_layout(); fig.savefig(output / "16_e2e_projection.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
