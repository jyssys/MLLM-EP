#!/usr/bin/env python3
"""Aggregate live Qwen3 TP4/EP4 expert timing and tail statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def load_rows(path: Path) -> list[dict]:
    rows = []
    for file in sorted(path.glob("rank*.jsonl")):
        rows.extend(json.loads(line) for line in file.read_text().splitlines())
    return rows


def add_features(row: dict, block: int) -> None:
    active = [int(x) for x in row["histogram"] if int(x) > 0]
    row["active_experts"] = len(active)
    row["Q"] = sum(math.ceil(x / block) for x in active)
    row["full_tiles"] = sum(x // block for x in active)
    row["tail_count"] = sum(x % block != 0 for x in active)
    row["tail_rows"] = sum(x % block for x in active)
    row["padded_rows"] = row["Q"] * block - row["N"]
    row["padding_amplification"] = row["Q"] * block / row["N"] if row["N"] else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.live_dir / "raw_live")
    # Auto-selected backend is FlashInfer CUTLASS; its generated H100 family
    # compiled M128 kernels. Do not reuse Triton config lookup as its block size.
    block = 128
    for row in rows:
        add_features(row, block)
    synthetic = json.loads((args.result_dir / "synthetic.json").read_text())
    qwen_shape = next(x for x in synthetic["shapes"] if x["name"] == "qwen3_30b_a3b")
    lut = {int(x["rows"]): float(x["median_ms"]) for x in qwen_shape["tail_lut"]}
    lut_x = np.asarray(sorted(lut))
    lut_y = np.asarray([lut[int(x)] for x in lut_x])
    full_tile_cost = lut[block]
    for row in rows:
        active = [int(x) for x in row["histogram"] if int(x) > 0]
        tails = [x % block for x in active if x % block]
        current = row["full_tiles"] + len(tails)
        ideal = row["full_tiles"] + sum(
            float(np.interp(tail, lut_x, lut_y)) / full_tile_cost
            for tail in tails
        )
        row["tail_oracle_speedup"] = current / ideal if ideal else 1.0
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["workload"], row["repeat"], row["layer"])].append(row)
    comparisons = []
    for key, group in groups.items():
        if len(group) != 4:
            raise AssertionError((key, len(group)))
        token = max(group, key=lambda x: x["N"])["ep_rank"]
        tile = max(group, key=lambda x: x["Q"])["ep_rank"]
        actual = max(group, key=lambda x: x["expert_ms"])["ep_rank"]
        comparisons.append({
            "workload": key[0], "repeat": key[1], "layer": key[2],
            "token_rank": token, "tile_rank": tile, "actual_rank": actual,
            "token_match": token == actual, "tile_match": tile == actual,
            "token_tile_differ": token != tile,
            "tile_corrects_token": token != actual and tile == actual,
            "oracle_makespan_speedup": max(x["Q"] for x in group) / max(
                x["Q"] / x["tail_oracle_speedup"] for x in group
            ),
        })
    active_total = sum(r["active_experts"] for r in rows)
    partial_total = sum(r["tail_count"] for r in rows)
    q_total = sum(r["Q"] for r in rows)
    padded_total = sum(r["padded_rows"] for r in rows)
    effective_total = q_total * block
    n = np.asarray([r["N"] for r in rows]); q = np.asarray([r["Q"] for r in rows]); t = np.asarray([r["expert_ms"] for r in rows])
    summary = {
        "backend": sorted(set(r["expert_backend"] for r in rows)),
        "topology": {"TP": 4, "DP": 1, "EP": 4, "PP": 1, "local_experts": 32, "placement": "linear"},
        "observations": len(rows), "layer_rank_groups": len(comparisons), "block_m_source": "FlashInfer generated H100 M128 kernel family",
        "partial_tail_expert_fraction": partial_total / active_total,
        "tail_tile_fraction": partial_total / q_total,
        "padded_row_fraction_of_effective": padded_total / effective_total,
        "padding_amplification_median": float(np.median([r["padding_amplification"] for r in rows])),
        "padding_amplification_p95": float(np.percentile([r["padding_amplification"] for r in rows], 95)),
        "assignment_latency_spearman": float(spearmanr(n, t).statistic),
        "tile_latency_spearman": float(spearmanr(q, t).statistic),
        "token_critical_match": float(np.mean([x["token_match"] for x in comparisons])),
        "tile_critical_match": float(np.mean([x["tile_match"] for x in comparisons])),
        "token_tile_critical_disagreement": float(np.mean([x["token_tile_differ"] for x in comparisons])),
        "tile_corrects_token_fraction": float(np.mean([x["tile_corrects_token"] for x in comparisons])),
        "rank_tail_oracle_median_speedup": float(np.median([r["tail_oracle_speedup"] for r in rows])),
        "rank_tail_oracle_p95_speedup": float(np.percentile([r["tail_oracle_speedup"] for r in rows], 95)),
        "layer_makespan_oracle_median_speedup": float(np.median([x["oracle_makespan_speedup"] for x in comparisons])),
        "layer_makespan_oracle_p95_speedup": float(np.percentile([x["oracle_makespan_speedup"] for x in comparisons], 95)),
        "timing_median_ms": float(np.median(t)), "timing_p95_ms": float(np.percentile(t, 95)),
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "real_trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.result_dir / "real_trace_rows.csv").open("w", newline="") as f:
        fields = ["workload","repeat","layer","ep_rank","N","active_experts","Q","tail_count","padded_rows","padding_amplification","tail_oracle_speedup","expert_ms"]
        w=csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows({k:r[k] for k in fields} for r in rows)
    fig_dir=args.result_dir/"figures"; fig_dir.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].hist([r["padding_amplification"] for r in rows],bins=30); axs[0].set_xlabel("padding amplification"); axs[0].set_ylabel("rank-layer observations")
    axs[1].hist([r["tail_count"]/r["Q"] for r in rows],bins=30); axs[1].set_xlabel("tail tiles / effective tiles")
    fig.tight_layout(); fig.savefig(fig_dir/"plot4_real_ep_tail_distribution.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].scatter(n,t,s=4,alpha=.35,label="N"); axs[0].set_xlabel("assignments"); axs[0].set_ylabel("live expert ms")
    axs[1].bar(["token argmax","tile argmax"],[summary["token_critical_match"],summary["tile_critical_match"]]); axs[1].set_ylim(0,1); axs[1].set_ylabel("actual critical-rank match")
    fig.tight_layout(); fig.savefig(fig_dir/"plot5_real_ep_tile_vs_straggler.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].hist([r["tail_oracle_speedup"] for r in rows],bins=30)
    axs[0].axvline(1,color="black",lw=.7); axs[0].set_xlabel("rank measured-tail LUT oracle speedup")
    axs[0].set_ylabel("rank-layer observations")
    axs[1].hist([x["oracle_makespan_speedup"] for x in comparisons],bins=30)
    axs[1].axvline(1,color="black",lw=.7); axs[1].set_xlabel("layer compute-makespan oracle speedup")
    fig.tight_layout(); fig.savefig(fig_dir/"plot6_exact_routing_oracle_headroom.png",dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
