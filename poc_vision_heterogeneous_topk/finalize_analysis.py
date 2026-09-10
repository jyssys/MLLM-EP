#!/usr/bin/env python3
"""Create the evidence tables and plots used by the final decision report."""

from __future__ import annotations

import collections
import csv
import glob
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "final_analysis"
PLOTS = OUT / "plots"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def med(values) -> float:
    return float(np.median(list(values)))


def router_mass() -> list[dict]:
    capture = RESULTS / "vision_capture_fresh_20260911_0050"
    manifest = json.loads((capture / "manifest.json").read_text())
    source = {row["sample_id"]: int(row["source_dp_rank"])
              for row in manifest["schedule"]}
    rows = []
    for sample in manifest["samples"]:
        sid, n = sample["sample_id"], int(sample["prompt_tokens"])
        start, end = map(int, sample["visual_span"])
        text_start, text_end = map(int, sample["post_visual_span"])
        modality = np.full(n, "other", dtype="<U6")
        modality[start:end] = "vision"; modality[text_start:text_end] = "text"
        for layer in (4, 24, 44):
            parts = [np.load(capture / "raw" / f"router.{sid}.dp{source[sid]}.tp{tp}.layer{layer}.npz")
                     for tp in (0, 1)]
            weights = np.concatenate([x["topk_weights"] for x in parts], axis=0)[:n]
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            order = np.sort(weights, axis=1)[:, ::-1]
            cumulative = np.cumsum(order, axis=1)
            entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1)
            for kind in ("vision", "text"):
                mask = modality == kind
                if not mask.any():
                    continue
                row = {"sample": sid, "category": sample["category"], "layer": layer,
                       "modality": kind, "tokens": int(mask.sum()),
                       "router_entropy_median": med(entropy[mask])}
                for k in range(1, 9):
                    row[f"top{k}_mass_median"] = med(cumulative[mask, k - 1])
                rows.append(row)
    write_csv(OUT / "router_mass_distribution.csv", rows)
    return rows


def attention_summary() -> list[dict]:
    trace = RESULTS / "bottleneck_attention_20260911_0138" / "attention_trace"
    by_pid_m: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for path in trace.glob("attention_pid*.jsonl"):
        for line in path.open():
            row = json.loads(line)
            if int(row["M"]) in (256, 8192, 16384):
                by_pid_m[(int(row["pid"]), int(row["M"]))].append(row)
    chunks: dict[tuple[int, int], list[float]] = {}
    for key, values in by_pid_m.items():
        complete = [values[i:i + 48] for i in range(0, len(values), 48)]
        chunks[key] = [sum(float(x["attention_cuda_ms"]) for x in c)
                       for c in complete if len(c) == 48]
    rows = []
    pids = sorted({key[0] for key in chunks})
    for m in (256, 8192, 16384):
        usable = min(len(chunks.get((pid, m), [])) for pid in pids)
        # The first complete request at each shape is the registered warmup.
        values = [max(chunks[(pid, m)][i] for pid in pids) for i in range(1, usable)]
        rows.append({"prompt_M": m, "samples": len(values),
                     "attention_48layer_critical_median_ms": med(values),
                     "attention_48layer_critical_p90_ms": float(np.quantile(values, .9))})
    write_csv(OUT / "attention_summary.csv", rows)
    return rows


def layer_bottleneck() -> list[dict]:
    source = RESULTS / "bottleneck_stage_20260911_0046" / "trace" / "invocations.jsonl"
    grouped: dict[tuple[int, int, int, str], list[dict]] = collections.defaultdict(list)
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("phase") != "prefill" or int(row.get("M", -1)) not in (128, 1024, 4096, 8192):
                continue
            grouped[(int(row["M"]), int(row["layer"]), int(row["dp_rank"]), row["route_id"])].append(row)
    by_shape_layer: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for (m, layer, dp, route), ranks in grouped.items():
        stages = []
        for row in ranks:
            stage = {x["stage"]: float(x["cuda_ms"]) for x in row["stage_records"]}
            stages.append({"expert": stage.get("expert", np.nan),
                           "dispatch": stage.get("deepep_dispatch", np.nan),
                           "combine": stage.get("deepep_combine", np.nan),
                           "moe": float(row["cuda_ms"]),
                           "rank_ratio": float(row["rank_max_mean"]),
                           "invocation": int(row["local_invocation_id"])})
        by_shape_layer[(m, layer)].append({
            "invocation": min(x["invocation"] for x in stages),
            "expert_max": max(x["expert"] for x in stages),
            "expert_mean": med(x["expert"] for x in stages),
            "dispatch_max": max(x["dispatch"] for x in stages),
            "combine_max": max(x["combine"] for x in stages),
            "moe_max": max(x["moe"] for x in stages),
            "rank_ratio": max(x["rank_ratio"] for x in stages),
        })
    rows = []
    for (m, layer), values in sorted(by_shape_layer.items()):
        values.sort(key=lambda x: x["invocation"])
        # Four routes are expected (one warmup + three measurements) per DP;
        # discard the two earliest route records at each layer (one per DP).
        values = values[2:]
        tails = [x["expert_max"] * max(0.0, 1 - 1 / x["rank_ratio"]) for x in values]
        rows.append({"per_tp_M": m, "prompt_length": m * 2, "layer": layer,
                     "samples": len(values), "expert_critical_median_ms": med(x["expert_max"] for x in values),
                     "expert_rank_spread_median_ms": med(x["expert_max"] - x["expert_mean"] for x in values),
                     "rank_max_mean_median": med(x["rank_ratio"] for x in values),
                     "load_balance_tail_oracle_median_ms": med(tails),
                     "dispatch_critical_median_ms": med(x["dispatch_max"] for x in values),
                     "combine_critical_median_ms": med(x["combine_max"] for x in values),
                     "moe_critical_median_ms": med(x["moe_max"] for x in values)})
    write_csv(OUT / "layer_bottleneck.csv", rows)
    return rows


def replay_summary() -> list[dict]:
    paths = [Path(x) for x in glob.glob(str(RESULTS / "replay_*.jsonl"))]
    rows = []
    for path in sorted(paths):
        data = [json.loads(x) for x in path.open()]
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for row in data:
            groups[row["policy"]].append(row)
        baseline = med(x["moe_ms"] for x in groups["full_k8"])
        baseline_expert = med(x["expert_ms"] for x in groups["full_k8"])
        meta = json.loads(path.with_suffix(".summary.json").read_text())
        for policy, values in groups.items():
            moe = med(x["moe_ms"] for x in values)
            expert = med(x["expert_ms"] for x in values)
            rows.append({"source": path.name, "sample": meta["sample"], "layer": meta["layer"],
                         "tokens": meta["tokens"], "renormalized": bool(meta.get("renormalize", False)), "policy": policy,
                         "reps": len(values), "moe_median_ms": moe,
                         "moe_p90_ms": float(np.quantile([x["moe_ms"] for x in values], .9)),
                         "moe_gain_pct": 100 * (1 - moe / baseline),
                         "expert_gain_pct": 100 * (1 - expert / baseline_expert),
                         "worst_rel_l2_median": med(x["worst_rel_l2"] for x in values),
                         "worst_cosine_median": med(x["worst_cosine"] for x in values),
                         "assignment_drop_fraction": med(x["all_assignment_drop"] for x in values),
                         "router_mass_drop_fraction": med(x["all_router_mass_drop"] for x in values)})
    write_csv(OUT / "replay_policy_summary.csv", rows)
    return rows


def projections(replays: list[dict]) -> list[dict]:
    atlas = json.loads((RESULTS / "bottleneck_analysis_20260911_0051" / "summary.json").read_text())["rows"]
    share = {int(x["per_tp_M"]): float(x["moe_ms_clean_ttft_share"]) for x in atlas}
    rows = []
    for row in replays:
        m = int(row["tokens"])
        if m not in share or row["policy"] == "full_k8":
            continue
        out = dict(row)
        out["moe_share_of_clean_ttft_observer_assisted"] = share[m]
        out["projected_ttft_gain_pct"] = row["moe_gain_pct"] * share[m]
        out["strict_local_quality_pass"] = row["worst_rel_l2_median"] <= .01
        rows.append(out)
    write_csv(OUT / "ttft_projections.csv", rows)
    return rows


def plots(atlas: list[dict], layers: list[dict], replay: list[dict], projections_: list[dict],
          router: list[dict]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    x = [r["prompt_length"] for r in atlas]
    plt.figure(figsize=(7, 4))
    for metric, label in (("dispatch_ms_clean_ttft_share", "Dispatch"),
                          ("expert_ms_clean_ttft_share", "Expert"),
                          ("combine_ms_clean_ttft_share", "Combine")):
        plt.plot(x, [100 * r[metric] for r in atlas], marker="o", label=label)
    plt.xscale("log", base=2); plt.xlabel("Prompt tokens"); plt.ylabel("Share of clean TTFT (%)")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "stage_fraction_vs_scale.png", dpi=180); plt.close()

    by_m = collections.defaultdict(list)
    for r in layers: by_m[r["per_tp_M"]].append(r)
    tail = []
    for r in atlas:
        m = r["per_tp_M"]
        tail_ms = sum(x["load_balance_tail_oracle_median_ms"] for x in by_m[m])
        tail.append(100 * tail_ms / r["clean_ttft_ms"])
    plt.figure(figsize=(7, 4)); plt.plot(x, [100 * r["expert_ms_clean_ttft_share"] for r in atlas], marker="o", label="Expert")
    plt.plot(x, tail, marker="s", label="Ideal rank-tail removable upper bound")
    plt.xscale("log", base=2); plt.xlabel("Prompt tokens"); plt.ylabel("Clean TTFT share (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "expert_and_tail_headroom.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 4)); plt.plot(x, [r["layer_mean_rank_max_mean_median"] for r in atlas], marker="o")
    plt.xscale("log", base=2); plt.xlabel("Prompt tokens"); plt.ylabel("Rank max / mean"); plt.tight_layout()
    plt.savefig(PLOTS / "rank_imbalance_vs_scale.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 4))
    for modality, marker in (("vision", "o"), ("text", "s")):
        group = [r for r in router if r["modality"] == modality]
        plt.plot(range(1, 9), [100 * med(r[f"top{k}_mass_median"] for r in group)
                              for k in range(1, 9)], marker=marker, label=modality)
    plt.xlabel("Retained top-k"); plt.ylabel("Median retained router mass (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "router_cumulative_mass.png", dpi=180); plt.close()

    route_path = OUT / "fresh_route_policy_summary.csv"
    if route_path.exists():
        route = list(csv.DictReader(route_path.open()))
        plt.figure(figsize=(7, 4))
        for name, color in {"random": "#888888", "semantic": "#2878b5", "tail": "#d95319"}.items():
            group = [r for r in route if r["policy"].startswith(f"budget_{name}_") and
                     float(r["vision_drop_fraction_median"]) >= .05]
            group.sort(key=lambda r: float(r["vision_drop_fraction_median"]))
            plt.plot([100 * float(r["vision_drop_fraction_median"]) for r in group],
                     [float(r["max_rank_reduction_pct_median"]) for r in group], marker="o",
                     label=name, color=color)
        plt.xlabel("Vision-assignment reduction (%)"); plt.ylabel("Max-rank assignment reduction (%)")
        plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "max_rank_reduction_vs_budget.png", dpi=180); plt.close()

    chosen = [r for r in replay if r["tokens"] == 8192 and not r["renormalized"] and
              ("smallbudget" in r["source"] or "0120" in r["source"] or "k2k1" in r["source"] or "0058" in r["source"])]
    colors = {"random": "#888888", "semantic": "#2878b5", "tail": "#d95319"}
    plt.figure(figsize=(7, 4))
    for name, color in colors.items():
        vals = [r for r in chosen if name in r["policy"]]
        plt.scatter([100 * r["assignment_drop_fraction"] for r in vals],
                    [100 * r["worst_rel_l2_median"] for r in vals], label=name, color=color)
    plt.xlabel("All-assignment reduction (%)"); plt.ylabel("Local MoE relative L2 (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "quality_vs_reduction.png", dpi=180); plt.close()

    vals = [r for r in projections_ if r["tokens"] == 8192 and not r["renormalized"]]
    plt.figure(figsize=(7, 4))
    for name, color in colors.items():
        g = [r for r in vals if name in r["policy"]]
        plt.scatter([100 * r["worst_rel_l2_median"] for r in g], [r["projected_ttft_gain_pct"] for r in g],
                    label=name, color=color)
    plt.axhline(8, color="black", linestyle="--", linewidth=.8); plt.xlabel("Local MoE relative L2 (%)")
    plt.ylabel("Projected clean TTFT gain (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "projected_ttft_quality_pareto.png", dpi=180); plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    router = router_mass(); attention = attention_summary(); layers = layer_bottleneck(); replay = replay_summary()
    projections_ = projections(replay)
    atlas = json.loads((RESULTS / "bottleneck_analysis_20260911_0051" / "summary.json").read_text())["rows"]
    plots(atlas, layers, replay, projections_, router)
    summary = {"router_rows": len(router), "attention": attention, "layer_rows": len(layers),
               "replay_conditions": len(replay), "projection_conditions": len(projections_)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
