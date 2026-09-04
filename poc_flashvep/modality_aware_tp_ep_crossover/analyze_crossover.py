"""Analyze paired TP-only/EP4 real Qwen3-VL crossover traces."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_worker(path: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(path.glob("rank*_pid*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("measured"):
                row["rank_file"] = f.name
                row["expert_histogram"] = {int(k): int(v) for k, v in row.get("expert_histogram", {}).items()}
                row["rank_histogram_ep4"] = {int(k): int(v) for k, v in row.get("rank_histogram_ep4", {}).items()}
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cv(values: list[float]) -> float:
    m = _mean(values)
    return math.sqrt(_mean([(v - m) ** 2 for v in values])) / m if m else 0.0


def aggregate(run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_rows = read_csv(run / "request_wall.csv")
    workers = read_worker(run / "worker_raw")
    by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in workers:
        by_key[(str(r.get("workload")), int(r.get("iteration", -1)), int(r.get("layer", -1)))].append(r)
    layer_rows: list[dict[str, Any]] = []
    for (workload, iteration, layer), rs in sorted(by_key.items()):
        if layer < 0:
            continue
        def mx(key: str) -> float:
            return max(float(r.get(key) or 0.0) for r in rs)
        def mn(key: str) -> float:
            vals = [float(r.get(key) or 0.0) for r in rs]
            return min(vals)
        ref = rs[0]
        rh = ref.get("rank_histogram_ep4", {})
        rank_values = [float(rh.get(i, 0)) for i in range(4)]
        # The routing histogram is captured before dispatch and is identical
        # on TP ranks; use one rank to avoid fourfold counting.
        eh = ref.get("expert_histogram", {})
        expert_values = [float(v) for v in eh.values()]
        layer_rows.append({
            "topology": run.name, "workload": workload, "iteration": iteration,
            "layer": layer, "rank_count": len(rs),
            "full_moe_ms": mx("full_moe_ms"), "dispatch_ms": mx("dispatch_ms"),
            "expert_ms": mx("expert_ms"), "combine_ms": mx("combine_ms"),
            "full_moe_mean_rank_ms": _mean([float(r.get("full_moe_ms") or 0.0) for r in rs]),
            "expert_max_mean_rank": (mx("expert_ms") / _mean([float(r.get("expert_ms") or 0.0) for r in rs])
                                     if _mean([float(r.get("expert_ms") or 0.0) for r in rs]) else 0.0),
            "rank_max_mean_load": (max(rank_values) / _mean(rank_values) if _mean(rank_values) else 0.0),
            "rank_cv": _cv(rank_values), "active_experts": len(eh),
            "expert_load_cv": _cv(expert_values),
            "max_expert_load": max(expert_values) if expert_values else 0.0,
            "mean_active_expert_load": _mean(expert_values),
            "total_assignments": int(sum(expert_values)),
            "critical_rank": int(max(range(4), key=lambda i: rank_values[i])) if rank_values else -1,
            "prepare_finalize_backend": ref.get("prepare_finalize_backend", "unknown"),
            "expert_backend": ref.get("expert_backend", "unknown"),
        })
    request_out: list[dict[str, Any]] = []
    by_req: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in layer_rows:
        by_req[(r["workload"], r["iteration"])].append(r)
    wall = {(r["workload"], int(r["iteration"])): r for r in request_rows}
    for (workload, iteration), rs in sorted(by_req.items()):
        w = wall.get((workload, iteration), {})
        row = {"topology": run.name, "workload": workload, "iteration": iteration,
               "wall_ms": float(w.get("wall_ms", 0.0)), "prompt_tokens": int(w.get("prompt_tokens", 0)),
               "layers": len(rs), "t_moe_ms": sum(r["full_moe_ms"] for r in rs),
               "dispatch_ms": sum(r["dispatch_ms"] for r in rs),
               "expert_ms": sum(r["expert_ms"] for r in rs),
               "combine_ms": sum(r["combine_ms"] for r in rs),
               "mean_active_experts": _mean([r["active_experts"] for r in rs]),
               "mean_expert_load_cv": _mean([r["expert_load_cv"] for r in rs]),
               "mean_rank_load_cv": _mean([r["rank_cv"] for r in rs]),
               "mean_rank_max_mean": _mean([r["rank_max_mean_load"] for r in rs]),
               "mean_expert_max_mean": _mean([r["expert_max_mean_rank"] for r in rs]),
               "max_expert_load": max(r["max_expert_load"] for r in rs),
               "prepare_finalize_backend": rs[0]["prepare_finalize_backend"],
               "expert_backend": rs[0]["expert_backend"]}
        request_out.append(row)
    return layer_rows, request_out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def make_figures(out: Path, req: list[dict[str, Any]], layer: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out / "figures_unavailable.txt").write_text(repr(exc) + "\n")
        return
    topo = sorted({r["topology"] for r in req})
    workloads = ["text_heavy", "mixed", "vision_heavy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ti, t in enumerate(topo):
        vals = []
        for w in workloads:
            x = [r["wall_ms"] for r in req if r["topology"] == t and r["workload"] == w]
            vals.append(sum(x) / len(x) if x else 0)
        axes[0].bar([i + (ti - .5) * .35 for i in range(3)], vals, .35, label=t)
        vals = []
        for w in workloads:
            x = [r["t_moe_ms"] for r in req if r["topology"] == t and r["workload"] == w]
            vals.append(sum(x) / len(x) if x else 0)
        axes[1].bar([i + (ti - .5) * .35 for i in range(3)], vals, .35, label=t)
    for ax, title in zip(axes, ["request wall", "sum of layer T_MoE"], strict=True):
        ax.set_xticks(range(3), workloads); ax.set_ylabel("ms"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out / "tp_ep_latency_breakdown.png", dpi=160); plt.close(fig)
    # Composition scatter; actual image assignment fraction is unavailable
    # from this read-only route hook, so use recorded prompt visual class and
    # explicitly label the x-axis categorical modality.
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in topo:
        for wi, w in enumerate(workloads):
            x = [r["t_moe_ms"] / max(1, r["prompt_tokens"]) for r in req if r["topology"] == t and r["workload"] == w]
            if x:
                ax.scatter([wi + (0.08 if t == topo[-1] else -0.08)] * len(x), x, label=t if wi == 0 else None)
    ax.set_xticks(range(3), workloads); ax.set_ylabel("T_MoE ms/token"); ax.set_title("Modality composition vs MoE cost"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "moe_cost_by_modality.png", dpi=160); plt.close(fig)
    # Per-layer robustness of TP/EP ratio.
    fig, ax = plt.subplots(figsize=(10, 4))
    for w in workloads:
        e = {(r["iteration"], r["layer"]): r["full_moe_ms"] for r in layer if r["topology"].endswith("ep4") and r["workload"] == w}
        t = {(r["iteration"], r["layer"]): r["full_moe_ms"] for r in layer if r["topology"].endswith("tp_only") and r["workload"] == w}
        xs = sorted(set(e) & set(t))
        if xs:
            ratios = [e[k] / t[k] if t[k] else 0 for k in xs]
            ax.plot(range(len(ratios)), ratios, label=w, alpha=.75)
    ax.axhline(1.0, color="black", lw=.8); ax.set_ylabel("EP4 / TP-only layer MoE"); ax.set_xlabel("paired layer invocation"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "per_layer_ep_tp_ratio.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ep4", type=Path, required=True); ap.add_argument("--tp-only", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    ep_layer, ep_req = aggregate(a.ep4); tp_layer, tp_req = aggregate(a.tp_only)
    layer = ep_layer + tp_layer; req = ep_req + tp_req
    write_csv(a.output / "layer_metrics.csv", layer); write_csv(a.output / "request_metrics.csv", req)
    # Stable route/load alias for downstream analysis and audit.  Keep timing
    # in layer_metrics.csv, while this table is intentionally route-focused.
    write_csv(a.output / "route_statistics.csv", [
        {k: v for k, v in row.items() if k not in {
            "full_moe_ms", "dispatch_ms", "expert_ms", "combine_ms",
            "full_moe_mean_rank_ms", "expert_max_mean_rank"
        }} for row in layer
    ])
    # Paired topology comparison, retaining every measured repetition.
    ep = {(r["workload"], r["iteration"]): r for r in ep_req}; tp = {(r["workload"], r["iteration"]): r for r in tp_req}
    pairs = []
    for key in sorted(set(ep) & set(tp)):
        e, t = ep[key], tp[key]
        pairs.append({"workload": key[0], "iteration": key[1], "prompt_tokens": e["prompt_tokens"],
                      "tp_wall_ms": t["wall_ms"], "ep_wall_ms": e["wall_ms"],
                      "wall_ep_vs_tp_reduction_pct": 100*(t["wall_ms"]-e["wall_ms"])/t["wall_ms"] if t["wall_ms"] else 0,
                      "tp_t_moe_ms": t["t_moe_ms"], "ep_t_moe_ms": e["t_moe_ms"],
                      "t_moe_ep_vs_tp_reduction_pct": 100*(t["t_moe_ms"]-e["t_moe_ms"])/t["t_moe_ms"] if t["t_moe_ms"] else 0,
                      "tp_dispatch_ms": t["dispatch_ms"], "ep_dispatch_ms": e["dispatch_ms"],
                      "tp_expert_ms": t["expert_ms"], "ep_expert_ms": e["expert_ms"],
                      "tp_combine_ms": t["combine_ms"], "ep_combine_ms": e["combine_ms"],
                      "tp_mean_active_experts": t["mean_active_experts"], "ep_mean_active_experts": e["mean_active_experts"],
                      "tp_mean_expert_load_cv": t["mean_expert_load_cv"], "ep_mean_expert_load_cv": e["mean_expert_load_cv"],
                      "ep_prepare_finalize_backend": e["prepare_finalize_backend"],
                      "ep_expert_backend": e["expert_backend"]})
    write_csv(a.output / "paired_comparisons.csv", pairs)
    summary: dict[str, Any] = {"status": "complete", "ep4_run": str(a.ep4), "tp_only_run": str(a.tp_only), "pairs": len(pairs), "workloads": {}}
    for w in ("text_heavy", "mixed", "vision_heavy"):
        p = [r for r in pairs if r["workload"] == w]
        summary["workloads"][w] = {"n": len(p),
            "prompt_tokens": sorted({r["prompt_tokens"] for r in p}),
            "median_wall_reduction_pct": statistics.median(
                r["wall_ep_vs_tp_reduction_pct"] for r in p) if p else None,
            "median_t_moe_reduction_pct": statistics.median(
                r["t_moe_ep_vs_tp_reduction_pct"] for r in p) if p else None,
            "mean_tp_t_moe_ms": _mean([r["tp_t_moe_ms"] for r in p]),
            "mean_ep_t_moe_ms": _mean([r["ep_t_moe_ms"] for r in p]),
            "phase_mean_tp_ms": {k: _mean([r[f"tp_{k}_ms"] for r in p]) for k in ("dispatch", "expert", "combine")},
            "phase_mean_ep_ms": {k: _mean([r[f"ep_{k}_ms"] for r in p]) for k in ("dispatch", "expert", "combine")},
            "ep_prepare_finalize_backend": p[0].get("ep_prepare_finalize_backend") if p else None,
            "ep_expert_backend": p[0].get("ep_expert_backend") if p else None}
    # Gate is deliberately conservative: TP/EP static crossover is not called
    # if token matching is poor or the paired T_MoE effect is below 5%.
    all_reductions = [r["t_moe_ep_vs_tp_reduction_pct"] for r in pairs]
    median_all = statistics.median(all_reductions) if all_reductions else 0.0
    summary["overall_median_t_moe_reduction_pct"] = median_all
    summary["raw_reduction_decision"] = (
        "GO" if median_all >= 5 else "HOLD" if median_all >= 2 else "NO_GO"
    )
    # A lower EP-flag latency is not a modality crossover.  The final gate is
    # overridden after the runtime semantic audit because TP4/DP1 cannot
    # activate DeepEP all-to-all in this vLLM build.
    summary["decision"] = "NO_GO"
    summary["requested_gate_note"] = (
        "The requested TP4/DP1 EP4 flag path does not activate vLLM all-to-all "
        "kernels in this local build: use_all2all_kernels requires dp_size > 1. "
        "Thus the paired EP flag path is an expert-sharded/no-DP-EP path, not "
        "a valid DeepEP topology crossover."
    )
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figures(a.output, req, layer)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
