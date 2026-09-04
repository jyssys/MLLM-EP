"""Analyze the final TP-only versus real DeepEP volume/modality run.

The runner intentionally captures each vLLM DP worker.  A DP participant that
does not own the request emits a small dummy forward (8 assignments); those
records are retained in the raw trace but excluded from the request-level
comparison.  Actual request rows are then paired by workload and iteration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# Counts measured with the same Qwen3-VL processor before the GPU run.  They
# make the modality fraction explicit instead of inferring it from image count.
VISION_TOKENS = {
    "small_text_heavy": 0, "medium_text_heavy": 0, "large_text_heavy": 0,
    "small_mixed": 196, "medium_mixed": 1568, "large_mixed": 4900,
    "small_vision_heavy": 196, "medium_vision_heavy": 1568,
    "large_vision_heavy": 7350,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _cv(xs: list[float]) -> float:
    m = _mean(xs)
    return math.sqrt(_mean([(x - m) ** 2 for x in xs])) / m if m else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = _mean(xs), _mean(ys)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def _route_values(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Sum the per-TP route histograms into global EP4 diagnostics."""
    rank = [0.0] * 4
    experts: dict[int, float] = defaultdict(float)
    for row in rows:
        for k, v in row.get("rank_histogram_ep4", {}).items():
            rank[int(k)] += _num(v)
        for k, v in row.get("expert_histogram", {}).items():
            experts[int(k)] += _num(v)
    return rank, list(experts.values())


def _load_rows(run: Path, is_ep: bool) -> list[dict[str, Any]]:
    """Read measured rows and select one actual call per wave/layer/rank."""
    selected: list[dict[str, Any]] = []
    for path in sorted((run / "worker_raw").glob("rank*_pid*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("measured"):
                continue
            row["expert_histogram"] = {
                int(k): int(v) for k, v in row.get("expert_histogram", {}).items()
            }
            row["rank_histogram_ep4"] = {
                int(k): int(v) for k, v in row.get("rank_histogram_ep4", {}).items()
            }
            row["rank_file"] = path.name
            selected.append(row)
    # EP DP participants emit many padding/dummy forwards.  The real request
    # has a much larger assignment count for every workload in this run.
    if is_ep:
        selected = [r for r in selected if int(r.get("total_assignments") or 0) > 8]
    # De-duplicate if a process wrote a late shutdown copy.  Keep the row with
    # the largest assignment volume, which is the real request rather than a
    # DP padding call.
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in selected:
        key = (row.get("wave"), row.get("layer"), row.get("local_rank"))
        old = by_key.get(key)
        if old is None or int(row.get("total_assignments") or 0) > int(old.get("total_assignments") or 0):
            by_key[key] = row
    return list(by_key.values())


def _request_id(row: dict[str, Any]) -> str:
    wave = str(row.get("wave") or "")
    # Runner wave names are e.g. measure-9-small_text_heavy.  Keeping the
    # request id is necessary because workload alone is shared by 3 volumes.
    if "-" in wave:
        return wave.split("-", 2)[-1]
    return str(row.get("request_id") or row.get("workload") or "unknown")


def aggregate(run: Path, is_ep: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_wall = read_csv(run / "request_wall.csv")
    rows = _load_rows(run, is_ep)
    # DP0 owns the requests.  Actual rows are still grouped by workload,
    # iteration and layer; local TP ranks are the critical-path participants.
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer = int(row.get("layer", -1))
        if layer >= 0:
            groups[(_request_id(row), int(row.get("iteration", -1)), layer)].append(row)
    layer_out: list[dict[str, Any]] = []
    for (request_id, iteration, layer), rs in sorted(groups.items()):
        # Every TP rank has the same routed activation in TP-only; for EP the
        # two source TP ranks partition it, so sum route histograms for shape.
        rank_vals, expert_vals = _route_values(rs)
        ref = rs[0]
        full = [_num(r.get("full_moe_ms")) for r in rs]
        dispatch = [_num(r.get("dispatch_ms")) for r in rs]
        expert = [_num(r.get("expert_ms")) for r in rs]
        combine = [_num(r.get("combine_ms")) for r in rs]
        m_rank = _mean(rank_vals)
        m_expert_load = _mean(expert_vals)
        m_expert_time = _mean(expert)
        workload = str(ref.get("workload") or "unknown")
        modality = workload
        layer_out.append({
            "topology": "REAL_DEEPEP" if is_ep else "TP_ONLY",
            "request_id": request_id, "workload": workload,
            "volume": request_id.split("_", 1)[0], "modality": modality, "iteration": iteration,
            "layer": layer, "rank_count": len(rs),
            "full_moe_ms": max(full), "dispatch_ms": max(dispatch),
            "expert_ms": max(expert), "combine_ms": max(combine),
            "mean_full_rank_ms": _mean(full), "mean_expert_rank_ms": m_expert_time,
            "expert_max_mean_rank": max(expert) / m_expert_time if m_expert_time else 0.0,
            "rank_max_mean_load": max(rank_vals) / m_rank if m_rank else 0.0,
            "rank_cv": _cv(rank_vals), "active_experts": len(expert_vals),
            "expert_load_cv": _cv(expert_vals),
            "max_expert_load": max(expert_vals) if expert_vals else 0.0,
            "mean_expert_load": m_expert_load,
            "total_assignments_local_sum": int(sum(_num(r.get("total_assignments")) for r in rs)),
            "rank_loads": json.dumps([int(x) for x in rank_vals]),
            "critical_rank": int(max(range(4), key=lambda i: rank_vals[i])) if rank_vals else -1,
            "prepare_finalize_backend": ref.get("prepare_finalize_backend", "unknown"),
            "expert_backend": ref.get("expert_backend", "unknown"),
        })
    wall = {(r.get("request_id", r["workload"]), int(r["iteration"])): r for r in request_wall}
    by_req: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in layer_out:
        by_req[(row["request_id"], row["iteration"])].append(row)
    req_out: list[dict[str, Any]] = []
    for key, rs in sorted(by_req.items()):
        w = wall.get(key, {})
        request_id = key[0]
        prompt = int(w.get("prompt_tokens", 0))
        vision = VISION_TOKENS.get(request_id, 0)
        t = {
            "topology": "REAL_DEEPEP" if is_ep else "TP_ONLY",
            "request_id": request_id, "workload": rs[0]["workload"],
            "volume": rs[0]["volume"], "modality": rs[0]["modality"],
            "iteration": key[1], "wall_ms": _num(w.get("wall_ms")),
            "prompt_tokens": prompt, "vision_tokens": vision,
            "vision_fraction": vision / prompt if prompt else 0.0,
            "layers": len(rs), "t_moe_ms": sum(r["full_moe_ms"] for r in rs),
            "dispatch_ms": sum(r["dispatch_ms"] for r in rs),
            "expert_ms": sum(r["expert_ms"] for r in rs),
            "combine_ms": sum(r["combine_ms"] for r in rs),
            "t_moe_ms_per_token": sum(r["full_moe_ms"] for r in rs) / prompt if prompt else 0.0,
            "mean_active_experts": _mean([r["active_experts"] for r in rs]),
            "mean_expert_load_cv": _mean([r["expert_load_cv"] for r in rs]),
            "mean_rank_load_cv": _mean([r["rank_cv"] for r in rs]),
            "mean_rank_max_mean": _mean([r["rank_max_mean_load"] for r in rs]),
            "mean_expert_max_mean": _mean([r["expert_max_mean_rank"] for r in rs]),
            "max_expert_load": max(r["max_expert_load"] for r in rs),
            "prepare_finalize_backend": rs[0]["prepare_finalize_backend"],
            "expert_backend": rs[0]["expert_backend"],
        }
        req_out.append(t)
    return layer_out, req_out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def figures(out: Path, pairs: list[dict[str, Any]], req: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out / "figures_unavailable.txt").write_text(repr(exc) + "\n", encoding="utf-8"); return
    # Workload-composition relative gain (positive = DeepEP lower T_MoE).
    fig, ax = plt.subplots(figsize=(9, 4))
    for w in sorted({r["request_id"] for r in pairs}):
        xs = [r["vision_fraction"] for r in pairs if r["request_id"] == w]
        ys = [r["t_moe_ep_vs_tp_reduction_pct"] for r in pairs if r["request_id"] == w]
        ax.plot(xs, ys, "o-", label=w)
    ax.axhline(0, color="black", lw=.8); ax.set_xlabel("vision assignment proxy (processor vision tokens / prompt tokens)")
    ax.set_ylabel("REAL_DEEPEP T_MoE reduction vs TP_ONLY (%)"); ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(out / "vision_fraction_vs_relative_gain.png", dpi=160); plt.close(fig)
    # Phase breakdown by volume and topology.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ti, phase in enumerate(("dispatch_ms", "expert_ms", "combine_ms")):
        for topo in ("TP_ONLY", "REAL_DEEPEP"):
            vals=[]
            for v in ("small", "medium", "large"):
                z=[r[phase] for r in req if r["topology"]==topo and r["volume"]==v]
                vals.append(_mean(z))
            axes[ti].plot((0,1,2), vals, "o-", label=topo)
        axes[ti].set_title(phase); axes[ti].set_xticks((0,1,2),("small","medium","large")); axes[ti].set_ylabel("ms")
    axes[0].legend(); fig.tight_layout(); fig.savefig(out / "phase_breakdown_by_volume.png", dpi=160); plt.close(fig)
    # T_MoE vs actual processor prompt tokens.
    fig, ax = plt.subplots(figsize=(7, 4))
    for topo, marker in (("TP_ONLY", "o"), ("REAL_DEEPEP", "s")):
        z=[r for r in req if r["topology"]==topo]
        ax.scatter([r["prompt_tokens"] for r in z], [r["t_moe_ms"] for r in z], label=topo, marker=marker)
    ax.set_xlabel("prompt tokens"); ax.set_ylabel("T_MoE (ms, sum of layer critical spans)"); ax.legend(); fig.tight_layout(); fig.savefig(out / "tmoe_vs_prompt_tokens.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp-only", type=Path, required=True)
    ap.add_argument("--real-deepep", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    tl, tr = aggregate(args.tp_only, False); el, er = aggregate(args.real_deepep, True)
    all_layer, all_req = tl + el, tr + er
    write_csv(args.output / "layer_metrics.csv", all_layer); write_csv(args.output / "request_metrics.csv", all_req)
    # Pair the first two common repetitions (the TP run has one extra repeat).
    tp = {(r["request_id"], r["iteration"]): r for r in tr}
    ep = {(r["request_id"], r["iteration"]): r for r in er}
    pairs=[]
    for key in sorted(set(tp)&set(ep)):
        t,e=tp[key],ep[key]
        pairs.append({
            "request_id": key[0], "volume": t["volume"], "modality": t["modality"], "iteration": key[1],
            "prompt_tokens": t["prompt_tokens"], "vision_tokens": t["vision_tokens"], "vision_fraction": t["vision_fraction"],
            "tp_wall_ms": t["wall_ms"], "ep_wall_ms": e["wall_ms"],
            "wall_ep_vs_tp_reduction_pct": 100*(t["wall_ms"]-e["wall_ms"])/t["wall_ms"] if t["wall_ms"] else 0.0,
            "tp_t_moe_ms": t["t_moe_ms"], "ep_t_moe_ms": e["t_moe_ms"],
            "t_moe_ep_vs_tp_reduction_pct": 100*(t["t_moe_ms"]-e["t_moe_ms"])/t["t_moe_ms"] if t["t_moe_ms"] else 0.0,
            "tp_dispatch_ms":t["dispatch_ms"],"ep_dispatch_ms":e["dispatch_ms"],
            "tp_expert_ms":t["expert_ms"],"ep_expert_ms":e["expert_ms"],
            "tp_combine_ms":t["combine_ms"],"ep_combine_ms":e["combine_ms"],
            "tp_t_moe_ms_per_token":t["t_moe_ms_per_token"],"ep_t_moe_ms_per_token":e["t_moe_ms_per_token"],
            "tp_mean_rank_max_mean":t["mean_rank_max_mean"],"ep_mean_rank_max_mean":e["mean_rank_max_mean"],
            "tp_mean_active_experts":t["mean_active_experts"],"ep_mean_active_experts":e["mean_active_experts"],
            "ep_prepare_finalize_backend":e["prepare_finalize_backend"],"ep_expert_backend":e["expert_backend"],
        })
    write_csv(args.output / "paired_comparisons.csv", pairs)
    # Greedy output and processor prompt length are a direct correctness
    # control for the static-topology comparison.
    tp_wall_rows = read_csv(args.tp_only / "request_wall.csv")
    ep_wall_rows = read_csv(args.real_deepep / "request_wall.csv")
    tp_ids = {(r["request_id"], int(r["iteration"])): (r.get("prompt_tokens"), r.get("output_token_ids")) for r in tp_wall_rows}
    ep_ids = {(r["request_id"], int(r["iteration"])): (r.get("prompt_tokens"), r.get("output_token_ids")) for r in ep_wall_rows}
    mismatches = []
    for key in sorted(set(tp_ids) & set(ep_ids)):
        if tp_ids[key] != ep_ids[key]:
            mismatches.append({"request_id": key[0], "iteration": key[1], "tp": tp_ids[key], "real_deepep": ep_ids[key]})
    (args.output / "correctness_check.json").write_text(json.dumps({"pairs": len(set(tp_ids) & set(ep_ids)), "mismatches": mismatches, "pass": not mismatches}, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output / "route_statistics.csv", [r for r in all_layer])
    by_volume: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs: by_volume[p["volume"]].append(p); by_modality[p["modality"]].append(p)
    def med(rows: list[dict[str, Any]], key: str) -> float | None:
        return statistics.median([_num(r[key]) for r in rows]) if rows else None
    volume_summary=[]
    for v in ("small","medium","large"):
        ps=by_volume[v]
        volume_summary.append({"volume":v,"n":len(ps),"median_prompt_tokens":med(ps,"prompt_tokens"),"median_tmoe_reduction_pct":med(ps,"t_moe_ep_vs_tp_reduction_pct"),"median_wall_reduction_pct":med(ps,"wall_ep_vs_tp_reduction_pct")})
    modality_summary=[]
    for m in ("text_heavy","mixed","vision_heavy"):
        ps=by_modality[m]
        modality_summary.append({"modality":m,"n":len(ps),"median_vision_fraction":med(ps,"vision_fraction"),"median_tmoe_reduction_pct":med(ps,"t_moe_ep_vs_tp_reduction_pct"),"median_wall_reduction_pct":med(ps,"wall_ep_vs_tp_reduction_pct"),"median_tp_tmoe_ms":med(ps,"tp_t_moe_ms"),"median_ep_tmoe_ms":med(ps,"ep_t_moe_ms")})
    write_csv(args.output / "volume_summary.csv", volume_summary); write_csv(args.output / "modality_summary.csv", modality_summary)
    # Keep the key matched comparison explicit: modality can only explain a
    # topology crossover if signs differ *within the same volume*.  A sign
    # change from small to large is a load/shape effect, not a modality effect.
    modality_by_volume = []
    volume_crossover = False
    for v in ("small", "medium", "large"):
        for m in ("text_heavy", "mixed", "vision_heavy"):
            z = [p for p in pairs if p["volume"] == v and p["modality"] == m]
            modality_by_volume.append({"volume": v, "modality": m, "n": len(z),
                                       "median_tmoe_reduction_pct": med(z, "t_moe_ep_vs_tp_reduction_pct"),
                                       "median_vision_fraction": med(z, "vision_fraction")})
        zs = [p["t_moe_ep_vs_tp_reduction_pct"] for p in pairs if p["volume"] == v]
        if zs and min(zs) < 0 < max(zs):
            volume_crossover = True
    write_csv(args.output / "modality_by_volume.csv", modality_by_volume)
    # Simple explanatory-power diagnostics on paired observations.
    gains=[_num(p["t_moe_ep_vs_tp_reduction_pct"]) for p in pairs]; tokens=[_num(p["prompt_tokens"]) for p in pairs]; vf=[_num(p["vision_fraction"]) for p in pairs]
    stats={"n_pairs":len(pairs),"corr_gain_prompt_tokens":_pearson(gains,tokens),"corr_gain_vision_fraction":_pearson(gains,vf),"corr_tmoe_prompt_tokens":_pearson([_num(p["tp_t_moe_ms"]) for p in pairs],tokens),"backend_proof":{"tp_prepare_finalize":sorted({r["prepare_finalize_backend"] for r in tl}),"real_deepep_prepare_finalize":sorted({r["prepare_finalize_backend"] for r in el}),"real_deepep_expert":sorted({r["expert_backend"] for r in el})}}
    stats["all_medians"]={"tmoe_reduction_pct":med(pairs,"t_moe_ep_vs_tp_reduction_pct"),"wall_reduction_pct":med(pairs,"wall_ep_vs_tp_reduction_pct")}
    stats["volume_summary"]=volume_summary; stats["modality_summary"]=modality_summary
    stats["actual_prompt_processor_counts"]=VISION_TOKENS
    (args.output / "analysis_summary.json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
    # No post-hoc gate changes: modality-aware crossover requires a repeated
    # topology winner change; a volume-only advantage is HOLD, not GO.
    effect_by_volume=[_num(r["median_tmoe_reduction_pct"]) for r in volume_summary if r["median_tmoe_reduction_pct"] is not None]
    ranges = []
    for v in ("small", "medium", "large"):
        z = [_num(p["t_moe_ep_vs_tp_reduction_pct"]) for p in pairs if p["volume"] == v]
        if z: ranges.append(max(z) - min(z))
    modality_effect = max(ranges, default=0.0)
    if volume_crossover and modality_effect >= 10: decision="STRONG_GO"
    elif volume_crossover and modality_effect >= 5: decision="GO"
    elif max(effect_by_volume, default=0.0) >= 5: decision="HOLD"
    else: decision="NO_GO"
    gate={"decision":decision,"deepep_runtime_verified":all(r["prepare_finalize_backend"]=="DeepEPHTPrepareAndFinalize" for r in el),"n_pairs":len(pairs),"modality_effect_max_within_volume_range_pct":modality_effect,"modality_crossover_within_volume":volume_crossover,"median_tmoe_reduction_pct":stats["all_medians"]["tmoe_reduction_pct"],"rule":"Only a repeated modality-dependent topology crossover within matched volume qualifies for GO; volume-only gains are HOLD."}
    (args.output / "gate_summary.json").write_text(json.dumps(gate,indent=2)+"\n",encoding="utf-8")
    figures(args.output,pairs,all_req)
    print(json.dumps({"gate":gate,"stats":stats},indent=2))


if __name__ == "__main__":
    main()
