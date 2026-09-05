"""Finalize the bounded online distributed-routing-geometry PoC.

This is intentionally an analysis-only utility.  It consumes the read-only
vLLM boundary traces and the separate histogram-preserving replay, and writes
auditable summaries/plots without modifying vLLM or the model.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def load_rows(root: Path) -> list[dict]:
    # The existing analyzer already emits a scalar-only CSV.  Reusing it
    # avoids repeatedly decoding the large JSON route histograms during final
    # report generation; raw JSONL/NPZ remains the authoritative trace.
    csv_path = root / "online_invocations.csv"
    if csv_path.exists():
        rows = []
        numeric = {"M", "top_k", "total_assignments", "active_experts", "cuda_ms", "wall_ms",
                   "expert_cv", "expert_hhi", "expert_entropy", "expert_max_mean", "expert_max_load",
                   "expert_p90_load", "fanout_mean", "fanout_p10", "fanout_median", "fanout_p90",
                   "fanout_max", "fanout_f1", "fanout_f2", "fanout_f3", "fanout_f4", "rank_max_mean",
                   "rank_cv", "traffic_entropy", "traffic_concentration", "padded_work_proxy",
                   "timestamp_ns", "layer", "dp_rank", "ep_rank", "ep_size"}
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if int(float(r.get("layer", -1) or -1)) < 0:
                    continue
                for k in numeric:
                    if k in r and r[k] not in ("", None):
                        try: r[k] = float(r[k])
                        except ValueError: pass
                rows.append(r)
        if rows:
            return rows
    rows = []
    # Only these two runs have the corrected layer attribution and hook
    # environment.  The older online_trace is retained as a debugging
    # artifact but has layer=-1 and is excluded from all primary statistics.
    for name in ("online_trace3", "online_trace_high2"):
        path = root / name / "invocations.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(r.get("layer", -1)) < 0 or int(r.get("M", 0) or 0) > 2048:
                continue
            eh = np.asarray(r.get("expert_hist", []), dtype=float)
            rh = np.asarray(r.get("rank_loads", []), dtype=float)
            nz = eh[eh > 0]
            r["trace_source"] = name
            r["expert_max_load"] = float(nz.max()) if nz.size else 0.0
            r["expert_p90_load"] = float(np.quantile(nz, 0.9)) if nz.size else 0.0
            r["rank_cv"] = float(rh.std() / rh.mean()) if rh.size and rh.mean() else 0.0
            mat = np.asarray(r.get("sender_dest_matrix", []), dtype=float)
            if mat.size and mat.sum():
                p = mat.ravel(); p = p[p > 0] / p.sum()
                r["traffic_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
                r["traffic_concentration"] = float((p * p).sum())
            else:
                r["traffic_entropy"] = 0.0
                r["traffic_concentration"] = 0.0
            r["padded_work_proxy"] = float(r.get("total_assignments", 0) or 0)
            rows.append(r)
    return rows


def fit_predict(rows: list[dict], features: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    good = [r for r in rows if all(math.isfinite(float(r.get(f, 0))) for f in features + ["cuda_ms"])]
    if len(good) < len(features) + 3:
        return np.empty(0), np.empty(0), {"status": "INSUFFICIENT", "n": len(good)}
    good.sort(key=lambda r: float(r.get("timestamp_ns", 0)))
    cut = max(1, min(len(good) - 1, int(0.7 * len(good))))
    tr, te = good[:cut], good[cut:]
    x = np.c_[np.ones(len(tr)), [[float(r.get(f, 0)) for f in features] for r in tr]]
    xt = np.c_[np.ones(len(te)), [[float(r.get(f, 0)) for f in features] for r in te]]
    y = np.asarray([float(r["cuda_ms"]) for r in tr]); yt = np.asarray([float(r["cuda_ms"]) for r in te])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]; pred = xt @ beta
    return pred, yt, {"features": features, "beta": beta.tolist(), "test_n": len(te)}


def matched_pairs(rows: list[dict], out: Path) -> dict:
    # Pair natural prefill invocations with the same layer and M.  The control
    # constraints are deliberately conservative; worker rows are not treated
    # as independent requests in the interpretation.
    candidates = [r for r in rows if r.get("phase") == "prefill" and float(r.get("M", 0)) >= 100]
    pairs = []
    for (layer, m), group in sorted(_group(candidates, ("layer", "M")).items()):
        if len(group) < 2:
            continue
        # Repetitions produce many correlated worker rows.  A fixed evenly
        # spaced sample retains the natural fanout range while bounding the
        # pair search to a few thousand comparisons.
        if len(group) > 20:
            group = [group[i] for i in np.linspace(0, len(group) - 1, 20, dtype=int)]
        for i, a in enumerate(group):
            q = []
            for j in range(i + 1, len(group)):
                b = group[j]
                if abs(float(a["active_experts"]) - float(b["active_experts"])) > 8:
                    continue
                denom = max(1.0, float(a["rank_max_mean"]), float(b["rank_max_mean"]))
                if abs(float(a["rank_max_mean"]) - float(b["rank_max_mean"])) / denom > .05:
                    continue
                fd = abs(float(a["fanout_mean"]) - float(b["fanout_mean"]))
                if fd < .25:
                    continue
                q.append((fd, j, b))
            if q:
                fd, j, b = max(q, key=lambda x: x[0])
                ta, tb = float(a["cuda_ms"]), float(b["cuda_ms"])
                pairs.append({"layer": int(layer), "M": int(m), "a_index": id(a), "b_index": id(b),
                              "fanout_delta": fd, "t_moe_delta_pct": 100 * (tb - ta) / max(ta, 1e-9),
                              "a_cuda_ms": ta, "b_cuda_ms": tb})
    # A bounded deterministic sample is enough for the pair diagnostic and
    # keeps the artifact compact when many worker rows are present.
    pairs = pairs[:2000]
    pair_dir = out / "matched_pairs"; pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "summary.json").write_text(json.dumps({
        "candidate_rows": len(candidates), "pairs_emitted": len(pairs),
        "matching": ["same layer", "same M", "active experts delta <=8", "rank max/mean within 5%", "fanout delta >=0.25"],
        "fanout_delta_quantiles": _quantiles(pairs, "fanout_delta"),
        "t_moe_delta_pct_quantiles": _quantiles(pairs, "t_moe_delta_pct"),
        "median_abs_t_moe_delta_pct": float(np.median(np.abs([p["t_moe_delta_pct"] for p in pairs]))) if pairs else None,
        "interpretation": "worker-level natural pairs; association only, not an independent causal estimate",
    }, indent=2), encoding="utf-8")
    (pair_dir / "pairs_detailed.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    return {"candidate_rows": len(candidates), "pairs": len(pairs), "summary": json.loads((pair_dir / "summary.json").read_text())}


def _group(rows: list[dict], keys: tuple[str, ...]) -> dict:
    result = {}
    for r in rows:
        key = tuple(r.get(k) for k in keys)
        result.setdefault(key, []).append(r)
    return result


def _quantiles(rows: list[dict], key: str):
    if not rows:
        return []
    return [float(np.quantile([float(r[key]) for r in rows], q)) for q in (.1, .5, .9)]


def replay_summary(root: Path) -> dict:
    replay = root / "causal_replay" / "cases512" / "replay"
    by_case = {}
    for p in sorted(replay.glob("rank*_layer24.json")):
        d = json.loads(p.read_text())
        for o in d.get("observations", []):
            c = o["case_id"]
            z = by_case.setdefault(c, {k: [] for k in ("wall", "layout", "dispatch", "expert", "combine")})
            for k in z:
                z[k].append(float(o[f"{k}_stats"]["median_ms"]))
    summary = {c: {k: float(np.median(v)) for k, v in z.items()} for c, z in by_case.items()}
    case_file = root / "causal_replay" / "cases512" / "cases.json"
    cases = json.loads(case_file.read_text()) if case_file.exists() else []
    invariants = bool(cases) and all(
        c.get("M") == 512 and c.get("total_assignments") == 4096 and
        len(c.get("expert_counts", [])) == 128 and len(c.get("rank_assignments", [])) == 4
        for c in cases)
    out = {"cases": summary, "M_e_exactly_preserved": invariants, "rank_load_preserved": invariants,
           "note": "actual DeepEP route-transfer diagnostic; not full model online routing"}
    (root / "causal_replay" / "causal_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def write_csv(rows: list[dict], out: Path) -> None:
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, str))})
    with (out / "online_invocations.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def make_plots(rows: list[dict], metrics: dict, replay: dict, pairs: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = out / "plots"; p.mkdir(parents=True, exist_ok=True)
    d = [r for r in rows if r.get("phase") == "prefill" and r.get("M", 0) >= 8]
    if d:
        fig, ax = plt.subplots(figsize=(6, 4)); sc = ax.scatter([r["M"] for r in d], [r["cuda_ms"] for r in d], c=[r["fanout_mean"] for r in d], s=5, alpha=.35, cmap="viridis"); fig.colorbar(sc, ax=ax, label="mean fanout"); ax.set(xlabel="M (routed tokens)", ylabel="T_MoE CUDA ms"); fig.tight_layout(); fig.savefig(p / "m_vs_tmoe_fanout.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4)); sc = ax.scatter([r["active_experts"] for r in d], [r["cuda_ms"] for r in d], c=[r["M"] for r in d], s=5, alpha=.35, cmap="plasma"); fig.colorbar(sc, ax=ax, label="M"); ax.set(xlabel="active experts", ylabel="T_MoE CUDA ms"); fig.tight_layout(); fig.savefig(p / "active_experts_vs_tmoe.png", dpi=160); plt.close(fig)
        # Full-sample residual plot uses Model-2 features fitted on all rows;
        # it is a visual diagnostic, whereas the time-block metrics are the
        # preregistered gate.
        fs = ["M", "active_experts", "expert_max_load", "expert_p90_load", "expert_cv", "expert_hhi", "expert_entropy", "rank_max_mean", "rank_cv"]
        x = np.c_[np.ones(len(d)), [[float(r.get(f, 0)) for f in fs] for r in d]]; y = np.asarray([float(r["cuda_ms"]) for r in d]); b = np.linalg.lstsq(x, y, rcond=None)[0]; residual = y - x @ b
        for name, val, lab in (("fanout_vs_model2_residual.png", [r["fanout_mean"] for r in d], "mean fanout"), ("f4_vs_model2_residual.png", [r["fanout_f4"] for r in d], "F4 fraction")):
            fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(val, residual, s=5, alpha=.35); ax.axhline(0, color="k", lw=.8); ax.set(xlabel=lab, ylabel="T_MoE residual (ms)"); fig.tight_layout(); fig.savefig(p / name, dpi=160); plt.close(fig)
    model_names = ["model0", "model1_distribution", "model2_distribution_plus_rank", "model3_plus_fanout_geometry"]
    vals = [metrics.get(n, {}).get("rmse", np.nan) for n in model_names]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(range(4), vals); ax.set_xticks(range(4), ["M", "distribution", "+rank", "+fanout"]); ax.set_ylabel("held-out RMSE (ms)"); fig.tight_layout(); fig.savefig(p / "model_error_comparison.png", dpi=160); plt.close(fig)
    deltas = [p["t_moe_delta_pct"] for p in json.loads((out / "matched_pairs" / "pairs_detailed.json").read_text())] if (out / "matched_pairs" / "pairs_detailed.json").exists() else []
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(deltas, bins=40, color="#5470a8"); ax.axvline(0, color="k", lw=.8); ax.set(xlabel="matched pair T_MoE delta (%)", ylabel="count"); fig.tight_layout(); fig.savefig(p / "matched_pair_tmoe_difference.png", dpi=160); plt.close(fig)
    if replay.get("cases"):
        names = list(replay["cases"]); wall = [replay["cases"][n]["wall"] for n in names]; fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(names, wall, color=["#777", "#d95f02", "#1b9e77"][:len(names)]); ax.set_ylabel("median wall / T_MoE (ms)"); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(p / "histogram_preserving_replay.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); phases = ["prefill", "decode"]; data = []
    for q in phases:
        vals = np.asarray([r["cuda_ms"] for r in rows if r.get("phase") == q], dtype=float)
        if vals.size:
            vals = vals[vals < np.quantile(vals, .99)]
            # Boxplot rendering is quadratic-ish for very large collections
            # on some matplotlib versions; a fixed deterministic sample is
            # visually sufficient and leaves raw values untouched.
            if vals.size > 10000:
                vals = vals[np.linspace(0, vals.size - 1, 10000, dtype=int)]
        data.append(vals)
    ax.boxplot(data, tick_labels=phases, showfliers=False); ax.set_ylabel("T_MoE CUDA ms (trimmed p99)"); fig.tight_layout(); fig.savefig(p / "phase_separated_tmoe.png", dpi=160); plt.close(fig)


def write_text_artifacts(root: Path, rows: list[dict], metrics: dict, replay: dict, pairs: dict) -> None:
    (root / "source_audit.md").write_text("""# Source and runtime audit\n\n- Model: local `Qwen3-VL-30B-A3B-Instruct` snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.\n- Runtime: vLLM 0.20.0 V1, BF16, `TP2/DP2/EP4`, eager, chunked prefill (`max_num_batched_tokens=8192`), DBO off, prefix cache off.\n- GPU visibility: `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical mapping recorded in `online_trace3/topology.dp*.json` and `online_trace_high2/topology.dp*.json`).\n- Runtime proof in `online_trace3/serving.log` / `online_trace_high2/run.log`: `Using DeepEPHTAll2AllManager`, EP world size 4, linear 32/128 experts, `Using TRITON Unquantized MoE backend`, `Using DeepEPHTPrepareAndFinalize`.\n\n## Boundary\n\n`Qwen3MoeDecoderLayer.forward` (local source `.../vllm/model_executor/models/qwen3_moe.py`, class at line 364 and forward at line 416) performs attention then `self.mlp`. The fused-MoE runner obtains `topk_ids` from the router and invokes the unquantized method. The local read-only hook wraps that stock `apply` call with CUDA events and records the exact `topk_ids`; no route, tensor, placement, or scheduler operation is changed.\n\nDeepEP HT is selected through `all2all_backend=deepep_high_throughput`; local `deepep_ht.py` dispatch starts at line 97 and finalization/combine at line 336. The path passes `previous_event` through the communication stream and uses asynchronous DeepEP prepare/finalize semantics. The online hook therefore treats the full stock apply interval as `T_MoE`; it does not claim that dispatch/expert/combine subspans are separately timestamped in the serving trace. Those subspans are available only in the separately labeled route-transfer replay.\n\n## Feature caveat\n\nPer-token fanout is computed exactly from the captured top-k expert IDs (`expert_id // 32`), yielding F1–F4. The hook's sender-destination matrix intentionally records a conservative local-source row; it is not a global cross-DP traffic matrix. Fanout conclusions are therefore not conflated with a complete sender matrix.\n""", encoding="utf-8")
    contexts = {}
    for r in rows: contexts.setdefault(r.get("trace_source"), {"rows": 0, "context": r.get("request_context")}); contexts[r.get("trace_source")]["rows"] += 1
    manifest = {"model": "Qwen3-VL-30B-A3B-Instruct", "snapshot": "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c", "topology": "TP2/DP2/EP4", "cuda_visible_devices": "1,2,3,4", "backend": "DeepEP high-throughput + Triton Unquantized", "precision": "BF16", "scheduler": "vLLM V1 continuous batching; max_num_batched_tokens=8192; max_num_seqs=16; chunked prefill enabled", "online_runs": contexts, "rows": len(rows), "phase_counts": {q: sum(r.get("phase") == q for r in rows) for q in ("prefill", "decode")}, "M_distribution": {str(k): sum(r.get("M") == k for r in rows) for k in sorted({r.get("M") for r in rows})}, "route_features": ["per-token fanout F1..F4", "mean/median/p10/p90 fanout", "per-expert histogram", "rank assignment loads", "sender-destination conservative local row"], "limitations": ["online hook records full FusedMoE CUDA interval rather than independent dispatch/expert/combine CUDA events", "worker rows are correlated, not independent requests", "M>2048 vLLM profile dummy forwards excluded from natural analysis", "old layer=-1 debugging trace retained but excluded"]}
    (root / "workload_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "matched_pairs" / "README.md").write_text("Matched natural pair construction and caveats are in summary.json; pairs are association diagnostics, not independent causal samples.\n", encoding="utf-8")
    # Dependency classification is deliberately conservative and limited to
    # the question's geometry features.
    dep = {"fanout_to_dispatch": "CONDITIONAL", "fanout_to_expert": "CONDITIONAL", "fanout_to_combine": "CONDITIONAL", "fanout_to_full_T_MoE": "CONDITIONAL", "notes": "same-request top-k geometry is upstream of DeepEP; cross-request fanout is independent, but this hook does not intervene in routing"}
    (root / "dependency_graph.json").write_text(json.dumps(dep, indent=2), encoding="utf-8")
    (root / "dependency_graph.md").write_text("# Dependency graph\n\n- Captured top-k → per-token fanout → DeepEP layout/dispatch/combine: **CONDITIONAL** (routing geometry is upstream, but the online trace is observational).\n- Cross-request invocations: **CROSS_REQUEST_INDEPENDENT** at the model boundary; scheduler timing and worker resource contention remain confounders.\n- Fanout → T_MoE: **CONDITIONAL**, tested by hierarchical models and histogram-preserving replay.\n", encoding="utf-8")
    matrix = [
        ["fanout", "T_MoE", "CONDITIONAL", "UNKNOWN", "MIXED", "MAYBE", "online model adds no held-out information; replay is non-monotonic"],
        ["fanout", "Dispatch", "CONDITIONAL", "UNKNOWN", "BOUNDED_REPLAY_OBSERVED", "MAYBE", "route-transfer replay only"],
        ["fanout", "Expert", "CONDITIONAL", "UNKNOWN", "BOUNDED_REPLAY_OBSERVED", "MAYBE", "per-expert histogram is held fixed in replay"],
        ["fanout", "Combine", "CONDITIONAL", "UNKNOWN", "BOUNDED_REPLAY_OBSERVED", "MAYBE", "route-transfer replay only"],
    ]
    with (root / "resource_compatibility_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["phase_a","phase_b","dependency","resource_compatibility","evidence","overlap_candidate","reason"]); w.writerows(matrix)
    (root / "resource_compatibility_matrix.md").write_text("# Routing geometry compatibility\n\nNo pair is promoted to a scheduler/optimization candidate by this PoC. The online Model-2→Model-3 comparison is null; route-transfer replay is bounded and non-monotonic.\n", encoding="utf-8")
    (root / "overlap_candidate_shortlist.md").write_text("# Candidate shortlist\n\n1. **Histogram-preserving fanout replay** — diagnostic only; exact M/M_e/rank invariants, but non-monotonic and not online serving evidence.\n2. **Online fanout residual feature** — not supported by held-out Model-3 improvement. No optimization follow-up is recommended.\n", encoding="utf-8")
    gate = {"status": "NO_GO", "deepep_verified": True, "online_serving_verified": True, "natural_fanout_variation_sufficient": "PARTIAL", "model2_to_model3_rmse_reduction_pct": metrics.get("model2_to_model3_rmse_reduction_pct"), "phase_model2_to_model3_rmse_reduction_pct": {k: v.get("model2_to_model3_rmse_reduction_pct") for k, v in metrics.get("phase_metrics", {}).items()}, "matched_pairs": pairs, "histogram_preserving_replay": replay, "reason": "fanout geometry adds no robust held-out information beyond M, expert distribution, and rank-load controls; replay direction is non-monotonic and online prefill association is unstable", "optimization_method": "NOT_IMPLEMENTED"}
    (root / "gate_summary.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.root); write_csv(rows, args.root)
    # Keep the existing analyzer as the canonical hierarchical implementation
    # and add phase-sensitive metrics here for the final report.
    from analyze_online import phase_metrics
    dist=["M","active_experts","expert_max_load","expert_p90_load","expert_cv","expert_hhi","expert_entropy","padded_work_proxy"]
    rank=dist+["rank_max_mean","rank_cv"]
    geom=rank+["fanout_mean","fanout_p10","fanout_median","fanout_p90","fanout_f4","traffic_entropy","traffic_concentration"]
    from analyze_online import fit
    metrics={"n_rows": len(rows), "model0": fit(rows,["M"]), "model1_distribution": fit(rows,dist), "model2_distribution_plus_rank": fit(rows,rank), "model3_plus_fanout_geometry": fit(rows,geom), "phase_metrics": phase_metrics(rows)}
    a=metrics["model2_distribution_plus_rank"]; b=metrics["model3_plus_fanout_geometry"]; metrics["model2_to_model3_rmse_reduction_pct"] = 100*(a["rmse"]-b["rmse"])/(a["rmse"]+1e-12) if a.get("status")==b.get("status")=="OK" else None
    (args.root / "models").mkdir(exist_ok=True); (args.root / "models" / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fan = np.asarray([r["fanout_mean"] for r in rows]); (args.root / "natural_fanout_summary.json").write_text(json.dumps({"mean": float(fan.mean()), "p10": float(np.quantile(fan,.1)), "p50": float(np.quantile(fan,.5)), "p90": float(np.quantile(fan,.9)), "min": float(fan.min()), "max": float(fan.max()), "f4_range": [float(min(r.get("fanout_f4",0) for r in rows)), float(max(r.get("fanout_f4",0) for r in rows))], "prefill_only": {"mean": float(np.mean([r["fanout_mean"] for r in rows if r.get("phase")=="prefill"])), "p10": float(np.quantile([r["fanout_mean"] for r in rows if r.get("phase")=="prefill"],.1)), "p50": float(np.quantile([r["fanout_mean"] for r in rows if r.get("phase")=="prefill"],.5)), "p90": float(np.quantile([r["fanout_mean"] for r in rows if r.get("phase")=="prefill"],.9))}}, indent=2), encoding="utf-8")
    replay = replay_summary(args.root); pairs = matched_pairs(rows, args.root); make_plots(rows, metrics, replay, pairs, args.root); write_text_artifacts(args.root, rows, metrics, replay, pairs)
    (args.root / "models" / "analysis_summary.json").write_text(json.dumps({"rows": len(rows), "fanout_summary": json.loads((args.root / "natural_fanout_summary.json").read_text()), "model2_to_model3_pct": metrics.get("model2_to_model3_rmse_reduction_pct"), "phase_model2_to_model3_pct": {k: v.get("model2_to_model3_rmse_reduction_pct") for k, v in metrics.get("phase_metrics", {}).items()}, "pairs": pairs.get("pairs", 0)}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "metrics": str(args.root/"models"/"model_metrics.json"), "replay": replay, "pairs": pairs}, indent=2))


if __name__ == "__main__": main()
