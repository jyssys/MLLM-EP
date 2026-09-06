#!/usr/bin/env python3
"""Read-only analysis for the wait-aware DeepEP tail PoC.

The parser intentionally treats native DeepEP event readiness as unavailable
when the installed EventHandle has no query method.  ``event_wait_cuda_ms`` is
reported as the closest observed same-device wait proxy, never as a cross-GPU
clock difference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ("deepep_layout", "deepep_dispatch", "expert", "deepep_combine")


def _stage(rec: dict, name: str) -> float:
    for item in rec.get("stage_records", []):
        if item.get("stage") == name and item.get("cuda_ms") is not None:
            try:
                return float(item["cuda_ms"])
            except (TypeError, ValueError):
                pass
    return float("nan")


def load_run(run: Path) -> pd.DataFrame:
    rows = []
    path = run / "invocations.jsonl"
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = {k: rec.get(k) for k in (
                "timestamp_ns", "local_invocation_id", "scheduler_iteration_id",
                "route_id", "layer", "dp_rank", "ep_rank", "ep_size", "phase",
                "M", "top_k", "total_assignments", "active_experts", "rank_max_mean",
                "expert_max_mean", "expert_cv", "expert_hhi", "expert_entropy",
                "fanout_mean", "fanout_f4", "wall_ms", "cuda_ms", "request_context",
                "previous_event_present", "previous_event_count", "previous_event_ready",
                "event_wait_count", "event_wait_cuda_ms", "event_wait_max_ms",
                "comm_stream_id", "comm_drain_wall_ms", "intervention_policy",
                "intervention_requested", "intervention_applied", "intervention_kind",
                "oracle_hit", "simple_hit", "prev_dispatch_ms")}
            for stage in STAGES:
                row[stage] = _stage(rec, stage)
            row["run"] = run.name
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric = [c for c in df.columns if c not in {"run", "phase", "route_id", "request_context", "previous_event_ready", "intervention_policy", "intervention_kind"}]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def quant(values: pd.Series | np.ndarray) -> dict:
    a = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(a):
        return {"n": 0}
    mean = float(a.mean())
    return {
        "n": int(len(a)), "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)), "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)), "p99_9_ms": float(np.percentile(a, 99.9)),
        "max_ms": float(a.max()), "mean_ms": mean,
        "cv_pct": float(a.std() / mean * 100.0) if mean else None,
    }


def run_throughput(run: Path) -> float | None:
    vals = []
    for path in run.glob("waves.dp*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for wave in data:
            wall = float(wave.get("wall_ms", 0.0) or 0.0)
            toks = sum(len(x) for x in wave.get("outputs", []))
            if wall > 0:
                vals.append((toks, wall))
    total_ms = sum(x[1] for x in vals)
    return float(sum(x[0] for x in vals) / (total_ms / 1000.0)) if total_ms else None


def summarize_run(df: pd.DataFrame, run: Path) -> dict:
    if df.empty:
        return {"run": run.name, "n": 0}
    decode = df[df.phase == "decode"]
    part = decode if len(decode) else df
    out = {"run": run.name, "n": int(len(df)), "decode_n": int(len(decode)),
           "prefill_n": int((df.phase == "prefill").sum()),
           "throughput_tokens_s": run_throughput(run),
           "previous_event_present_rate": float(df.previous_event_present.fillna(False).astype(bool).mean()),
           "event_wait": quant(df.event_wait_cuda_ms),
           "moe": quant(part.cuda_ms), "dispatch": quant(part.deepep_dispatch)}
    for threshold in (5, 10, 20, 50, 100):
        out[f"dispatch_gt_{threshold}_count"] = int((part.deepep_dispatch > threshold).sum())
        out[f"dispatch_gt_{threshold}_rate_pct"] = float((part.deepep_dispatch > threshold).mean() * 100.0)
    # The proxy threshold is intentionally explicit and sensitivity is emitted.
    for threshold in (0.05, 0.1, 0.5, 1.0):
        ready = part.event_wait_cuda_ms <= threshold
        not_ready = part.event_wait_cuda_ms > threshold
        tail = part.deepep_dispatch > 10.0
        out[f"wait_proxy_{threshold:g}_n_ready"] = int(ready.sum())
        out[f"wait_proxy_{threshold:g}_n_not_ready"] = int(not_ready.sum())
        out[f"p_tail10_ready_{threshold:g}_pct"] = float(tail[ready].mean() * 100.0) if ready.any() else None
        out[f"p_tail10_not_ready_{threshold:g}_pct"] = float(tail[not_ready].mean() * 100.0) if not_ready.any() else None
    out["stage_medians_decode"] = {s: float(part[s].median()) for s in STAGES}
    out["wait_dispatch_corr"] = float(part[["event_wait_cuda_ms", "deepep_dispatch"]].corr().iloc[0, 1])
    return out


def markdown_table(rows: list[dict], keys: list[str]) -> str:
    if not rows:
        return "(no rows)\n"
    text = "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n"
    for row in rows:
        text += "| " + " | ".join(str(row.get(k, "")) for k in keys) + " |\n"
    return text


def analyze(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "invocations.jsonl").exists())
    frames, summaries = [], []
    for run in runs:
        df = load_run(run)
        if df.empty:
            continue
        frames.append(df)
        summaries.append(summarize_run(df, run))
    if not frames:
        raise SystemExit(f"no invocation traces under {root}")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(root / "all_stage_rows.csv", index=False)

    flat = []
    for s in summaries:
        r = {k: v for k, v in s.items() if not isinstance(v, dict)}
        for prefix in ("moe", "dispatch", "event_wait"):
            for k, v in s.get(prefix, {}).items():
                r[f"{prefix}_{k}"] = v
        flat.append(r)
    pd.DataFrame(flat).to_csv(root / "policy_run_summary.csv", index=False)
    (root / "policy_run_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # Event-readiness analysis is explicitly a proxy analysis when native
    # EventHandle query is unavailable.
    decode = all_df[all_df.phase == "decode"].copy()
    if decode.empty:
        decode = all_df.copy()
    readiness_rows = []
    for cutoff in (0.05, 0.1, 0.5, 1.0):
        for label, mask in (("READY_PROXY", decode.event_wait_cuda_ms <= cutoff),
                            ("NOT_READY_PROXY", decode.event_wait_cuda_ms > cutoff)):
            p = decode.loc[mask, "deepep_dispatch"]
            readiness_rows.append({"cutoff_ms": cutoff, "class": label, "n": int(len(p)),
                                   "dispatch_p50_ms": float(p.quantile(.5)) if len(p) else None,
                                   "dispatch_p99_ms": float(p.quantile(.99)) if len(p) else None,
                                   "tail_gt10_rate_pct": float((p > 10).mean() * 100) if len(p) else None,
                                   "tail_gt20_rate_pct": float((p > 20).mean() * 100) if len(p) else None})
    pd.DataFrame(readiness_rows).to_csv(root / "event_readiness.csv", index=False)
    event_md = ["# Event readiness analysis\n",
                "The installed DeepEP EventHandle has no nonblocking query method. "
                "`previous_event_present` was therefore recorded directly and the table below uses "
                "same-device CUDA timing around the stock `EventOverlap.current_stream_wait()` as a "
                "closest wait proxy. It is not a cross-GPU absolute readiness clock.\n",
                f"- invocations: {len(decode):,}\n",
                f"- previous-event object present: {int(decode.previous_event_present.fillna(False).astype(bool).sum()):,} "
                f"({decode.previous_event_present.fillna(False).astype(bool).mean()*100:.3f}%)\n",
                f"- event-wait median/p99/max: {decode.event_wait_cuda_ms.quantile(.5):.4f} / "
                f"{decode.event_wait_cuda_ms.quantile(.99):.4f} / {decode.event_wait_cuda_ms.max():.4f} ms\n",
                "\n" + markdown_table(readiness_rows, ["cutoff_ms", "class", "n", "dispatch_p50_ms", "dispatch_p99_ms", "tail_gt10_rate_pct", "tail_gt20_rate_pct"])]
    (root / "EVENT_READINESS_ANALYSIS.md").write_text("\n".join(event_md), encoding="utf-8")

    policy_rows = []
    for s in summaries:
        policy_rows.append({"run": s["run"], "n_decode": s.get("decode_n"),
                            "MoE_p50_ms": s.get("moe", {}).get("p50_ms"),
                            "MoE_p99_ms": s.get("moe", {}).get("p99_ms"),
                            "MoE_p99.9_ms": s.get("moe", {}).get("p99_9_ms"),
                            "MoE_max_ms": s.get("moe", {}).get("max_ms"),
                            "Dispatch_p50_ms": s.get("dispatch", {}).get("p50_ms"),
                            "Dispatch_p99_ms": s.get("dispatch", {}).get("p99_ms"),
                            "Dispatch_max_ms": s.get("dispatch", {}).get("max_ms"),
                            ">10ms_rate_pct": s.get("dispatch_gt_10_rate_pct"),
                            ">20ms_rate_pct": s.get("dispatch_gt_20_rate_pct"),
                            ">100ms_rate_pct": s.get("dispatch_gt_100_rate_pct"),
                            "throughput_tokens_s": s.get("throughput_tokens_s"),
                            "wait_dispatch_corr": s.get("wait_dispatch_corr")})
    (root / "POLICY_COMPARISON.md").write_text(
        "# Policy comparison\n\n" + markdown_table(policy_rows, list(policy_rows[0].keys())) +
        "\nThe current stock and diagnostic runs are independent serving runs; their tails must be "
        "compared with run-level medians, not single invocations. P2/P3 columns are populated when "
        "those bounded runs are added.\n", encoding="utf-8")

    # Preserve a concise direct stage decomposition for root-cause attribution.
    stage_summary = []
    for run, group in all_df.groupby("run"):
        part = group[group.phase == "decode"]
        if part.empty: part = group
        row = {"run": run, "n": len(part), "MoE_p50_ms": part.cuda_ms.median(),
               "Dispatch_p50_ms": part.deepep_dispatch.median(), "Expert_p50_ms": part.expert.median(),
               "Combine_p50_ms": part.deepep_combine.median(), "wait_p99_ms": part.event_wait_cuda_ms.quantile(.99),
               "dispatch_max_ms": part.deepep_dispatch.max()}
        stage_summary.append(row)
    pd.DataFrame(stage_summary).to_csv(root / "stage_decomposition.csv", index=False)

    gate = {"event_handle_native_query": False,
            "previous_event_present_observed": bool(all_df.previous_event_present.fillna(False).astype(bool).any()),
            "stock_runs": [s["run"] for s in summaries if "stock" in s["run"]],
            "always_sync_runs": [s["run"] for s in summaries if "always_sync" in s["run"]],
            "comm_drain_runs": [s["run"] for s in summaries if "comm_drain" in s["run"]],
            "oracle_runs": [s["run"] for s in summaries if "oracle" in s["run"]],
            "online_simple_runs": [s["run"] for s in summaries if "simple" in s["run"]],
            "status": "MEASUREMENT_LIMITATION_NATIVE_READY_QUERY_UNAVAILABLE"}
    (root / "gate_summary.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return {"runs": len(summaries), "rows": len(all_df), "root": str(root)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    print(json.dumps(analyze(args.root), indent=2))


if __name__ == "__main__":
    main()
