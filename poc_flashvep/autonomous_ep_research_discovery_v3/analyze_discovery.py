#!/usr/bin/env python3
"""Compact, auditable analysis for the autonomous EP discovery sprint.

The online hook emits one row per TP/EP worker.  This utility never adds rank
rows: it collapses a logical invocation by taking the maximum same-DP worker
span, and treats stage records in the same way.  It is intentionally a
descriptive atlas and residual miner, not an optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    a = sorted(float(x) for x in xs)
    pos = (len(a) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (pos - lo)


def load_rows(paths: list[Path], measured_only: bool = False) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(r.get("layer", -1)) < 0 or int(r.get("M", 0) or 0) > 2048:
                continue
            # The hook sets active_wave only after warmup.  Keeping this
            # filter explicit prevents model-load/first-use rows from being
            # mistaken for measured serving observations.
            if measured_only and r.get("active_wave") is None:
                continue
            # Older versions of the local observer omitted rank_cv even
            # though they emitted the exact rank-load vector.  Reconstruct
            # the statistic from that vector instead of silently treating it
            # as zero; this keeps the rank-load control auditable across
            # observer revisions.
            loads = r.get("rank_loads")
            if isinstance(loads, list) and loads:
                try:
                    vals = [float(x) for x in loads]
                    mean = sum(vals) / len(vals)
                    if mean > 0:
                        r["rank_cv"] = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5 / mean
                        r["rank_max_mean"] = max(vals) / mean
                except (TypeError, ValueError):
                    pass
            r["source"] = str(path.parent)
            rows.append(r)
    return rows


def collapse(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        # local_invocation_id is process-local; DP rank keeps the two serving
        # timelines separate.  TP/EP rows are correlated observations.
        groups[(int(r.get("dp_rank", -1)), int(r.get("local_invocation_id", -1)),
                int(r.get("layer", -1)), str(r.get("phase", "unknown")))].append(r)
    out: list[dict] = []
    for key, rs in groups.items():
        z = {
            "dp_rank": key[0], "local_invocation_id": key[1], "layer": key[2],
            "phase": key[3], "M": int(rs[0].get("M", 0)),
            "cuda_ms": max(float(x.get("cuda_ms", 0.0) or 0.0) for x in rs),
            "wall_ms": max(float(x.get("wall_ms", 0.0) or 0.0) for x in rs),
            "active_experts": float(rs[0].get("active_experts", 0) or 0),
            "rank_max_mean": float(rs[0].get("rank_max_mean", 0) or 0),
            "rank_cv": float(rs[0].get("rank_cv", 0) or 0),
            "expert_cv": float(rs[0].get("expert_cv", 0) or 0),
            "fanout_mean": float(rs[0].get("fanout_mean", 0) or 0),
            "fanout_f4": float(rs[0].get("fanout_f4", 0) or 0),
            "timestamp_ns": min(int(x.get("timestamp_ns", 0) or 0) for x in rs),
            "source": rs[0].get("source", ""),
        }
        for stage in ("deepep_layout", "deepep_dispatch", "expert", "deepep_combine", "deepep_event_wait"):
            values = []
            waits = []
            for r in rs:
                for s in r.get("stage_records", []) or []:
                    if s.get("stage") == stage:
                        values.append(float(s.get("cuda_ms", 0.0) or 0.0))
                        if s.get("wait_max_ms") is not None:
                            waits.append(float(s.get("wait_max_ms", 0.0) or 0.0))
            z[stage + "_ms"] = max(values) if values else 0.0
            if waits:
                z[stage + "_wait_max_ms"] = max(waits)
        out.append(z)
    return sorted(out, key=lambda x: (x["timestamp_ns"], x["dp_rank"], x["layer"]))


def linear_metrics(rows: list[dict], features: list[str]) -> dict:
    """Time-block linear residual metrics using only stdlib math."""
    good = [r for r in rows if all(math.isfinite(float(r.get(f, 0))) for f in features + ["cuda_ms"])]
    if len(good) < len(features) + 8:
        return {"status": "INSUFFICIENT", "n": len(good), "features": features}
    good.sort(key=lambda r: int(r.get("timestamp_ns", 0)))
    cut = max(len(features) + 2, min(len(good) - 2, int(.7 * len(good))))
    train, test = good[:cut], good[cut:]
    # Normal equations with a small ridge term avoid a dependency on numpy in
    # deployment environments.  Feature scales are normalized by train RMS.
    scales = []
    for f in features:
        rms = math.sqrt(sum(float(r.get(f, 0)) ** 2 for r in train) / len(train))
        scales.append(rms if rms > 1e-12 else 1.0)
    X = [[1.0] + [float(r.get(f, 0)) / s for f, s in zip(features, scales)] for r in train]
    y = [float(r["cuda_ms"]) for r in train]
    n = len(features) + 1
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for row, yi in zip(X, y):
        for i in range(n):
            b[i] += row[i] * yi
            for j in range(n):
                A[i][j] += row[i] * row[j]
    for i in range(n):
        A[i][i] += 1e-8
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(A[k][i]))
        if abs(A[pivot][i]) < 1e-12:
            return {"status": "SINGULAR", "n": len(good), "features": features}
        A[i], A[pivot] = A[pivot], A[i]; b[i], b[pivot] = b[pivot], b[i]
        d = A[i][i]
        for j in range(i, n): A[i][j] /= d
        b[i] /= d
        for k in range(n):
            if k == i: continue
            d = A[k][i]
            for j in range(i, n): A[k][j] -= d * A[i][j]
            b[k] -= d * b[i]
    pred = [b[0] + sum(b[i + 1] * float(r.get(f, 0)) / scales[i] for i, f in enumerate(features)) for r in test]
    yt = [float(r["cuda_ms"]) for r in test]
    rmse = math.sqrt(sum((a - c) ** 2 for a, c in zip(pred, yt)) / len(yt))
    mae = sum(abs(a - c) for a, c in zip(pred, yt)) / len(yt)
    mean = statistics.mean(yt)
    r2 = 1.0 - sum((a - c) ** 2 for a, c in zip(pred, yt)) / (sum((c - mean) ** 2 for c in yt) + 1e-12)
    return {"status": "OK", "n": len(good), "train_n": len(train), "test_n": len(test),
            "rmse": rmse, "mae": mae, "r2": r2,
            "p90_abs_error": quantile([abs(a - c) for a, c in zip(pred, yt)], .9),
            "features": features}


def stage_summary(rows: list[dict]) -> dict:
    result = {}
    for phase in sorted({r["phase"] for r in rows}):
        subset = [r for r in rows if r["phase"] == phase]
        total = sum(r["cuda_ms"] for r in subset) or 1.0
        z = {"n": len(subset), "M_values": sorted({r["M"] for r in subset}),
             "cuda_p50_ms": quantile([r["cuda_ms"] for r in subset], .5),
             "cuda_p90_ms": quantile([r["cuda_ms"] for r in subset], .9),
             "cuda_p99_ms": quantile([r["cuda_ms"] for r in subset], .99),
             "cuda_max_ms": max((r["cuda_ms"] for r in subset), default=None)}
        for stage in ("deepep_layout", "deepep_dispatch", "expert", "deepep_combine", "deepep_event_wait"):
            vals = [r[stage + "_ms"] for r in subset]
            z[stage] = {"p50_ms": quantile(vals, .5), "p90_ms": quantile(vals, .9),
                        "p99_ms": quantile(vals, .99), "max_ms": max(vals, default=None),
                        "sum_ms": sum(vals), "sum_over_tmoe": sum(vals) / total}
        result[phase] = z
    return result


def e2e_summary(root: Path) -> dict:
    values = []
    for fn in sorted(root.glob("waves.dp*.json")):
        try: waves = json.loads(fn.read_text(encoding="utf-8"))
        except Exception: continue
        for w in waves:
            for m in w.get("request_metrics", []) or []:
                if m.get("first_token_latency") is None or m.get("last_token_ts") is None or m.get("first_token_ts") is None:
                    continue
                e2e = float(m["first_token_latency"]) * 1000 + (float(m["last_token_ts"]) - float(m["first_token_ts"])) * 1000
                values.append(e2e)
    return {"n": len(values), "p50_ms": quantile(values, .5), "p90_ms": quantile(values, .9),
            "p99_ms": quantile(values, .99), "mean_ms": statistics.mean(values) if values else None,
            "sum_ms": sum(values)}


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["dp_rank", "local_invocation_id", "layer", "phase", "M", "cuda_ms", "wall_ms",
              "active_experts", "rank_max_mean", "rank_cv", "expert_cv", "fanout_mean", "fanout_f4",
              "deepep_layout_ms", "deepep_dispatch_ms", "expert_ms", "deepep_combine_ms", "deepep_event_wait_ms"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n"); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, 0) for k in fields})


def analyze(root: Path, out: Path, measured_only: bool = False) -> dict:
    paths = sorted(root.glob("**/invocations.jsonl"))
    raw = load_rows(paths, measured_only=measured_only); logical = collapse(raw)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(logical, out / "logical_invocations.csv")
    features0 = ["M"]
    features1 = ["M", "active_experts", "expert_cv", "rank_max_mean", "rank_cv"]
    features2 = features1 + ["fanout_mean", "fanout_f4"]
    metrics = {"rows_raw": len(raw), "logical_rows": len(logical),
               "measured_only": measured_only,
               "stage_summary": stage_summary(logical), "e2e": e2e_summary(root),
               "model0_M": linear_metrics(logical, features0),
               "model1_load": linear_metrics(logical, features1),
               "model2_load_plus_fanout": linear_metrics(logical, features2)}
    m1, m2 = metrics["model1_load"], metrics["model2_load_plus_fanout"]
    metrics["model1_to_model2_rmse_reduction_pct"] = (100 * (m1["rmse"] - m2["rmse"]) / m1["rmse"]
        if m1.get("status") == m2.get("status") == "OK" else None)
    (out / "atlas_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--measured-only", action="store_true",
                    help="exclude warmup/profile rows (active_wave is required)")
    args = ap.parse_args(); print(json.dumps(analyze(args.root, args.out, args.measured_only), indent=2))


if __name__ == "__main__": main()
