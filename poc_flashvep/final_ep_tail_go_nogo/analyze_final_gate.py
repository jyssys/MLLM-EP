#!/usr/bin/env python3
"""Conservative final gate for history-dependent DeepEP tails.

This consumes the existing three-run economic trace and the new live run with
request-context instrumentation.  It deliberately reports direct request
numbers as an upper bound: a co-batched MoE invocation is associated with all
scheduled requests, then capped by each request's observed E2E latency.  No
rank rows or cross-device timestamps are summed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    a = sorted(float(x) for x in xs)
    if len(a) == 1:
        return a[0]
    pos = (len(a) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return a[lo]
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def trimmed_mean(xs: list[float], trim: float = .10) -> float:
    a = sorted(float(x) for x in xs)
    if not a:
        return 0.0
    k = int(len(a) * trim)
    b = a[k:len(a) - k] if len(a) > 2 * k else a
    return statistics.mean(b)


def load_contextual_rows(run: Path) -> list[dict]:
    raw = []
    with (run / "invocations.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("active_wave") is not None:
                raw.append(d)
    # Two TP rows represent one logical DP-local invocation.  Keep the slower
    # same-device span, never add rows from different devices.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in raw:
        key = (d.get("dp_rank"), d.get("local_invocation_id"), d.get("layer"),
               str(d.get("active_wave")))
        groups[key].append(d)
    return [max(v, key=lambda x: float(x.get("cuda_ms") or 0.0)) for v in groups.values()]


def stage_value(row: dict, name: str) -> float:
    return sum(float(s.get("cuda_ms") or 0.0) for s in row.get("stage_records", [])
               if s.get("stage") == name)


def load_request_metrics(run: Path) -> dict[tuple[int, int, str], dict]:
    result: dict[tuple[int, int, str], dict] = {}
    for fn in glob.glob(str(run / "waves.dp*.json")):
        dp = 0 if ".dp0." in fn else 1
        for wave in json.loads(Path(fn).read_text(encoding="utf-8")):
            w = int(wave["wave"])
            for m in wave.get("request_metrics", []):
                if m.get("first_token_latency") is None:
                    continue
                first = float(m["first_token_latency"]) * 1000.0
                last = m.get("last_token_ts")
                first_ts = m.get("first_token_ts")
                gen = (float(last) - float(first_ts)) * 1000.0 if last is not None and first_ts is not None else 0.0
                rid = str(m.get("vllm_request_id"))
                result[(dp, w, rid)] = {
                    "request_label": m.get("request_label"),
                    "ttft_ms": first,
                    "e2e_ms": first + gen,
                    "prompt_tokens": m.get("prompt_tokens"),
                    "output_tokens": m.get("output_tokens"),
                }
    return result


def direct_analysis(run: Path, out: Path) -> dict:
    rows = load_contextual_rows(run)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["dp_rank"], r["phase"], r["layer"], r["M"])].append(float(r["cuda_ms"]))
    bases = {
        "p25": {k: q(v, .25) for k, v in groups.items()},
        "p50": {k: q(v, .50) for k, v in groups.items()},
        "trimmed_mean": {k: trimmed_mean(v) for k, v in groups.items()},
    }
    for r in rows:
        key = (r["dp_rank"], r["phase"], r["layer"], r["M"])
        r["baseline_p25"] = bases["p25"][key]
        r["baseline_p50"] = bases["p50"][key]
        r["baseline_trimmed_mean"] = bases["trimmed_mean"][key]
        r["excess_p50"] = max(0.0, float(r["cuda_ms"]) - float(r["baseline_p50"]))
        r["excess_p25"] = max(0.0, float(r["cuda_ms"]) - float(r["baseline_p25"]))
        r["excess_trimmed_mean"] = max(0.0, float(r["cuda_ms"]) - float(r["baseline_trimmed_mean"]))

    total = sum(float(r["cuda_ms"]) for r in rows)
    summary: dict = {
        "run": str(run), "raw_contextual_rows": sum(1 for _ in open(run / "invocations.jsonl", encoding="utf-8")),
        "logical_contextual_invocations": len(rows),
        "total_moe_ms": total,
        "phase": {}, "thresholds": {}, "direct_request": {}, "wave": {},
    }
    for phase in ("prefill", "decode"):
        a = [r for r in rows if r["phase"] == phase]
        vals = [float(r["cuda_ms"]) for r in a]
        summary["phase"][phase] = {
            "n": len(a), "p50_ms": q(vals, .5), "p90_ms": q(vals, .9),
            "p99_ms": q(vals, .99), "p99_9_ms": q(vals, .999), "max_ms": max(vals) if vals else None,
            "excess_mass_p25_pct": 100 * sum(r["excess_p25"] for r in a) / sum(vals) if vals else None,
            "excess_mass_p50_pct": 100 * sum(r["excess_p50"] for r in a) / sum(vals) if vals else None,
            "excess_mass_trimmed_mean_pct": 100 * sum(r["excess_trimmed_mean"] for r in a) / sum(vals) if vals else None,
        }
    for threshold in (5, 10, 20, 50, 100):
        a = [r for r in rows if float(r["cuda_ms"]) > threshold]
        summary["thresholds"][f">{threshold}ms"] = {
            "count": len(a), "fraction_pct": 100 * len(a) / len(rows) if rows else None,
            "excess_p50_ms": sum(r["excess_p50"] for r in a),
            "excess_p50_mass_pct": 100 * sum(r["excess_p50"] for r in a) / total if total else None,
        }

    # Build request-level upper bounds.  Deep scheduler IDs have a stable
    # numeric request prefix matching RequestOutput.request_id in this driver.
    reqs = load_request_metrics(run)
    assigned_excess: dict[tuple[int, int, str], float] = defaultdict(float)
    assigned_max: dict[tuple[int, int, str], float] = defaultdict(float)
    assigned_20: Counter[tuple[int, int, str]] = Counter()
    for r in rows:
        dp, wave = int(r["dp_rank"]), int(r["active_wave"])
        for sid in r.get("scheduled_request_ids", []):
            rid = str(sid).split("-", 1)[0]
            key = (dp, wave, rid)
            assigned_excess[key] += r["excess_p50"]
            assigned_max[key] = max(assigned_max[key], float(r["cuda_ms"]))
            if float(r["cuda_ms"]) > 20:
                assigned_20[key] += 1
    request_rows = []
    for key, m in reqs.items():
        excess = assigned_excess.get(key, 0.0)
        cap = min(excess, float(m["e2e_ms"]))
        request_rows.append({
            "dp_rank": key[0], "wave": key[1], "request_id": key[2],
            "request_label": m["request_label"], "ttft_ms": m["ttft_ms"], "e2e_ms": m["e2e_ms"],
            "assigned_excess_upper_ms": excess, "capped_excess_upper_ms": cap,
            "removable_fraction_pct": 100 * cap / m["e2e_ms"] if m["e2e_ms"] else None,
            "max_assigned_moe_ms": assigned_max.get(key, 0.0),
            "events_over_20ms": assigned_20.get(key, 0),
        })
    fracs = [r["removable_fraction_pct"] / 100.0 for r in request_rows if r["removable_fraction_pct"] is not None]
    e2e = [r["e2e_ms"] for r in request_rows]
    cap = [r["capped_excess_upper_ms"] for r in request_rows]
    summary["direct_request"] = {
        "request_count": len(request_rows),
        "join_method": "scheduler request-id numeric prefix -> RequestOutput id; co-batched excess capped per request",
        "upper_bound_status": "UPPER_BOUND_NOT_EXACT_CRITICAL_PATH",
        "aggregate_capped_excess_ms": sum(cap), "aggregate_e2e_ms": sum(e2e),
        "aggregate_removable_share_pct": 100 * sum(cap) / sum(e2e) if e2e else None,
        "mean_share_pct": 100 * statistics.mean(fracs) if fracs else None,
        "median_share_pct": 100 * q(fracs, .5) if fracs else None,
        "p90_share_pct": 100 * q(fracs, .9) if fracs else None,
        "p99_share_pct": 100 * q(fracs, .99) if fracs else None,
        "max_share_pct": 100 * max(fracs) if fracs else None,
        "affected_requests_over_10ms": sum(r["max_assigned_moe_ms"] > 10 for r in request_rows),
        "affected_requests_over_20ms": sum(r["max_assigned_moe_ms"] > 20 for r in request_rows),
        "affected_requests_over_100ms": sum(r["max_assigned_moe_ms"] > 100 for r in request_rows),
    }
    # Sensitivity to the largest logical event(s).  This is deliberately
    # recomputed before request capping; it answers whether one outlier alone
    # creates the request-level headline.
    ordered = sorted(range(len(rows)), key=lambda i: float(rows[i]["cuda_ms"]), reverse=True)
    outlier_sensitivity = {}
    for label, excluded in (("with_max", set()), ("without_max", set(ordered[:1])),
                            ("without_top5", set(ordered[:5]))):
        a2: dict[tuple[int, int, str], float] = defaultdict(float)
        for i, r in enumerate(rows):
            if i in excluded:
                continue
            for sid in r.get("scheduled_request_ids", []):
                a2[(int(r["dp_rank"]), int(r["active_wave"]), str(sid).split("-", 1)[0])] += r["excess_p50"]
        caps = [min(a2.get(k, 0.0), float(m["e2e_ms"])) for k, m in reqs.items()]
        den = sum(float(m["e2e_ms"]) for m in reqs.values())
        outlier_sensitivity[label] = {"excluded_logical_events": len(excluded),
                                      "max_event_ms": float(rows[ordered[0]]["cuda_ms"]) if rows else None,
                                      "capped_excess_ms": sum(caps),
                                      "aggregate_e2e_ms": den,
                                      "aggregate_share_pct": 100 * sum(caps) / den if den else None}
    summary["direct_request"]["outlier_sensitivity"] = outlier_sensitivity
    for r in request_rows:
        pass
    # Per-wave critical upper bound avoids counting the same co-batched wait
    # twice across requests or DP ranks.
    by_wave_dp: dict[tuple[int, int], float] = defaultdict(float)
    for r in rows:
        by_wave_dp[(int(r["active_wave"]), int(r["dp_rank"]))] += r["excess_p50"]
    wave_rows = []
    for fn in glob.glob(str(run / "waves.dp*.json")):
        dp = 0 if ".dp0." in fn else 1
        for w in json.loads(Path(fn).read_text(encoding="utf-8")):
            wi = int(w["wave"])
            metrics = [m for m in w.get("request_metrics", []) if m.get("first_token_latency") is not None]
            e2e_ms = [float(m["first_token_latency"]) * 1000 +
                      (float(m["last_token_ts"]) - float(m["first_token_ts"])) * 1000
                      for m in metrics if m.get("last_token_ts") is not None and m.get("first_token_ts") is not None]
            wave_rows.append({"dp_rank": dp, "wave": wi, "max_request_e2e_ms": max(e2e_ms) if e2e_ms else None,
                              "critical_excess_ms": by_wave_dp.get((wi, dp), 0.0)})
    critical_by_wave = {}
    for w in sorted({x["wave"] for x in wave_rows}):
        x = [r for r in wave_rows if r["wave"] == w]
        e2e_ms = max((r["max_request_e2e_ms"] or 0) for r in x)
        excess_ms = max(r["critical_excess_ms"] for r in x)
        critical_by_wave[w] = {"e2e_ms": e2e_ms, "critical_excess_ms": excess_ms,
                               "capped_excess_ms": min(excess_ms, e2e_ms)}
    summary["wave"] = {"waves": critical_by_wave,
                        "aggregate_capped_share_pct": 100 * sum(x["capped_excess_ms"] for x in critical_by_wave.values()) /
                        sum(x["e2e_ms"] for x in critical_by_wave.values()) if critical_by_wave else None}

    out.mkdir(parents=True, exist_ok=True)
    with (out / "direct_invocation_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["dp_rank", "active_wave", "local_invocation_id", "layer", "phase", "M", "cuda_ms",
                  "baseline_p25", "baseline_p50", "baseline_trimmed_mean", "excess_p25", "excess_p50",
                  "excess_trimmed_mean", "scheduled_request_ids", "event_wait_cuda_ms"]
        wr = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n"); wr.writeheader()
        for r in rows:
            wr.writerow({k: json.dumps(r.get(k)) if k == "scheduled_request_ids" else r.get(k) for k in fields})
    with (out / "request_impact_upper_bound.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(request_rows[0]) if request_rows else ["request_id"]
        wr = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n"); wr.writeheader(); wr.writerows(request_rows)
    (out / "direct_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(direct_analysis(args.run, args.out), indent=2))


if __name__ == "__main__":
    main()
