#!/usr/bin/env python3
"""Aggregate the three cheap-oracle pivots without double-counting EP ranks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


def median(xs):
    return float(stats.median(xs)) if xs else float("nan")


def percentile(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(ys[lo])
    return float(ys[lo] * (hi - pos) + ys[hi] * (pos - lo))


def cv(xs):
    if not xs or median(xs) == 0:
        return float("nan")
    return float(stats.pstdev(xs) / median(xs))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def critical_rows(rank_docs, key_fields, value_fields):
    """Join logical observations and select max same-device duration across ranks."""
    grouped = defaultdict(lambda: defaultdict(list))
    for doc in rank_docs:
        seen = set()
        for row in doc["rows"]:
            key = tuple(row[k] for k in key_fields)
            # timing files are duplicated by the all-rank flush RPC; collapse exact copies.
            signature = key + tuple(row.get(k) for k in value_fields)
            if signature in seen:
                continue
            seen.add(signature)
            for field in value_fields:
                grouped[key][field].append(float(row[field]))
    out = []
    for key, vals in grouped.items():
        row = dict(zip(key_fields, key))
        row.update({field: max(v) for field, v in vals.items()})
        out.append(row)
    return out


def summarize_pairwise(run_root, modality):
    docs = [read_json(p) for p in sorted((run_root / "raw").glob("pairwise.rank*.json"))]
    rows = critical_rows(
        docs,
        ["phase", "iteration"],
        ["attention_ms", "stage_ms", "concurrent_wall_ms",
         "attention_concurrent_ms", "stage_concurrent_ms"],
    )
    for r in rows:
        r["modality"] = modality
        r["eta"] = ((r["attention_ms"] + r["stage_ms"] - r["concurrent_wall_ms"])
                    / min(r["attention_ms"], r["stage_ms"]))
        r["wall_saving_pct"] = 100 * (1 - r["concurrent_wall_ms"] /
                                       (r["attention_ms"] + r["stage_ms"]))
    summary = []
    for phase in sorted({r["phase"] for r in rows}):
        rr = [r for r in rows if r["phase"] == phase]
        etas = [r["eta"] for r in rr]
        savings = [r["wall_saving_pct"] for r in rr]
        summary.append({
            "modality": modality,
            "phase": phase,
            "repetitions": len(rr),
            "attention_ms": median([r["attention_ms"] for r in rr]),
            "stage_ms": median([r["stage_ms"] for r in rr]),
            "concurrent_wall_ms": median([r["concurrent_wall_ms"] for r in rr]),
            "eta_median": median(etas),
            "eta_p10": percentile(etas, .1),
            "eta_p90": percentile(etas, .9),
            "eta_cv_abs": cv([abs(x) for x in etas]),
            "wall_saving_pct_median": median(savings),
        })
    correctness = [x for d in docs for x in d.get("attention_correctness", [])]
    correct = {
        "n": len(correctness),
        "passed": all(x["passed"] for x in correctness),
        "min_cosine": min(x["cosine_similarity"] for x in correctness),
        "max_abs": max(x["max_abs_error"] for x in correctness),
    }
    return rows, summary, correct


def summarize_timing(run_root, modality):
    docs = [read_json(p) for p in sorted((run_root / "raw").glob("timing.rank*.json"))]
    rows = critical_rows(docs, ["wave", "iteration", "layer", "stage", "tokens"], ["duration_ms"])
    for r in rows:
        r["modality"] = modality
    out = []
    for (layer, stage), rr in _groups(rows, ["layer", "stage"]):
        out.append({
            "modality": modality,
            "layer": layer,
            "stage": stage,
            "repetitions": len(rr),
            "duration_ms": median([r["duration_ms"] for r in rr]),
            "p10_ms": percentile([r["duration_ms"] for r in rr], .1),
            "p90_ms": percentile([r["duration_ms"] for r in rr], .9),
            "tokens": rr[0]["tokens"],
        })
    return rows, out


def _groups(rows, fields):
    d = defaultdict(list)
    for row in rows:
        d[tuple(row[f] for f in fields)].append(row)
    return sorted(d.items())


def summarize_policy(run_root, modality, kind):
    docs = [read_json(p) for p in sorted((run_root / "raw").glob(f"policy.{kind}.rank*.json"))]
    rows = critical_rows(docs, ["kind", "aggregation", "sms", "iteration", "tokens_per_rank"],
                         ["total_ms", "dispatch_ms", "expert_ms", "combine_ms"])
    summary = []
    for (agg, sms), rr in _groups(rows, ["aggregation", "sms"]):
        summary.append({
            "modality": modality,
            "kind": kind,
            "aggregation": agg,
            "sms": sms,
            "repetitions": len(rr),
            "tokens_per_rank": rr[0]["tokens_per_rank"],
            **{f: median([r[f] for r in rr])
               for f in ["total_ms", "dispatch_ms", "expert_ms", "combine_ms"]},
            "per_unit_ms": median([r["total_ms"] for r in rr]) / agg,
        })
    correctness = [x for d in docs for x in d.get("correctness", [])]
    return rows, summary, {
        "n": len(correctness),
        "passed": all(x["passed"] for x in correctness),
        "min_cosine": min((x["cosine_similarity"] for x in correctness), default=float("nan")),
        "max_abs": max((x["max_abs_error"] for x in correctness), default=float("nan")),
    }


def driver_summary(run_root, modality):
    docs = [read_json(p) for p in sorted(run_root.glob("driver.*.json"))]
    records = [x for d in docs for x in d["records"]]
    out = {"modality": modality, "drivers_ok": all(d["ok"] for d in docs)}
    for mode in ["clean", "instrumented"]:
        # A replicated request was issued to both DP ranks; one logical wave uses critical max.
        waves = defaultdict(list)
        for r in records:
            if r["mode"] == mode:
                waves[r["wave"]].append(r["ttft_ms"])
        vals = [max(v) for v in waves.values()]
        out[f"{mode}_ttft_ms"] = median(vals)
        out[f"{mode}_n"] = len(vals)
    out["observer_tax_pct"] = 100 * (out["instrumented_ttft_ms"] /
                                      out["clean_ttft_ms"] - 1)
    return out


def scheduling_oracle(stage_summary, pair_summary):
    """Two exact request chains (A_l,M_l), no token-axis split.

    We report a perfect resource oracle and a contention-corrected layer oracle.
    The latter uses measured whole-MoE eta for each attention modality.  This is an
    analytical bound, not observed request TTFT.
    """
    cost = defaultdict(dict)
    for r in stage_summary:
        cost[(r["modality"], int(r["layer"]))][r["stage"]] = r["duration_ms"]
    eta = {(r["modality"], r["phase"]): r["eta_median"] for r in pair_summary}
    tasks = {}
    for mod in ["vision", "text"]:
        tasks[mod] = []
        for layer in range(48):
            tasks[mod].append(("A", cost[(mod, layer)]["attention"]))
            tasks[mod].append(("M", cost[(mod, layer)]["moe"]))

    def dp(pair_mode):
        a, b = tasks["vision"], tasks["text"]
        n, m = len(a), len(b)
        table = [[float("inf")] * (m + 1) for _ in range(n + 1)]
        table[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                cur = table[i][j]
                if i < n:
                    table[i + 1][j] = min(table[i + 1][j], cur + a[i][1])
                if j < m:
                    table[i][j + 1] = min(table[i][j + 1], cur + b[j][1])
                if i < n and j < m and a[i][0] != b[j][0]:
                    x, y = a[i], b[j]
                    if pair_mode == "perfect":
                        pc = max(x[1], y[1])
                    else:
                        amod = "vision" if x[0] == "A" else "text"
                        e = eta[(amod, "moe")]
                        pc = x[1] + y[1] - e * min(x[1], y[1])
                        if pair_mode == "heuristic" and e < 0.2:
                            continue
                    table[i + 1][j + 1] = min(table[i + 1][j + 1], cur + pc)
        return table[n][m]

    serial = sum(x[1] for v in tasks.values() for x in v)
    perfect = dp("perfect")
    measured = dp("measured")
    heuristic = dp("heuristic")
    return [
        {"policy": "FCFS/static serial", "makespan_ms": serial, "gain_pct": 0.0,
         "evidence": "measured component costs"},
        {"policy": "perfect phase-aware oracle", "makespan_ms": perfect,
         "gain_pct": 100 * (1 - perfect / serial), "evidence": "zero-contention analytical"},
        {"policy": "contention-corrected layer oracle", "makespan_ms": measured,
         "gain_pct": 100 * (1 - measured / serial), "evidence": "measured eta analytical"},
        {"policy": "simple eta>=0.2 heuristic", "makespan_ms": heuristic,
         "gain_pct": 100 * (1 - heuristic / serial), "evidence": "measured eta analytical"},
    ]


def policy_oracle(policy_summary, stage_summary, driver):
    # Per-unit cost is a throughput/BCT metric for pools of whole request units.
    # Aggregation is allowed to fill the available pool under one static cap; the
    # only genuinely fixed runtime knob here is num_sms.  Treating g=4 versus
    # g=64 as different phase policies would manufacture an oracle from demand.
    kinds = ["vision", "text", "decode"]
    vals = defaultdict(lambda: defaultdict(list))
    for r in policy_summary:
        k = r["kind"] if r["kind"] == "decode" else r["modality"]
        vals[k][(int(r["aggregation"]), int(r["sms"]))].append(float(r["per_unit_ms"]))
    tab = {k: {p: median(v) for p, v in vv.items()} for k, vv in vals.items()}
    sms_values = sorted(set.intersection(*[
        {sms for _, sms in tab[k]} for k in kinds]))

    # Throughput/BCT: each phase may fill all currently available whole requests.
    best_at_sms = {k: {s: min(v for (a, ss), v in tab[k].items() if ss == s)
                       for s in sms_values} for k in kinds}
    static_throughput = {s: sum(best_at_sms[k][s] for k in kinds) for s in sms_values}
    best_static_sms, best_static_cost = min(static_throughput.items(), key=lambda x: x[1])
    phase_throughput = {k: min(best_at_sms[k].items(), key=lambda x: x[1]) for k in kinds}
    oracle_throughput = sum(v[1] for v in phase_throughput.values())
    throughput_gain = 100 * (1 - oracle_throughput / best_static_cost)

    # Latency: one whole request unit, aggregation=1 everywhere.
    latency = {k: {s: tab[k][(1, s)] for s in sms_values} for k in kinds}
    static_latency = {s: sum(latency[k][s] for k in kinds) for s in sms_values}
    best_latency_sms, best_latency_cost = min(static_latency.items(), key=lambda x: x[1])
    phase_latency = {k: min(latency[k].items(), key=lambda x: x[1]) for k in kinds}
    oracle_latency = sum(v[1] for v in phase_latency.values())
    latency_gain = 100 * (1 - oracle_latency / best_latency_cost)

    # Conservative Amdahl projection: only measured MoE share of prefill TTFT can benefit.
    moe_totals = defaultdict(float)
    for r in stage_summary:
        if r["stage"] == "moe":
            moe_totals[r["modality"]] += r["duration_ms"]
    clean = {r["modality"]: r["clean_ttft_ms"] for r in driver}
    shares = {m: min(1.0, moe_totals[m] / clean[m]) for m in ["vision", "text"]}
    # Apply measured per-phase reduction relative to best static; decode share is unavailable.
    reductions = {k: max(0.0, 1 - phase_latency[k][1] / latency[k][best_latency_sms])
                  for k in kinds}
    e2e = 100 * stats.mean([shares[m] * reductions[m] for m in ["vision", "text"]])
    return {
        "best_static_throughput": {"sms": best_static_sms,
                                   "sum_per_unit_ms": best_static_cost},
        "phase_throughput_choices": {k: {"sms": v[0], "per_unit_ms": v[1]}
                                     for k, v in phase_throughput.items()},
        "moe_throughput_oracle_gain_pct": throughput_gain,
        "best_static_latency": {"sms": best_latency_sms,
                                "sum_single_unit_ms": best_latency_cost},
        "phase_latency_choices": {k: {"sms": v[0], "single_unit_ms": v[1]}
                                  for k, v in phase_latency.items()},
        "moe_latency_oracle_gain_pct": latency_gain,
        "prefill_moe_share": shares,
        "amdahl_prefill_ttft_gain_pct": e2e,
        "caveat": "aggregation uses whole request/layer units; projected TTFT is Amdahl, not observed",
    }


def logits_correctness(root):
    import numpy as np
    result = {}
    for modality in ["text", "vision"]:
        p = root / f"run_{modality}" / "raw" / "logits.rank0.npz"
        z = np.load(p)
        keys = list(z.files)
        schedule = read_json(root / f"run_{modality}" / "schedule.json")
        wave_mode = {f"wave_{r['wave']}": r["mode"] for r in schedule}
        base_key = next(k for k in keys if wave_mode.get(k) == "clean")
        base = z[base_key].astype("float64").ravel()
        rr = {}
        for k in keys:
            x = z[k].astype("float64").ravel()
            cos = float(x.dot(base) / (np.linalg.norm(x) * np.linalg.norm(base)))
            rel = float(np.linalg.norm(x - base) / np.linalg.norm(base))
            rr[f"{k}:{wave_mode.get(k, 'unknown')}"] = {
                     "cosine": cos, "rel_l2": rel, "argmax": int(x.argmax()),
                     "argmax_match": bool(x.argmax() == base.argmax())}
        result[modality] = rr
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    all_pair, pair_summary, pair_correct = [], [], {}
    all_stage, stage_summary, driver = [], [], []
    all_policy, policy_summary, policy_correct = [], [], {}
    for modality in ["text", "vision"]:
        rr = root / f"run_{modality}"
        rows, summ, corr = summarize_pairwise(rr, modality)
        all_pair += rows; pair_summary += summ; pair_correct[modality] = corr
        rows, summ = summarize_timing(rr, modality)
        all_stage += rows; stage_summary += summ
        driver.append(driver_summary(rr, modality))
        for kind in [modality, "decode"]:
            rows, summ, corr = summarize_policy(rr, modality, kind)
            # Decode is identical in structure but retained per originating workload/run.
            if kind == "decode":
                for x in summ: x["kind"] = "decode"
            all_policy += rows; policy_summary += summ
            policy_correct[f"{modality}:{kind}"] = corr
    # Keep one decode summary per modality to expose restart variation; policy oracle pools them.
    schedules = scheduling_oracle(stage_summary, pair_summary)
    policy = policy_oracle(policy_summary, stage_summary, driver)
    correctness = {"attention": pair_correct, "policy": policy_correct,
                   "full_logits": logits_correctness(root)}
    write_csv(root / "pairwise_samples.csv", all_pair)
    write_csv(root / "pairwise_summary.csv", pair_summary)
    write_csv(root / "stage_summary.csv", stage_summary)
    write_csv(root / "policy_summary.csv", policy_summary)
    write_csv(root / "scheduling_oracle.csv", schedules)
    write_csv(root / "driver_summary.csv", driver)
    (root / "correctness.json").write_text(json.dumps(correctness, indent=2))
    (root / "policy_oracle.json").write_text(json.dumps(policy, indent=2))
    print(json.dumps({"driver": driver, "pairwise": pair_summary,
                      "scheduling": schedules, "policy_oracle": policy,
                      "correctness": correctness}, indent=2))


if __name__ == "__main__":
    main()
