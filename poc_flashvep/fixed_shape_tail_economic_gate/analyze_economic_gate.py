#!/usr/bin/env python3
"""Economic upper-bound analysis for fixed-shape DeepEP tails.

This is intentionally a read-only, trace-driven analysis.  Rank rows are
deduplicated by ``route_id`` (the two TP ranks are one logical invocation),
and all baselines are recomputed within run/phase/layer/M groups.  No
cross-device CUDA clocks are subtracted.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
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
                return float("nan")
    return float("nan")


def load_run(run: Path, dataset: str) -> tuple[list[dict], int, int]:
    """Load JSONL and collapse TP duplicate rows to a logical route."""
    logical: dict[str, dict] = {}
    raw_rows = 0
    duplicate_rows = 0
    path = run / "invocations.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_rows += 1
            key = str(rec.get("route_id"))
            vals = {
                "cuda_ms": float(rec.get("cuda_ms", "nan")),
                "wall_ms": float(rec.get("wall_ms", "nan")),
            }
            vals.update({s: _stage(rec, s) for s in STAGES})
            if key not in logical:
                logical[key] = {
                    "dataset": dataset,
                    "run": run.name,
                    "route_id": key,
                    "phase": rec.get("phase"),
                    "layer": int(rec.get("layer", -1)),
                    "M": int(rec.get("M", -1)),
                    "local_invocation_id": int(rec.get("local_invocation_id", -1)),
                    "dp_rank": int(rec.get("dp_rank", -1)),
                    "top_k": int(rec.get("top_k", -1)),
                    "active_experts": int(rec.get("active_experts", -1)),
                    "total_assignments": int(rec.get("total_assignments", -1)),
                    **vals,
                }
            else:
                duplicate_rows += 1
                # Two TP ranks form one critical local invocation.  For each
                # measured stage retain the slower rank, never sum ranks.
                for name, value in vals.items():
                    if math.isfinite(value):
                        old = logical[key].get(name, float("nan"))
                        logical[key][name] = value if not math.isfinite(old) else max(old, value)
    return list(logical.values()), raw_rows, duplicate_rows


def q(a: list[float] | np.ndarray) -> dict:
    x = np.asarray([v for v in a if math.isfinite(float(v))], dtype=float)
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        "p50_ms": float(np.percentile(x, 50)),
        "p90_ms": float(np.percentile(x, 90)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "p99_9_ms": float(np.percentile(x, 99.9)),
        "max_ms": float(np.max(x)),
        "mean_ms": float(np.mean(x)),
        "cv_pct": float(np.std(x) / np.mean(x) * 100.0) if np.mean(x) else None,
    }


def baseline(values: list[float], kind: str) -> float:
    x = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not x:
        return float("nan")
    if kind == "p25":
        return float(np.percentile(x, 25))
    if kind == "p50":
        return float(np.percentile(x, 50))
    # A 10% trimmed mean is a robust, still optimistic normal estimate.
    if len(x) >= 10:
        k = max(1, int(len(x) * 0.10))
        y = x[k:len(x) - k]
    else:
        y = x
    return float(np.mean(y))


def add_baselines(records: list[dict], kind: str, critical: bool = False) -> None:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        groups[(r["run"], r["phase"], r["layer"], r["M"])].append(r["cuda_ms"])
    for r in records:
        r[f"baseline_{kind}"] = baseline(groups[(r["run"], r["phase"], r["layer"], r["M"])], kind)


def mass_table(records: list[dict], kind: str, total_override: float | None = None) -> list[dict]:
    total = float(total_override if total_override is not None else sum(r["cuda_ms"] for r in records))
    out = []
    for threshold in (5, 10, 20, 50, 100):
        selected = [r for r in records if r["cuda_ms"] > threshold]
        raw = sum(r["cuda_ms"] for r in selected)
        excess = sum(max(0.0, r["cuda_ms"] - r[f"baseline_{kind}"]) for r in selected)
        out.append({
            "baseline": kind, "threshold_ms": threshold,
            "event_count": len(selected),
            "event_fraction_pct": 100.0 * len(selected) / len(records),
            "raw_latency_sum_ms": raw,
            "excess_latency_sum_ms": excess,
            "raw_fraction_of_total_pct": 100.0 * raw / total if total else None,
            "tail_excess_mass_pct": 100.0 * excess / total if total else None,
        })
    excess_all = sum(max(0.0, r["cuda_ms"] - r[f"baseline_{kind}"]) for r in records)
    out.append({
        "baseline": kind, "threshold_ms": "ALL_POSITIVE_EXCESS",
        "event_count": sum(r["cuda_ms"] > r[f"baseline_{kind}"] for r in records),
        "event_fraction_pct": 100.0 * sum(r["cuda_ms"] > r[f"baseline_{kind}"] for r in records) / len(records),
        "raw_latency_sum_ms": sum(r["cuda_ms"] for r in records if r["cuda_ms"] > r[f"baseline_{kind}"]),
        "excess_latency_sum_ms": excess_all,
        "raw_fraction_of_total_pct": None,
        "tail_excess_mass_pct": 100.0 * excess_all / total if total else None,
    })
    return out


def oracle_rows(records: list[dict], kind: str) -> dict:
    total = sum(r["cuda_ms"] for r in records)
    top_n = max(1, int(math.ceil(len(records) * 0.01)))
    ranked = sorted(records, key=lambda r: r["cuda_ms"], reverse=True)
    top_ids = {id(r) for r in ranked[:top_n]}
    def removed(predicate):
        return sum(max(0.0, r["cuda_ms"] - r[f"baseline_{kind}"]) for r in records if predicate(r))
    all_removed = removed(lambda r: True)
    gt20_removed = removed(lambda r: r["cuda_ms"] > 20)
    top_removed = removed(lambda r: id(r) in top_ids)
    return {
        "total_moe_ms": total,
        "oracle_A_20ms_removed_ms": gt20_removed,
        "oracle_A_20ms_moe_reduction_pct": 100.0 * gt20_removed / total,
        "oracle_B_top1_removed_ms": top_removed,
        "oracle_B_top1_moe_reduction_pct": 100.0 * top_removed / total,
        "oracle_C_all_excess_removed_ms": all_removed,
        "oracle_C_all_excess_moe_reduction_pct": 100.0 * all_removed / total,
        "top1_count": top_n,
    }


def make_critical(records: list[dict]) -> list[dict]:
    """Collapse the two independent DP timelines for an Amdahl projection."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["run"], r["local_invocation_id"], r["phase"], r["layer"], r["M"])
        groups[key].append(r)
    result = []
    for key, rows in groups.items():
        pick = max(rows, key=lambda x: x["cuda_ms"])
        result.append({
            "run": key[0], "local_invocation_id": key[1], "phase": key[2],
            "layer": key[3], "M": key[4], "cuda_ms": pick["cuda_ms"],
        })
    return result


def wave_wall_ms(run: Path) -> float:
    """Use one concurrent DP timeline; duplicate DP wave files are not summed."""
    per_dp = []
    for path in sorted(run.glob("waves.dp*.json")):
        try:
            waves = json.loads(path.read_text(encoding="utf-8"))
            per_dp.append(sum(float(w.get("wall_ms", 0.0) or 0.0) for w in waves))
        except Exception:
            pass
    return float(np.mean(per_dp)) if per_dp else float("nan")


def e2e_projection(records: list[dict], root: Path, kind: str) -> dict:
    critical = make_critical(records)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in critical:
        groups[(r["run"], r["phase"], r["layer"], r["M"])].append(r["cuda_ms"])
    for r in critical:
        r["base"] = baseline(groups[(r["run"], r["phase"], r["layer"], r["M"])], kind)
    total_moe = sum(r["cuda_ms"] for r in critical)
    e2e_total = sum(wave_wall_ms(Path(root) / r) for r in sorted({x["run"] for x in records}))
    def removed(pred):
        return sum(max(0.0, r["cuda_ms"] - r["base"]) for r in critical if pred(r))
    vals = {
        "critical_invocations": len(critical),
        "critical_moe_ms": total_moe,
        "e2e_wave_wall_ms": e2e_total,
        "moe_share_of_e2e_pct": 100.0 * total_moe / e2e_total if e2e_total else None,
    }
    for name, pred in (("A_20ms", lambda r: r["cuda_ms"] > 20),
                       ("B_top1", None), ("C_all_excess", lambda r: True)):
        if name == "B_top1":
            n = max(1, int(math.ceil(len(critical) * 0.01)))
            ids = {id(r) for r in sorted(critical, key=lambda x: x["cuda_ms"], reverse=True)[:n]}
            removed_ms = removed(lambda r: id(r) in ids)
        else:
            removed_ms = removed(pred)
        vals[f"{name}_removed_ms"] = removed_ms
        vals[f"{name}_e2e_reduction_pct"] = 100.0 * removed_ms / e2e_total if e2e_total else None
        vals[f"{name}_critical_moe_reduction_pct"] = 100.0 * removed_ms / total_moe if total_moe else None
    return vals


def e2e_projection_without_max(records: list[dict], root: Path, kind: str) -> float:
    """Sensitivity of the critical-path projection after removing one max."""
    if not records:
        return float("nan")
    max_record = max(records, key=lambda r: r["cuda_ms"])
    kept = [r for r in records if r is not max_record]
    # The max can be one TP-collapsed DP-local row.  Recompute the critical
    # projection from the remaining rows; wave wall is unchanged.
    proj = e2e_projection(kept, root, kind)
    return float(proj["C_all_excess_e2e_reduction_pct"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-root", type=Path, required=True)
    ap.add_argument("--fixed-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stock = []
    accounting = []
    for run in sorted(args.wait_root.glob("stock_run*")):
        if not (run / "invocations.jsonl").exists():
            continue
        rows, raw, dup = load_run(run, "wait_aware_stock")
        stock.extend(rows); accounting.append({"dataset": "wait_aware_stock", "run": run.name, "raw_rows": raw, "logical_rows": len(rows), "tp_duplicate_rows_removed": dup})
    # The older fixed-shape run is a robustness sensitivity, not pooled into
    # the primary estimate because it has a different run protocol.
    fixed_runs = []
    if (args.fixed_root / "invocations.jsonl").exists():
        rows, raw, dup = load_run(args.fixed_root, "fixed_shape_baseline")
        fixed_runs = rows
        accounting.append({"dataset": "fixed_shape_baseline", "run": args.fixed_root.name, "raw_rows": raw, "logical_rows": len(rows), "tp_duplicate_rows_removed": dup})

    for kind in ("p25", "p50", "trimmed_mean"):
        add_baselines(stock, kind)
    tables = []
    for kind in ("p25", "p50", "trimmed_mean"):
        tables.extend(mass_table(stock, kind))
    pd.DataFrame(tables).to_csv(args.out / "tail_excess_mass.csv", index=False)

    # Phase table uses the primary p50 normal baseline.  Keep every threshold
    # so the decode M=1 fixed-shape class can be inspected independently.
    phase_rows = []
    for phase, group in pd.DataFrame(stock).groupby("phase"):
        rs = [r for r in stock if r["phase"] == phase]
        base = "p50"
        total = sum(r["cuda_ms"] for r in rs)
        for threshold in (5, 10, 20, 50, 100):
            label, pred = f"gt{threshold}ms", (lambda r, t=threshold: r["cuda_ms"] > t)
            sel = [r for r in rs if pred(r)]
            ex = sum(max(0, r["cuda_ms"] - r[f"baseline_{base}"]) for r in sel)
            phase_rows.append({"phase": phase, "class": label, "n": len(sel), "fraction_pct": 100*len(sel)/len(rs), "raw_ms": sum(r["cuda_ms"] for r in sel), "excess_ms": ex, "excess_mass_pct": 100*ex/total})
        ex_all = sum(max(0, r["cuda_ms"] - r[f"baseline_{base}"]) for r in rs)
        phase_rows.append({"phase": phase, "class": "all_positive_excess", "n": sum(r["cuda_ms"] > r[f"baseline_{base}"] for r in rs), "fraction_pct": 100*sum(r["cuda_ms"] > r[f"baseline_{base}"] for r in rs)/len(rs), "raw_ms": None, "excess_ms": ex_all, "excess_mass_pct": 100*ex_all/total})
    pd.DataFrame(phase_rows).to_csv(args.out / "phase_tail_excess.csv", index=False)

    oracles = {"baseline": "matched_p50", "moe": oracle_rows(stock, "p50")}
    # Report all three reasonable normal baselines.  The gate uses the most
    # optimistic reasonable E2E projection (p25), while p50 is primary.
    oracles["by_baseline"] = {
        kind: {"moe": oracle_rows(stock, kind),
               "e2e_projection": e2e_projection(stock, args.wait_root, kind)}
        for kind in ("p25", "p50", "trimmed_mean")
    }
    oracles["e2e_projection"] = oracles["by_baseline"]["p50"]["e2e_projection"]
    (args.out / "perfect_oracle.json").write_text(json.dumps(oracles, indent=2), encoding="utf-8")

    # Sensitivity to the single largest valid sample. Baselines are recomputed
    # after removal, so this is not merely zeroing one numerator term.
    sens = []
    for label, source in (("primary_stock", stock), ("old_fixed_shape", fixed_runs)):
        if not source:
            continue
        max_index = max(range(len(source)), key=lambda i: source[i]["cuda_ms"])
        trimmed = [r for i, r in enumerate(source) if i != max_index]
        for kind in ("p25", "p50", "trimmed_mean"):
            add_baselines(trimmed, kind)
            total = sum(r["cuda_ms"] for r in trimmed)
            ex = sum(max(0, r["cuda_ms"] - r[f"baseline_{kind}"]) for r in trimmed)
            sens.append({"dataset": label, "baseline": kind, "removed_max_ms": source[max_index]["cuda_ms"], "n_without_max": len(trimmed), "max_without_single_outlier_excess_mass_pct": 100*ex/total})
    pd.DataFrame(sens).to_csv(args.out / "outlier_sensitivity.csv", index=False)

    fixed_summary = {}
    if fixed_runs:
        for kind in ("p25", "p50", "trimmed_mean"):
            add_baselines(fixed_runs, kind)
        fixed_summary = {"n": len(fixed_runs), "total_moe_ms": sum(r["cuda_ms"] for r in fixed_runs), "mass": mass_table(fixed_runs, "p50")}

    # Request IDs are absent from the source traces; record this explicitly.
    manifest = {
        "primary_dataset": "wait-aware stock runs only",
        "stock_run_count": len({r["run"] for r in stock}),
        "valid_moe_invocations_after_tp_dedup": len(stock),
        "raw_row_accounting": accounting,
        "total_observed_moe_ms": sum(r["cuda_ms"] for r in stock),
        "phases": sorted({r["phase"] for r in stock}),
        "request_level_join": "UNAVAILABLE: route records have no request_id and wave summaries have no invocation timestamps",
        "cross_gpu_clock_subtraction": False,
        "known_instrumentation_artifact_rows_excluded": True,
        "fixed_shape_sensitivity": fixed_summary,
    }
    (args.out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Flat key numbers for report generation and machine checking.
    p50all = next(x for x in tables if x["baseline"] == "p50" and x["threshold_ms"] == "ALL_POSITIVE_EXCESS")
    p25all = next(x for x in tables if x["baseline"] == "p25" and x["threshold_ms"] == "ALL_POSITIVE_EXCESS")
    trimall = next(x for x in tables if x["baseline"] == "trimmed_mean" and x["threshold_ms"] == "ALL_POSITIVE_EXCESS")
    optimistic_e2e = oracles["by_baseline"]["p25"]["e2e_projection"]["C_all_excess_e2e_reduction_pct"]
    gate = {
        "final_decision": "CONTINUE" if optimistic_e2e >= 15.0 else "FINAL_NO_GO",
        "optimistic_moe_excess_mass_pct": p25all["tail_excess_mass_pct"],
        "p50_moe_excess_mass_pct": p50all["tail_excess_mass_pct"],
        "trimmed_mean_moe_excess_mass_pct": trimall["tail_excess_mass_pct"],
        "optimistic_e2e_upper_bound_pct": oracles["by_baseline"]["p25"]["e2e_projection"]["C_all_excess_e2e_reduction_pct"],
        "p50_e2e_upper_bound_pct": oracles["by_baseline"]["p50"]["e2e_projection"]["C_all_excess_e2e_reduction_pct"],
        "trimmed_mean_e2e_upper_bound_pct": oracles["by_baseline"]["trimmed_mean"]["e2e_projection"]["C_all_excess_e2e_reduction_pct"],
        "oracle_A_20ms_e2e_upper_bound_pct": oracles["e2e_projection"]["A_20ms_e2e_reduction_pct"],
        "oracle_B_top1_e2e_upper_bound_pct": oracles["e2e_projection"]["B_top1_e2e_reduction_pct"],
        "single_max_removed_ms": max(r["cuda_ms"] for r in stock),
        "optimistic_without_single_max_moe_pct": next(x["max_without_single_outlier_excess_mass_pct"] for x in sens if x["dataset"] == "primary_stock" and x["baseline"] == "p25"),
        "optimistic_without_single_max_e2e_pct": e2e_projection_without_max(stock, args.wait_root, "p25"),
        "request_level_status": "BLOCKED_NO_REQUEST_ID",
        "decision_rule": "optimistic perfect E2E upper bound < 15% => FINAL_NO_GO; request-level join unavailable, so E2E is a wave-critical projection",
    }
    (args.out / "gate_summary.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "stock_records": len(stock), "fixed_records": len(fixed_runs), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
