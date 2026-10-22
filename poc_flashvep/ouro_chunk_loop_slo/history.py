"""Recompute historical request compliance, never infer it from pooled p99."""

import csv
import json
import numpy as np
from .common import ROOT, OLD, write


def table(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def q(values, p):
    return float(np.percentile(values, p)) if values else None


def main():
    surface = []
    slo = []
    request_stats = []
    for source in sorted((OLD / "raw").glob("*/*.json")):
        d = json.loads(source.read_text())
        family = source.parent.name
        if family not in ("fa4_grid_v1", "static_grid_v1") and not family.startswith(
            "fa4_confirm_warm_s"
        ):
            continue
        if d.get("warmup") or "requests" not in d:
            continue
        scope = "HISTORICAL_MEASURED_REANALYSIS"
        base = dict(
            source=str(source.relative_to(OLD)),
            run=family,
            mode=d["mode"],
            C=d["chunk_tokens"],
            G=d["loop_group"],
            P=d["prompt_bin"],
            N=d["concurrency"],
            trial=d["trial"],
            evidence=scope,
        )
        backgrounds = [r for r in d["requests"] if r["background"]]
        intervals = [x for r in backgrounds for x in r["measured_ITLs_s"]]
        long = next(r for r in d["requests"] if not r["background"])
        host = []
        sizes = []
        for quantum in d["quantums"]:
            if any(
                s["id"] == long["id"]
                and s["phase"] in ("prefill", "prefilling_recurrent")
                for s in quantum["spans"]
            ):
                host.append(1000 * (quantum["host_submit_end"] - quantum["host_begin"]))
                sizes.extend(
                    s["tokens"]
                    for s in quantum["spans"]
                    if s["id"] == long["id"]
                    and s["phase"] in ("prefill", "prefilling_recurrent")
                )
        makespan = d["end"] - min(r["arrival"] for r in d["requests"])
        surface.append(
            dict(
                base,
                long_TTFT_ms=1000 * long["TTFT_s"],
                ITL_p50_ms=1000 * q(intervals, 50),
                ITL_p95_ms=1000 * q(intervals, 95),
                ITL_p99_ms=1000 * q(intervals, 99),
                ITL_max_ms=1000 * max(intervals),
                observed_ITLs=len(intervals),
                long_prefill_submissions=len(host),
                submissions_per1000_prompt_tokens=1000
                * len(host)
                / long["prompt_tokens"],
                host_submit_mean_ms=float(np.mean(host)) if host else None,
                host_submit_p95_ms=q(host, 95),
                actual_C_mean=float(np.mean(sizes)) if sizes else None,
                prefill_effective_tokens_s=long["prompt_tokens"] / long["TTFT_s"],
                output_tokens_s=sum(len(r["output_ids"]) for r in d["requests"])
                / makespan,
                full_cohort_s=makespan,
                staged_bytes=d["memory"]["activation_bytes"],
                peak_allocated=d["memory"]["peak"],
                GPU_utilization="NOT_AVAILABLE_HISTORICAL",
                quantum_GPU_duration="NOT_AVAILABLE_CLEAN_HOST_ENQUEUE_IS_NOT_GPU_COMPLETION",
            )
        )
        all_itls = [v for r in d["requests"] for v in r["decode_ITLs_s"]]
        for r in d["requests"]:
            values = r["decode_ITLs_s"]
            row = dict(
                base,
                id=r["id"],
                background=r["background"],
                output_tokens=len(r["output_ids"]),
                TTFT_ms=1000 * r["TTFT_s"],
                ITL_p50_ms=1000 * q(values, 50),
                ITL_p95_ms=1000 * q(values, 95),
                ITL_p99_ms=1000 * q(values, 99),
                max_ITL_ms=1000 * max(values),
                E2E_s=r["E2E_s"],
                finish_reason=r["finish_reason"],
                quality_scope="BOUNDED128_NOT_COMPLETE_QUALITY",
            )
            for threshold in (50, 75, 100, 125, 150, 200, 250, 500):
                row[f"ITLs_over_{threshold}ms"] = sum(
                    v > threshold / 1000 for v in values
                )
            request_stats.append(row)
        for ttft in (0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0):
            for tbt in (0.075, 0.100, 0.125, 0.150, 0.200, 0.250, 0.500):
                tp = [r["TTFT_s"] <= ttft for r in d["requests"]]
                bp = [max(r["decode_ITLs_s"], default=0) <= tbt for r in d["requests"]]
                both = sum(a and b for a, b in zip(tp, bp))
                slo.append(
                    dict(
                        base,
                        TTFT_limit_s=ttft,
                        TBT_limit_s=tbt,
                        offered=len(tp),
                        TTFT_pass=sum(tp),
                        TBT_all_token_pass=sum(bp),
                        joint_pass=both,
                        joint_attainment=both / len(tp),
                        pooled_token_p99_s=q(all_itls, 99),
                        pooled_p99_pass=q(all_itls, 99) <= tbt,
                        drained_cohort_qualified_requests_s=both / makespan,
                        capacity_status="NOT_AN_OPEN_LOOP_RATE",
                    )
                )
    table("historical_cg_response_surface.csv", surface)
    table("historical_slo_grid.csv", slo)
    table("historical_itls_by_request.csv", request_stats)
    write(
        ROOT / "historical_reanalysis_summary.json",
        dict(
            surface_cells=len(surface),
            request_rows=len(request_stats),
            absolute_SLO_grid_rows=len(slo),
            historical_decision="CONDITIONAL_FIXED_QUANTUM_ONLY_UNCHANGED",
            new_GPU_measurements=False,
        ),
    )
    print("HISTORY", len(surface), "cells", len(slo), "SLO rows")


if __name__ == "__main__":
    main()
