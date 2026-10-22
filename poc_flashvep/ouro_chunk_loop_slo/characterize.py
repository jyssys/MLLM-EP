"""Per-token host submission exposure and C/G surfaces, not GPU attribution."""

import json
import numpy as np
from .common import ROOT
from .history import table, q


def main():
    surfaces = []
    exposures = []
    for path in sorted((ROOT / "raw").glob("grid*/*_T*.json")):
        d = json.loads(path.read_text())
        if "requests" not in d:
            continue
        long = next(r for r in d["requests"] if not r["background"])
        base = dict(
            job=path.parent.name,
            GPU=d["GPU"],
            startup=d["startup"],
            trial=d["trial"],
            P=d["prompt_bin"],
            N=d["concurrency"],
            mode=d["mode"],
            C=d["chunk_tokens"],
            G=d["loop_group"],
            evidence="LIVE_MEASURED_COUNTERBALANCED_CLOSED_COHORT",
        )
        long_q = []
        all_prefill_q = []
        issued = {r["id"]: 0 for r in d["requests"]}
        for quantum in d["quantums"]:
            if any(
                s["id"] == long["id"]
                and s["phase"]
                in ("prefill", "prefilling_prelude", "prefilling_recurrent")
                for s in quantum["spans"]
            ):
                all_prefill_q.append(quantum)
            for s in quantum["spans"]:
                if s["phase"] in ("prefill", "prefilling_recurrent"):
                    issued[s["id"]] += s["tokens"] * s["group"]
            if any(
                s["id"] == long["id"]
                and s["phase"] in ("prefill", "prefilling_recurrent")
                for s in quantum["spans"]
            ):
                long_q.append(quantum)
        assert all(issued[r["id"]] == r["prompt_tokens"] * 4 for r in d["requests"])
        intervals = []
        for r in d["requests"]:
            if not r["background"]:
                continue
            for token, (begin, end) in enumerate(
                zip(r["token_times"], r["token_times"][1:]), start=1
            ):
                if end < d["origin"]:
                    continue
                pref = [z for z in all_prefill_q if begin <= z["host_begin"] < end]
                core_pref = [z for z in long_q if begin <= z["host_begin"] < end]
                exposure = sum(
                    max(0, min(end, z["host_submit_end"]) - max(begin, z["host_begin"]))
                    for z in all_prefill_q
                )
                recurrent = [
                    s
                    for z in d["quantums"]
                    if begin <= z["host_begin"] < end
                    for s in z["spans"]
                    if s["id"] == r["id"] and s["phase"] == "recurrent"
                ]
                intervals.append(end - begin)
                exposures.append(
                    dict(
                        base,
                        id=r["id"],
                        token_index=token,
                        ITL_ms=1000 * (end - begin),
                        long_prefill_or_mixed_submissions=len(pref),
                        prefill_core_containing_submissions=len(core_pref),
                        background_recurrent_submissions=len(recurrent),
                        host_submission_exposure_ms=1000 * exposure,
                        attribution="HOST_INTERVAL_EXPOSURE_NOT_GPU_CRITICAL_PATH",
                    )
                )
        times = [1000 * (z["host_submit_end"] - z["host_begin"]) for z in long_q]
        actual = [
            s["tokens"] for z in long_q for s in z["spans"] if s["id"] == long["id"]
        ]
        surfaces.append(
            dict(
                base,
                long_TTFT_ms=1000 * long["TTFT_s"],
                ITL_p50_ms=1000 * q(intervals, 50),
                ITL_p95_ms=1000 * q(intervals, 95),
                ITL_p99_ms=1000 * q(intervals, 99),
                ITL_max_ms=1000 * max(intervals),
                ITL_over100_fraction=sum(v > 0.1 for v in intervals) / len(intervals),
                ITL_over200_fraction=sum(v > 0.2 for v in intervals) / len(intervals),
                prompt_tokens=long["prompt_tokens"],
                prefill_tokens_per_TTFT_s=long["prompt_tokens"] / long["TTFT_s"],
                long_prefill_submissions=len(long_q),
                long_prefill_submissions_definition="Core-containing only. Atomic includes its bundled prelude. Staged prelude is separate.",
                all_prefill_submissions_including_staged_prelude=len(all_prefill_q),
                separate_prefill_prelude_submissions=len(all_prefill_q) - len(long_q),
                submissions_per1000_prompt_tokens=1000
                * len(long_q)
                / long["prompt_tokens"],
                host_quantum_mean_ms=float(np.mean(times)),
                host_quantum_p95_ms=q(times, 95),
                actual_prompt_rows_mean=float(np.mean(actual)),
                staged_bytes=d["memory"]["activation_bytes"],
                peak_allocated_bytes=d["memory"]["peak"],
                full_R4_work_PASS=True,
                graph_captures_cumulative=d.get("graph_stats", {}).get("captures"),
                graph_replays_cumulative=d.get("graph_stats", {}).get("replays"),
                graph_fallbacks_cumulative=d.get("graph_stats", {}).get("fallbacks"),
                graph_entries_cumulative=d.get("graph_entries"),
                output_tokens=sum(len(r["output_ids"]) for r in d["requests"]),
                drained_cohort_s=d["end"] - min(r["arrival"] for r in d["requests"]),
                quantum_GPU_ms="NOT_AVAILABLE_CLEAN_RUN",
                GPU_utilization="SEE_SEPARATE_MONITOR_IF_RECORDED",
            )
        )
    table("cg_response_surface.csv", surfaces)
    table("token_prefill_exposure.csv", exposures)
    summary = []
    stalls = []
    for key in sorted({(r["P"], r["N"], r["mode"], r["C"]) for r in surfaces}):
        rows = [r for r in surfaces if (r["P"], r["N"], r["mode"], r["C"]) == key]
        selected = [
            r
            for r in exposures
            if (r["P"], r["N"], r["mode"], r["C"]) == key
            and r["long_prefill_or_mixed_submissions"] > 0
        ]
        whole = [r for r in exposures if (r["P"], r["N"], r["mode"], r["C"]) == key]
        row = dict(
            P=key[0],
            N=key[1],
            mode=key[2],
            C=key[3],
            cohorts=len(rows),
            observed_background_ITLs=len(whole),
            evidence="LIVE_MEASURED_POOLED_COUNTS_NOT_INDEPENDENT_TOKEN_INFERENCE",
        )
        for threshold in (50, 100, 150, 200, 250):
            number = sum(r["ITL_ms"] > threshold for r in whole)
            row[f"ITLs_over_{threshold}ms"] = number
            row[f"fraction_over_{threshold}ms"] = number / len(whole)
        stalls.append(row)
        summary.append(
            dict(
                P=key[0],
                N=key[1],
                mode=key[2],
                C=key[3],
                cohorts=len(rows),
                exposed_token_intervals=len(selected),
                exposed_interval_mean_prefill_submissions=float(
                    np.mean([r["long_prefill_or_mixed_submissions"] for r in selected])
                )
                if selected
                else None,
                exposed_interval_p95_prefill_submissions=q(
                    [r["long_prefill_or_mixed_submissions"] for r in selected], 95
                ),
                max_prefill_submissions_in_one_observed_ITL=max(
                    [r["long_prefill_or_mixed_submissions"] for r in selected],
                    default=0,
                ),
                exposure_host_ms_mean=float(
                    np.mean([r["host_submission_exposure_ms"] for r in selected])
                )
                if selected
                else None,
            )
        )
    table("token_prefill_exposure_summary.csv", summary)
    table("ITL_stall_counts.csv", stalls)
    print(
        "CHARACTERIZED", len(surfaces), "cohorts", len(exposures), "observed token gaps"
    )


if __name__ == "__main__":
    main()
