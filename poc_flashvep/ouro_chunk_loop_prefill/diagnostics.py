"""Summarize measured controls. CUDA intervals are not kernel-busy time."""

from collections import defaultdict
import numpy as np
from .common import ROOT, write
from .analyze import csvfile, metrics, scan


def pct(values, percentile):
    return float(np.percentile(values, percentile)) if values else None


def main():
    records = list(scan())
    quanta = []
    for path, data in records:
        if not data.get("heavy"):
            continue
        measured = {q["quantum"]: q for q in data["gpu_quantums"] if "quantum" in q}
        for i, quantum in enumerate(data["quantums"]):
            long = [s for s in quantum["spans"] if s["id"].startswith("long_")]
            if not long or i not in measured:
                continue
            prefill = [
                s for s in long if s["phase"] in ("prefill", "prefilling_recurrent")
            ]
            if not prefill:
                continue
            gpu = measured[i]
            quanta.append(
                dict(
                    run=path.parent.name,
                    mode=data["mode"],
                    C=data["chunk_tokens"],
                    G=data["loop_group"],
                    quantum=i,
                    phase=prefill[0]["phase"],
                    long_prompt_rows=sum(s["tokens"] for s in prefill),
                    all_active_rows=sum(s["tokens"] for s in quantum["spans"]),
                    actual_group=prefill[0]["group"],
                    host_submission_ms=1000
                    * (quantum["host_submit_end"] - quantum["host_begin"]),
                    CUDA_elapsed_ms=gpu["gpu_compute_ms"],
                    padded_graph_rows="NOT_AVAILABLE",
                    evidence="LIVE_MEASURED_HEAVY_INTERVAL_NOT_KERNEL_BUSY",
                )
            )
    csvfile("prefill_quantum_measurements.csv", quanta)
    aggregate = []
    keys = sorted({(q["run"], q["mode"]) for q in quanta})
    for run, mode in keys:
        q = [v for v in quanta if (v["run"], v["mode"]) == (run, mode)]
        aggregate.append(
            dict(
                run=run,
                mode=mode,
                quanta=len(q),
                CUDA_p50_ms=pct([x["CUDA_elapsed_ms"] for x in q], 50),
                CUDA_p95_ms=pct([x["CUDA_elapsed_ms"] for x in q], 95),
                CUDA_max_ms=max(x["CUDA_elapsed_ms"] for x in q),
                host_submission_p50_ms=pct([x["host_submission_ms"] for x in q], 50),
                host_submission_p95_ms=pct([x["host_submission_ms"] for x in q], 95),
                evidence="LIVE_MEASURED_HEAVY_INTERVAL_NOT_CRITICAL_PATH_SUM",
            )
        )
    csvfile("prefill_quantum_summary.csv", aggregate)

    observer = []
    labels = (
        "fa4_trace_v1",
        "fa4_observer_off_v1",
        "fa4_graph_off_v1",
        "fa4_prefill_only_v1",
    )
    reference = {
        (d["mode"], d["chunk_tokens"], d["prompt_bin"], d["concurrency"]): (p, d)
        for p, d in records
        if p.parent.name == "fa4_confirm_warm_s0"
        and d["trial"] == 2
        and not d.get("warmup")
    }
    for path, data in records:
        if path.parent.name not in labels or data.get("warmup"):
            continue
        row = metrics(path, data)
        key = (
            data["mode"],
            data["chunk_tokens"],
            data["prompt_bin"],
            data["concurrency"],
        )
        if key in reference:
            refpath, ref = reference[key]
            refmetrics = metrics(refpath, ref)
            a = {r["id"]: r for r in ref["requests"]}
            row.update(
                reference=refpath.parent.name,
                paired_requests=len(data["requests"]),
                exact_request_outputs=sum(
                    r["output_ids"] == a[r["id"]]["output_ids"]
                    for r in data["requests"]
                ),
                exact_request_depths=sum(
                    r["depths"] == a[r["id"]]["depths"] for r in data["requests"]
                ),
                E2E_ratio_to_light_observer=row["client_cohort_makespan_s"]
                / refmetrics["client_cohort_makespan_s"],
                TTFT_ratio_to_light_observer=row["long_TTFT_s"]
                / refmetrics["long_TTFT_s"],
            )
        observer.append(row)
    csvfile("observer_and_control_measurements.csv", observer)

    # Graph counters are cumulative in a resident engine. Preserve both forms.
    graphgroups = defaultdict(list)
    for path, data in records:
        graphgroups[
            (
                path.parent.name,
                data["mode"],
                data["chunk_tokens"],
                data["prompt_bin"],
                data["concurrency"],
            )
        ].append((path, data))
    graphrows = []
    for key, items in sorted(graphgroups.items()):
        previous = {}
        for path, data in sorted(
            items, key=lambda x: (not x[1].get("warmup", False), x[1]["trial"])
        ):
            current = data.get("graph_stats", {})
            deltas = {k: current[k] - previous.get(k, 0) for k in current}
            graphrows.append(
                dict(
                    run=key[0],
                    mode=key[1],
                    C=key[2],
                    P=key[3],
                    N=key[4],
                    trial=data["trial"],
                    warmup=data.get("warmup", False),
                    **{k + "_cumulative": v for k, v in current.items()},
                    **{k + "_increment": v for k, v in deltas.items()},
                    graph_entries=data["graph_entries"],
                    evidence="LIVE_RUNNER_COUNTERS_NOT_KERNEL_LAUNCH_COUNTS",
                )
            )
            if data.get("engine_reused"):
                previous = current
    csvfile("graph_counter_measurements.csv", graphrows)
    warm = [
        (p, d)
        for p, d in records
        if p.parent.name.startswith("fa4_confirm_warm_s") and not d.get("warmup")
    ]
    parity = list(
        __import__("csv").DictReader((ROOT / "natural_output_parity.csv").open())
    )
    warmparity = [
        r
        for r in parity
        if r["run"].startswith("fa4_confirm_warm_s")
        and r["mode"] == "grouped"
        and int(r["trial"]) >= 2
    ]
    tails = []
    for p, n in ((2048, 4), (8192, 8)):
        for mode, chunk in (
            ("atomic", 1024),
            ("split", 1024),
            ("mixed", 128),
            ("mixed", 1024),
            ("grouped", 1024),
        ):
            chosen = [
                d
                for _, d in warm
                if (d["prompt_bin"], d["concurrency"], d["mode"], d["chunk_tokens"])
                == (p, n, mode, chunk)
            ]
            intervals = [
                v
                for d in chosen
                for r in d["requests"]
                if r["background"]
                for v in r["measured_ITLs_s"]
            ]
            tails.append(
                dict(
                    P=p,
                    N=n,
                    mode=mode,
                    C=chunk,
                    cohorts=len(chosen),
                    observed_ITLs=len(intervals),
                    over50ms=sum(v > 0.050 for v in intervals),
                    over100ms=sum(v > 0.100 for v in intervals),
                    over200ms=sum(v > 0.200 for v in intervals),
                    note="Descriptive event counts only. Tokens are not independent inference units.",
                )
            )
    csvfile("ITL_stall_counts.csv", tails)
    write(
        ROOT / "diagnostic_summary.json",
        dict(
            heavy_prefill_quanta=len(quanta),
            warm_confirmation_cells=len(warm),
            warm_confirmation_requests=sum(len(d["requests"]) for _, d in warm),
            warm_confirmation_truncated=sum(
                r["finish_reason"] == "length" for _, d in warm for r in d["requests"]
            ),
            warm_confirmation_EOS=sum(
                r["finish_reason"] == "stop" for _, d in warm for r in d["requests"]
            ),
            grouped_vs_split_cohort_output_checks=len(warmparity),
            grouped_vs_split_exact_outputs=sum(
                r["tokens_exact"] == "True" for r in warmparity
            ),
            grouped_vs_split_exact_depths=sum(
                r["depths_exact"] == "True" for r in warmparity
            ),
            quality_status="NOT_RUN_COMPLETE_BENCHMARK_ACCURACY",
        ),
    )
    print("DIAGNOSTICS", len(quanta), "prefill intervals", len(observer), "controls")


if __name__ == "__main__":
    main()
