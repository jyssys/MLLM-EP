"""Render the idle-window diagnostic, causal replay and offline fit reference."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


LABELS = {"idle_only": "Idle-only admission", "causal_handoff_cadence": "Past-only cadence",
          "offline_next_ready": "Next-native-ready oracle"}
STYLES = {"idle_only": ("#aa592f", "o-"), "causal_handoff_cadence": ("#315f92", "s-"),
          "offline_next_ready": ("#347e72", "^--")}


def totals(data, reserve, family=None):
    values = defaultdict(Counter)
    for row in data["evaluation"]:
        if family is not None and row["case"]["family"] != family:
            continue
        for result in row["policy_replays"]:
            if result["reserve_ms"] == reserve:
                counts = values[result["quantum_ms"], result["policy"]]
                for name in ("admissions", "within_reserve", "crosses_native_handoff", "consumes_reserve", "unobserved"):
                    counts[name] += result[name]
    return values


def figure(data, output):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "mllm_epd_modality_aware_idle_execution_discovery"
    import matplotlib.pyplot as plt

    windows = defaultdict(list)
    for row in data["evaluation"]:
        windows[row["case"]["family"]].extend(row["D_proxy_quiet_windows"]["interval_ms"])
    reserve = max(data["contract"]["reserves_ms"])
    quanta = data["contract"]["quantum_ms"]
    fig, (idle, safe, cross) = plt.subplots(1, 3, figsize=(13.3, 4.05), layout="constrained")

    def survival(ax, lengths, color, style, name):
        points = sorted(x for x in lengths if x > 0)
        if points:
            n = len(points)
            ax.step(points, [100 * (n - i) / n for i in range(n)], where="pre",
                    color=color, linestyle=style, label=name, linewidth=1.5)

    for i, (family, durations) in enumerate(windows.items()):
        survival(idle, durations, ["#315f92", "#aa592f", "#78588a"][i % 3], "-", family + " clean proxy")
    kernel_profiles = [row for row in data["profiles"] if "D_kernel_idle_proxy_quiet" in row]
    if kernel_profiles:
        sample = max(kernel_profiles, key=lambda row: (row["evidence"] == "TRACE_ANALYSIS", row["case"]["count"]))
        qualifier = "qualified" if sample["evidence"] == "TRACE_ANALYSIS" else "diagnostic"
        survival(idle, sample["D_kernel_idle_proxy_quiet"]["interval_ms"], "#324238", "--",
                 f"N{sample['case']['count']} kernel-idle ∩ proxy quiet ({qualifier})")
    graph_profiles = [row for row in data["profiles"] if "D_outside_graph_and_proxy_quiet" in row]
    if graph_profiles:
        sample = max(graph_profiles, key=lambda row: row["case"]["count"])
        qualifier = "qualified" if sample["evidence"] == "TRACE_ANALYSIS" else "diagnostic"
        survival(idle, sample["D_outside_graph_and_proxy_quiet"]["interval_ms"], "#79558c", "-.",
                 f"N{sample['case']['count']} graph-clear ∩ proxy quiet ({qualifier})")
    idle.axvline(1149, color="#807488", linestyle=":", linewidth=1.2, label="N24 D-local HTTP median")
    idle.set(xscale="log", xlabel="Contiguous window (ms)", ylabel="Windows at least this long (%)",
             title="Observed quiet-window duration")
    idle.legend(frameon=False, fontsize=7)

    family = "long" if "long" in windows else next(iter(windows))
    long = totals(data, reserve, family)
    for name, (color, marker) in STYLES.items():
        safe.plot(quanta, [long[q, name]["within_reserve"] for q in quanta], marker,
                  color=color, linewidth=1.4, markersize=4, label=LABELS[name])
    safe.set(xscale="log", xlabel="Hypothetical work quantum (ms)",
             ylabel="Admissions with reserve", title=f"{family} E-ready decision replay   {reserve:g} ms reserve")
    safe.legend(frameon=False, fontsize=7)

    for name in ("idle_only", "causal_handoff_cadence"):
        color, marker = STYLES[name]
        for fam in windows:
            count = totals(data, reserve, fam)
            values = [(q, 100. * count[q, name]["crosses_native_handoff"] / count[q, name]["admissions"])
                      for q in quanta if count[q, name]["admissions"]]
            if values:
                cross.plot(*zip(*values), marker, color=color, linewidth=1.3, markersize=4,
                           alpha=1. if fam == family else .47, label=f"{LABELS[name]} · {fam}")
    cross.set(xscale="log", xlabel="Hypothetical work quantum (ms)",
              ylabel="Admissions crossing native handoff (%)", ylim=(-3, 105), title="Frozen-baseline collision rate")
    cross.legend(frameon=False, fontsize=7)
    fig.suptitle("Native EPD idle feasibility   Physical intervals and replay   The next-ready oracle uses future handoffs")
    fig.savefig(output, format="svg", metadata={"Date": None}, bbox_inches="tight")
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


def num(value, digits=1):
    return f"{value:,.{digits}f}" if value is not None else "—"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(c) for c in row) + " |" for row in rows)]


def observer_table(profiles):
    rows = []
    metrics = ("duration_seconds", "TTFT_p50_ms", "TTFT_p95_ms", "TPOT_p95_ms", "ITL_p99_ms")
    for row in profiles:
        observed = row["observer_gate"].get("metrics")
        if observed:
            rows.append((f"`{row['case']['epoch']}/{row['case']['tag']}`",
                         *("/".join(num(value * 100, 2) + "%" if value is not None else "—" for value in (
                             observed[name]["clean_relative_spread"], observed[name]["profile_relative_deviation"])) for name in metrics)))
    return table(("Physical capture", "Duration", "TTFT p50", "TTFT p95", "TPOT p95", "ITL p99"), rows)


def report(data, link_root):
    cases = data["evaluation"]
    reserve = max(data["contract"]["reserves_ms"])
    minimum = data["contract"]["minimum_P_outstanding"]
    summary = ["# MLLM EPD idle motivation qualification", "",
               "## Native proxy traces", "",
               "The prior GPU profiles carry the diagnostic label under the 5% observer gate. The clean proxy analysis "
               "also has diagnostic scope. The future-aware fit is an offline oracle.", "",
               f"{len(cases)} held-out serving arms provide {sum(row['E_ready_decisions'] for row in cases)} E-ready decision timestamps "
               "inside the central p20–p80 scheduled-arrival interval. The analysis joins complete native E→P→D requests. "
               "An outstanding proxy request covers its native P-ready moment through D HTTP completion. This is the stage controller's "
               "request boundary and leaves other work, including GPU kernels and native connector activity, to separate measurements.", ""]
    summary += table(("Clean native cohort", "Campaign", "ROI s", "D proxy quiet", "Quiet p50/p95/max ms",
                      "E-ready calls", f"D quiet and P outstanding ≥{minimum}", "Eligible-call residual p50/p95/max ms"), [
        (f"`{r['case']['epoch']}/{r['case']['tag']}`", r["case"]["campaign"],
         num(r["ROI_seconds"]), num(100 * r["D_proxy_quiet_fraction"]) + "%",
         "/".join(num(r["D_proxy_quiet_windows"]["quantiles_ms"][key]) for key in ("p50", "p95", "max")),
         r["E_ready_decisions"], r["E_ready_D_proxy_quiet_P_pending"],
         "/".join(num(r["E_ready_to_actual_next_native_P_ready_ms"][key]) for key in ("p50", "p95", "max")))
        for r in cases])

    kernel_profiles = [row for row in data["profiles"] if "D_kernel_idle_proxy_quiet" in row
                       and row["case"]["campaign"] == "previous"]
    if kernel_profiles:
        summary.extend(["", "## Prior physical captures", "",
                        "These fits use physically observed D kernel intervals and preserve the original observer classification. "
                        "The optional connector column also excludes the native READ enclosures with complete accounting.", ""])
        summary += table(("GPU capture", "E/P/D kernel activity", "D kernel idle and proxy quiet p50/p95/max ms",
                          "Also outside accounted native READs, max ms", "Evidence"), [
            (f"`{r['case']['epoch']}/{r['case']['tag']}`",
             "/".join(num(100 * r["E_P_D_kernel_busy_fraction"][role]) for role in ("encoder", "prefill", "decode")) + "%",
             "/".join(num(r["D_kernel_idle_proxy_quiet"]["quantiles_ms"][k]) for k in ("p50", "p95", "max")),
             num(r["D_kernel_idle_proxy_quiet_outside_NIXL"]["quantiles_ms"]["max"])
             if r["D_kernel_idle_proxy_quiet_outside_NIXL"] is not None else "—", r["evidence"])
            for r in kernel_profiles])
    current_exact = [row for row in data["profiles"] if "observed_ROI_seconds_after_first_arrival" in row]
    if current_exact:
        summary.extend(["", "## Current exact-kernel capture", "",
                        "The delayed trace is analysed over the intersection of the canonical scheduled-arrival interval and "
                        "all three actual CUPTI anchor envelopes. All E/P/D activity fractions share this trimmed monotonic window. "
                        "The cohort-level observer guard compares the full clean-before, profile and clean-after requests.", ""])
        summary += table(("Physical capture", "Canonical p20–p80 s", "Actually observed s", "Canonical coverage",
                          "E/P/D kernel activity", "E/P kernel busy and D kernel idle", "D kernel idle and proxy quiet",
                          "Observer gate", "Evidence"), [
            (f"`{r['case']['epoch']}/{r['case']['tag']}`",
             "–".join(num(p, 2) for p in r["canonical_ROI_seconds_after_first_arrival"]),
             "–".join(num(p, 2) for p in r["observed_ROI_seconds_after_first_arrival"]),
             num(100 * r["canonical_ROI_coverage_fraction"]) + "%",
             "/".join(num(100 * r["E_P_D_kernel_busy_fraction"][role]) for role in ("encoder", "prefill", "decode")) + "%",
             num(100 * r["E_P_kernel_busy_and_D_kernel_idle_fraction"]) + "%",
             num(100 * r["D_kernel_idle_proxy_quiet_fraction"]) + "%",
             r["observer_gate"]["status"], r["evidence"])
            for r in current_exact])
        summary.extend(["", "P-ready duration has sample-and-hold scope. Decode SLO covers requests with all output token emissions "
                        "inside the common window. P throughput uses actual completed prompt tokens per common-window second.", ""])
        readiness_rows = []
        for item in current_exact:
            state = item["P_scheduler_state"] or {}
            admit = state.get("sampled_admission_operations") or {}
            measured_slo = item["decode_SLO_wholly_in_ROI"]
            readiness_rows.append((f"`{item['case']['epoch']}/{item['case']['tag']}`",
                                   *(num(100 * state[key]) + "%" if key in state else "—" for key in (
                                       "observed_fraction", "P_ready_fraction", "P_ready_and_D_kernel_proxy_quiet_fraction")),
                                   num(item["prefill_input_tokens_per_second"]),
                                   f"{admit['kv_allocate_rejected']}/{admit['kv_allocate_attempts']}"
                                   if admit.get("kv_allocate_attempts") is not None else "—",
                                   f"{measured_slo['passes']}/{measured_slo['requests']}"))
        summary += table(("Physical capture", "P state covered", "P known ready", "P ready plus idle/quiet D", "P prompt tokens/s",
                          "Sampled P KV rejects/attempts", "D absolute SLO"), readiness_rows)
        summary.extend(["", "Host queue values are means of periodic snapshots. The D quiet envelope uses request state at the proxy.",
                        "Transfer coverage is stated separately because absent kernel activity alone leaves native reads and DMA unobserved.", ""])
        host_rows = []
        for item in current_exact:
            queue = item["host_queue_samples"] or {}
            roles = queue.get("by_role", {})
            gap = item["D_kernel_idle_proxy_quiet"]["quantiles_ms"]
            host_rows.append((f"`{item['case']['epoch']}/{item['case']['tag']}`", queue.get("sample_count", "—"),
                              *(num(roles.get(role, {}).get(metric, {}).get("mean"), 2) for role, metric in (
                                  ("encoder", "vllm:num_requests_waiting"), ("prefill", "vllm:num_requests_waiting"),
                                  ("decode", "vllm:num_requests_waiting_by_reason:capacity"))),
                              "/".join(num(gap[key]) for key in ("p50", "p95", "max")),
                              "Captured" if item["D_DMA_activity_traced"] else "Unavailable",
                              "Accounted" if item["D_kernel_idle_proxy_quiet_outside_NIXL"] is not None else "Unavailable"))
        summary += table(("Physical capture", "Host samples", "E waiting", "P waiting", "D capacity waiting",
                          "D idle and quiet p50/p95/max ms", "D CUDA copy trace", "Native READ enclosures"), host_rows)
        summary.extend(["", "The observer table gives clean spread / profile departure from the clean center. Each limit is 5%, "
                        "with complete matching outputs and intact collector clocks.", "", *observer_table(current_exact)])
    graph_profiles = [row for row in data["profiles"] if "D_outside_graph_and_proxy_quiet" in row]
    if graph_profiles:
        summary.extend(["", "## D graph-envelope captures", "",
                        "E and P activity comes from individual physical kernels. D activity here is an envelope around each replayed CUDA "
                        "graph. Time outside these envelopes can include individual D kernels or connector work. These overlap values "
                        "have graph scope and carry separate observer qualification.", ""])
        summary += table(("Graph capture", "E/P kernel activity", "D graph envelope coverage", "D outside graph and proxy quiet",
                          "E/P kernels busy inside those D intervals", "Also outside accounted READs", "5% observer gate", "Evidence"), [
            (f"`{r['case']['epoch']}/{r['case']['tag']}`",
             "/".join(num(100 * r["E_P_kernel_busy_fraction"][role]) for role in ("encoder", "prefill")) + "%",
             num(100 * r["D_graph_envelope_fraction"]) + "%", num(100 * r["D_outside_graph_and_proxy_quiet_fraction"]) + "%",
             num(100 * r["E_P_kernel_busy_and_D_graph_clear_proxy_quiet_fraction"]) + "%",
             num(100 * r["D_graph_clear_proxy_quiet_outside_accounted_READ_fraction"]) + "%"
             if "D_graph_clear_proxy_quiet_outside_accounted_READ_fraction" in r else "—",
             r["observer_gate"]["status"], r["evidence"])
            for r in graph_profiles])
        summary.extend(["", "The cells report clean spread / profile departure from the clean center. Each limit is 5%, "
                            "with complete matching outputs and intact collector clocks.", ""])
        summary += observer_table(graph_profiles)

    failures = data.get("graph_failures", ())
    if failures:
        summary.extend(["", "## Failed D graph-observer attempt", ""])
        for row in failures:
            status = row["statuses"]
            clean = (status["clean_before"] or {}).get("MEASURED_ONLINE", 0)
            done = (status["profile"] or {}).get("MEASURED_ONLINE", 0)
            failed = (status["profile"] or {}).get("FAILED_EXECUTION", 0)
            summary.extend([f"`{row['case']}` · `DIAGNOSTIC`. Clean-before completed {clean}/{row['planned']}. "
                            f"The profiled arm completed {done}/{row['planned']} and recorded {failed} explicit failures. "
                            + ("The clean-after arm has no client receipt." if status["clean_after"] is None else ""),
                            "The D API accepted the profile begin RPC after the engine had already instantiated its replay graphs. "
                            "The native segfault stack reached `cuGraphLaunch`, `cudaGraphLaunch`, and PyTorch graph replay. "
                            "The D graph CSV and close ROI were absent. E later reached the 12 GiB backing-storage guard for live encoder tensors "
                            "while the failed stream could no longer retire normally. That guard was a secondary failure.", ""])
        summary.extend(["The strongest explanation is the graph-instrumentation activation order. NVIDIA documents selecting graph "
                        "tracing before graph executable instantiation. Retrofitting pre-existing graphs lacks the required timing instrumentation. "
                        "A clean early-activation smoke would be needed to isolate the exact driver fault. This attempt supplies no D graph duration or "
                        "observer-qualified activity window. [CUPTI trace-level lifetime](https://docs.nvidia.com/cupti/main/main.html#trace-level-selection-and-lifetime)",
                        "", f"[Diagnostic excerpts and source log hashes]({link_root}/graph_failure_receipt.json)"])

    summary.extend(["", "## Admission at E readiness", "",
                    f"The reactive comparator launches a fixed quantum when D has no outstanding proxy request and at least {minimum} "
                    "earlier prefills remain outstanding. The causal cadence comparator also requires its predicted time to the next native "
                    "handoff to exceed the quantum plus reserve. Prediction subtracts time since the most recent observed P-ready from a "
                    "lower empirical handoff interval frozen on separate cohorts. Prediction stays unavailable before the first observed handoff.", "",
                    "The oracle uses the actual next native P-ready after each decision. The scoring keeps all original native handoffs, "
                    "including the diverted candidate's native copy, and prevents hypothetical assists from overlapping on D. "
                    "A crossing is an overlap with ready native work. The replay measures neither a resulting ITL violation nor a service gain. "
                    "A visual operator that becomes runnable before whole-request E readiness has a different decision surface and needs its own trace. "
                    "Some held-out cohorts share a workload roster, seed or server epoch with calibration. This checks distinct cohorts under those shared conditions.", ""])
    summary += table(("Training family", "Calibration inter-handoff gaps", "Frozen empirical p10 ms"), [
        (family, value["training_gaps"], num(value["p10_native_inter_P_ready_ms"]))
        for family, value in data["calibration"].items()])
    for family in dict.fromkeys(r["case"]["family"] for r in cases):
        results = totals(data, reserve, family)
        summary.extend(["", f"### {family}, {reserve:g} ms reserve", "",
                        "Each entry gives admissions / within reserve / crossing the handoff. The rest consume the reserve or have an unobserved finish boundary.", ""])
        summary += table(("Quantum ms", "Idle only · diagnostic", "Past-only cadence · diagnostic", "Next-ready · offline oracle"), [
            (q, *("/".join(str(results[q, policy][key]) for key in ("admissions", "within_reserve", "crosses_native_handoff"))
                  for policy in LABELS)) for q in data["contract"]["quantum_ms"]])

    summary.extend(["", "## Live control and required state", "",
                    "The strongest generic implementation uses identical bounded work, waits for a truly admissible D slot, "
                    "and gates that quantum with a live P-progress-derived lower prediction of the next native handoff. "
                    "It should use the minimum over in-flight P requests, reserve time for import, yield, scheduling and connector operations, "
                    "and decline work when progress is stale or the prediction is uncalibrated. A modality-aware condition has incremental "
                    "value only if it improves decisions or changes the physical work and movement costs under this same control.", "",
                    "Clean proxy records expose submission and completion. Existing profiled P scheduler rows are sparse and record "
                    "the scheduler's token counters after schedule advances them. The pinned vLLM scheduler tracks completed progress via "
                    "`max(0, num_computed_tokens - num_in_flight_tokens)`. A live signal needs the request key and a monotonic freshness timestamp "
                    "along with that progress. It also needs D admission, import occupancy and "
                    "duration of the actual non-preemptible work quantum. A prior-window fit cannot supply those guarantees.", "",
                    f"[Figure]({link_root}/idle_feasibility.svg) · [Measured and replay reductions]({link_root}/idle_feasibility.json) · "
                    f"[Decision input and outcome rows]({link_root}/decision_inputs_and_outcomes.csv)"])
    fresh = [f"    --fresh-evaluate {row['case']['spec']} "
             for row in cases if row["case"]["campaign"] == "current"]
    fresh += [f"    --fresh-profile {row['case']['spec']} "
              for row in data["profiles"] if row["case"]["campaign"] == "current"]
    fresh += [f"    --fresh-failed-graph {row['case']} " for row in data.get("graph_failures", ())]
    summary.extend(["", "## Reproduction", "",
                    "The previous raw input can be recovered from the archived native evidence. The capture JSON records the source file digests and selected case IDs.",
                    "[Previous raw evidence guide](../mllm_epd_prefill_heavy_modality_discovery/results/raw_evidence/README.md)", "",
                    "```bash",
                    "module=poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery",
                    "study=poc_flashvep/mllm_epd_modality_aware_idle_execution_discovery",
                    'python -m "$module.idle_feasibility" --raw-root ../h200_epd_modality_runs ' + "\\"])
    if fresh:
        summary.append("    --fresh-root ../h200_idle_execution_runs " + "\\")
        summary.extend(line + "\\" for line in fresh)
    if data.get("graph_failures"):
        summary.append('    --graph-failure-receipt "$study/results/idle_feasibility/graph_failure_receipt.json" ' + "\\")
    summary.extend(['    --out "$study/results/idle_feasibility"',
                    'python -m "$module.idle_feasibility_report" ' + "\\",
                    '    --reduction "$study/results/idle_feasibility/idle_feasibility.json" ' + "\\",
                    "    --report poc_flashvep/reports/mllm_epd_idle_motivation_qualification.md " + "\\",
                    f"    --link-root {link_root}", "```"])
    return "\n".join(summary) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--link-root", required=True)
    args = parser.parse_args()
    data = json.loads(args.reduction.read_text())
    figure(data, args.reduction.with_name("idle_feasibility.svg"))
    args.report.write_text(report(data, args.link_root))


if __name__ == "__main__":
    main()
