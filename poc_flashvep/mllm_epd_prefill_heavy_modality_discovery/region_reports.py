"""Render operating region, joint saturation, and fragmentation reports from one reduction."""

import argparse
from pathlib import Path

from .timeline import ROLES, read_json


def number(value, digits=2):
    return f"{value:,.{digits}f}" if value is not None else "—"


def percent(value):
    return f"{100 * value:.2f}%" if value is not None else "—"


def trace_fraction(row, key):
    return (row.get("trace") or {}).get("fraction_of_window", {}).get(key)


def label(row):
    return f"`{row['epoch']}/{row['tag']}`"


def field(row, name):
    return row.get("serving", {}).get("geometry", {}).get(name, {}).get("p50")


def trace_class(row):
    return (row.get("trace") or {}).get("evidence", "NOT_MEASURED")


def trace_priority(row):
    return (not row.get("M2_pass"), not row.get("M1_pass"), trace_class(row) != "TRACE_ANALYSIS",
            -row["serving"]["completed"], row["epoch"], row["tag"])


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)]


def count_ratio(value):
    return f"{value['passes']}/{value['requests']}" if value and value["requests"] else "—"


def references(root, names):
    return " · ".join(f"[{label}]({root}/{path})" for label, path in names)


def operating_region(report, prefix):
    cohorts = report["cohorts"]
    online = [row for row in cohorts if "serving" in row and row["metadata"].get("route", "native") == "native"]
    candidates = [row for row in online if trace_class(row) in {"TRACE_ANALYSIS", "DIAGNOSTIC"}]
    passed = [row for row in candidates if row.get("M1_pass")]
    status = ("CONFIRMED_PREFILL_HEAVY_EPD" if passed else "INCONCLUSIVE_PROFILING"
              if candidates and not any(trace_class(row) == "TRACE_ANALYSIS" for row in candidates)
              else "CONDITIONAL_PREFILL_HEAVY_EPD" if candidates else "NOT_RUN")
    result = ["# MLLM EPD prefill-heavy operating region", "", f"**Motivation assessment** `{status}`", "",
              f"Native online complete cohorts {sum(row['serving']['complete'] for row in online)}/{len(online)}. "
              f"Observer-qualified three-role captures {sum(trace_class(row) == 'TRACE_ANALYSIS' for row in candidates)}/{len(candidates)}. "
              f"Same-window M1 passes {len(passed)}.", ""]
    reproductions = [row for row in candidates if row.get("M1_criteria", {}).get("same_window_observer_pass") is False
                     and all(passed for key, passed in row["M1_criteria"].items() if key != "same_window_observer_pass")]
    if reproductions:
        result.extend([f"{len(reproductions)} diagnostic captures satisfy the non-observer M1 checks in one window. "
                       "Their physical measurements remain conditional evidence because the matched latency guard failed.", ""])
    result.extend([
              "## Contract", "",
              "The reducer uses one monotonic interval from the scheduled arrival at index floor(N×.20) to floor(N×.80). "
              "These endpoints stay inside the offered load and exclude drain. The cohort-level serving Goodput denominator runs from the first scheduled arrival to the last completed response. "
              "P service proximity divides P-ready completion tokens per ROI second by the strongest finite independent P reference for that cohort. "
              "Because completions are discrete and request sizes differ, a short-window ratio can exceed 100% and does not by itself establish saturation. "
              "The Decode SLO in the trace applies to successful decode outputs entirely inside this ROI. The online row also preserves all-cohort absolute decode success and joint SLO.", "",
              "Observer qualification uses duration, TTFT p50/p95, TPOT p95 and ITL p99. "
              "Both the relative clean spread and the profiled deviation from the clean center must be at most 5% on each metric. "
              "Request order, prompt contract, interarrival offsets and greedy tokens must match. Each role must record valid CUPTI events across the entire selected window. "
              "NVML SM activity is an independent weighted device counter. It can guide load selection. The M1 decision requires physical kernel spans.", "",
              "## Actual online sweep", ""])
    result += table(("Epoch and cohort", "QPS", "Median images", "Median LM input", "Median output", "Median visual share",
                     "E/P/D SM %", "P service proximity", "Decode SLO", "TTFT p95 ms", "Joint req/s", "Observer"), [
        (label(row), number(row.get("offered_qps"), 3), number(field(row, "image_count"), 1),
         number(field(row, "input_tokens_manifest"), 0), number(field(row, "output_tokens"), 1),
         percent(field(row, "visual_ratio")), "/".join(number((row.get("gpm") or {}).get("by_role", {}).get(role, {}).get("SM_UTIL"), 1)
                                                        for role in ROLES),
         percent(row.get("same_window_capacity_proximity", {}).get("prefill", {}).get("tokens")),
         f"{row['serving']['absolute_decode_pass']}/{row['serving']['planned']}",
         number(row["serving"]["TTFT_p95_ms"], 1), number(row["serving"]["joint_goodput_req_s"], 3),
         ", ".join(row["auxiliary_observers"]) if row.get("auxiliary_observers") else row["metadata"].get("observer", "clean"))
        for row in online])
    text_route = [row for row in cohorts if "serving" in row and row["metadata"].get("route") == "text"]
    text_controls = [row for row in text_route if row["serving"]["geometry"]["image_count"]["max"] == 0]
    if text_controls:
        result.extend(["", "## Text-only backbone control", "",
                       "The same checkpoint and BF16 native workers serve these no-image prompts through the 1P/1D path. "
                       "They skip the encoder and keep the model's original LM forward and NIXL handoff.", ""])
        result += table(("Epoch and cohort", "Offered req/s", "Median LM input", "P/D SM %", "P service proximity",
                         "Decode SLO", "TTFT p95 ms", "Joint req/s"), [
            (label(row), number(row.get("offered_qps"), 3), number(field(row, "input_tokens_manifest"), 0),
             "/".join(number((row.get("gpm") or {}).get("by_role", {}).get(role, {}).get("SM_UTIL"), 1)
                      for role in ("prefill", "decode")),
             percent(row.get("same_window_capacity_proximity", {}).get("prefill", {}).get("tokens")),
             f"{row['serving']['absolute_decode_pass']}/{row['serving']['planned']}", number(row["serving"]["TTFT_p95_ms"], 1),
             number(row["serving"]["joint_goodput_req_s"], 3)) for row in text_controls])
        if any(row["metadata"].get("source") == "controlled" and row.get("gpm") for row in text_controls):
            result.extend(["", references(prefix, (("Controlled visual-ratio activity", "figures/controlled_visual_ratio_sm_activity.svg"),))])
    mixed = [row for row in text_route if row["serving"]["geometry"]["image_count"]["min"] == 0
             and (row["serving"]["geometry"]["image_count"]["max"] or 0) > 0]
    if mixed:
        result.extend(["", "## Mixed multimodal and text traffic", "",
                       "The native routing control submits image prompts to E before P. Text prompts reach P directly. "
                       "Both use the same D worker and native NIXL.", ""])
        result += table(("Epoch and cohort", "Offered req/s", "Completed image", "Completed text", "Completed",
                         "E/P/D SM %", "Decode SLO", "TTFT p95 ms", "Joint req/s"), [
            (label(row), number(row["offered_qps"], 3),
             sum(event["completed"] and (event.get("image_count") or 0) > 0 for event in row["request_event_rows"]),
             sum(event["completed"] and event.get("image_count") == 0 for event in row["request_event_rows"]),
             f"{row['serving']['completed']}/{row['serving']['planned']}",
             "/".join(number((row.get("gpm") or {}).get("by_role", {}).get(role, {}).get("SM_UTIL"), 1) for role in ROLES),
             f"{row['serving']['absolute_decode_pass']}/{row['serving']['planned']}",
             number(row["serving"]["TTFT_p95_ms"], 1), number(row["serving"]["joint_goodput_req_s"], 3)) for row in mixed])
    result.extend(["", "## Profiled observations and decision", ""])
    result += table(("Epoch and cohort", "E/P/D kernel busy", "E/P/D physical region", "Evidence", "ROI with a recent P-ready sample",
                     "P service proximity", "Same-window Decode SLO", "M1 missing or unmet"), [
        (label(row), "/".join(percent(trace_fraction(row, role)) for role in ROLES), row["observed_region"], trace_class(row),
         percent(trace_fraction(row, "prefill_known_compute_ready_waiting"))
         if row["trace"]["scheduler"].get("prefill", {}).get("compute_ready_field_observed") else "—",
         percent(row.get("same_window_capacity_proximity", {}).get("prefill", {}).get("tokens")),
         count_ratio(row["trace"]["decode_SLO_wholly_in_ROI"]),
         ", ".join(("missing " if value is None else "") + name for name, value in row["M1_criteria"].items()
                   if value is not True) or "Pass") for row in candidates])
    if candidates:
        result.extend(["", "## Profiling observer check", "",
                       "Each metric passes only when both (A) the clean-before/after spread and (B) the profiled movement from the clean center stay within 5%. "
                       "Cells report A/B. The threshold also requires exact outputs, complete request accounting and an intact three-role trace.", ""])
        metrics = ("duration_seconds", "TTFT_p50_ms", "TTFT_p95_ms", "TPOT_p95_ms", "ITL_p99_ms")
        result += table(("Epoch and cohort", "Duration", "TTFT p50", "TTFT p95", "TPOT p95", "ITL p99", "Gate"), [
            (label(row), *(f"{percent(value.get('clean_relative_spread'))}/{percent(value.get('profile_relative_deviation'))}"
                           if (value := gate.get("metrics", {}).get(metric)) else "—" for metric in metrics), gate.get("status", "—"))
            for row in candidates if (gate := row.get("observer_gate"))])
    result.extend(["", "The labels classify physical activity only. R1 uses E ≥80%, P <80%, D ≤30%. R2 uses E <80%, P ≥80%, D ≤30%. "
                   "R3 uses E and P ≥80%, D ≤30%. R4 uses E and P ≥80%, D ≥50%. R5 uses D ≥50% and D at least as busy as either upstream stage. "
                   "Other measured combinations remain outside these bands. The primary M1 gate additionally requires a ready backlog, proximity, decode SLO, and usable timing evidence from the same capture. "
                   "Natural versus controlled scope follows the frozen cohort receipts.", "",
                   references(prefix, (("Reduced cohort and observation records", "operating_region.json"),
                                       ("Online cohort table", "online_cohorts.csv")))])
    if candidates:
        result.extend(["", references(prefix, (("Figure A", "figures/figure_a_operating_region.svg"),))])
    return result


def joint_saturation(report, prefix):
    rows = [row for row in report["cohorts"] if trace_class(row) in {"TRACE_ANALYSIS", "DIAGNOSTIC"}
            and row["metadata"].get("route", "native") == "native"]
    result = ["# Encoder–Prefill joint saturation", "",
              "E and P work overlap in a measured interval only when their clock-aligned kernel unions intersect. "
              "The three-way overlap additionally intersects the complement of D kernels. All fractions use the same central scheduled-arrival duration. "
              "E capacity proximity uses completed encoder HTTP requests and the corresponding independent E witness. "
              "The M2 screen uses an explicit 85% comparison with that witness and an E kernel threshold of 80%. "
              "P compute-ready sampling and D idle scope follow the operating-region contract.", "",
              "## Same-window intervals", ""]
    result += table(("Epoch and cohort", "ROI s", "E", "P", "D", "E ∩ P", "E ∩ P ∩ D idle", "E ∩ D idle",
                     "E capacity proximity", "M2", "Evidence"), [
        (label(row), number(row["trace"]["duration_seconds"]), *(percent(trace_fraction(row, role)) for role in ROLES),
         percent(trace_fraction(row, "E_busy_P_busy")), percent(trace_fraction(row, "E_busy_P_busy_D_kernel_idle")),
         percent(trace_fraction(row, "E_busy_D_kernel_idle")),
         percent(row.get("same_window_capacity_proximity", {}).get("encoder", {}).get("requests")),
         "Pass" if row.get("M2_pass") else "Open", trace_class(row)) for row in rows])
    result.extend(["", "The E→P readiness event here is a whole-request proxy acknowledgment. "
                   "An E HTTP span includes host preprocessing and export. P HTTP includes queueing, embedding import and model execution. "
                   "The matched E renderer span includes preprocessing and serializing the same-request processor handoff. "
                   "Optimizer claims about per-image readiness require separate image-level producer events. "
                   "Time left by recorded kernels does not establish that a stage can take additional model weights or prefill work without interference."])
    if rows:
        result.extend(["", references(prefix, (("Figure B", "figures/figure_b_joint_timeline.svg"),
                                              ("Event CSV", "request_timeline.csv")))])
    sampled = [row for row in report["cohorts"] if row.get("gpm") and row["metadata"].get("route", "native") == "native"
               and row["serving"]["complete"]]
    if sampled:
        result.extend(["", "## Device resources in the same scheduled-arrival interval", "",
                       "These duration-weighted NVML GPM samples quantify device resources. "
                       "SM active and tensor metrics are different quantities from the physical kernel union. "
                       "DRAM figures are device bandwidth-utilization percentages. NVLink metrics are device totals "
                       "and may contain concurrent connector traffic. Transfer time and GPU operator throughput need additional observations.", ""])
        positive = [min(rows, key=trace_priority)] if rows else []
        chosen = {f"{row['epoch']}/{row['tag']}": row for row in
                  positive + sorted(sampled, key=lambda row: -row["serving"]["completed"])[:3]}
        result += table(("Epoch and cohort", "Device slot", "SM active %", "Tensor activity %", "DRAM BW %",
                         "NVLink TX MiB/s", "NVLink RX MiB/s", "Counter ROI coverage"), [
            (label(row), role, *(number(counters.get(key), 2) for key in ("SM_UTIL", "ANY_TENSOR_UTIL", "DRAM_BW_UTIL",
                         "NVLINK_TOTAL_TX_PER_SEC", "NVLINK_TOTAL_RX_PER_SEC")), percent(row["gpm"]["coverage"].get(role)))
            for row in chosen.values() if row.get("gpm") for role, counters in row["gpm"]["by_role"].items()])
    witnesses = [row for row in report["cohorts"] if row.get("image_readiness")]
    if witnesses:
        result.extend(["", "## Original per-image transport readiness", "",
                       "An opted-in diagnostic polls existing CUDA IPC producer events. Each event binds the real encoder data to its original request-scoped image identifier. "
                       "A successful query gives an upper bound and the most recent failed query gives a lower bound. "
                       "The source bounds marker publication between save entry and return. Both the marker and the GPU data are needed for transport readiness. "
                       "These intervals describe the current encoder implementation. Individual vision-kernel completion, consumer clone, and useful early LM work require separate evidence.", ""])
        result += table(("Epoch and cohort", "Validated requests", "Native events", "Matched visual rows", "Matched LM span lengths",
                         "Multi-image requests", "Proven positive first-image lead", "Guaranteed lead p95 ms", "Possible lead p95 ms",
                         "Full ready before HTTP ack median lower ms"), [
            (label(row), f"{counts.get('validated_complete_requests', 0)}/{counts.get('eligible_completed_requests', 0)}",
             counts.get("validated_native_events", 0), counts.get("validated_total_visual_token_requests", 0),
             counts.get("validated_span_geometry_requests", 0),
             data["multimodal_complete_requests"], data["positive_guaranteed_first_image_gap_count"],
             number(data["guaranteed_first_image_gap_ms"]["p95"], 3),
             number(data["possible_first_image_gap_ms"]["p95"], 3),
             number(data["transport_ready_before_E_ack_ms_lower"]["p50"], 3))
            for row in witnesses if (data := row["image_readiness"]) and (counts := data["coverage"])])
        if any(row["image_readiness"]["requests"] for row in witnesses):
            result.extend(["", references(prefix, (("Per-image readiness bounds", "figures/native_per_image_readiness_bounds.svg"),))])
    return result


def fragmentation(report, prefix):
    rows = [row for row in report["cohorts"] if (row.get("trace") or {}).get("kernel_idle_histogram")
            and row["metadata"].get("route", "native") == "native"]
    result = ["# Decode idle fragmentation", "",
              "Kernel idle is the complement of the aligned D kernel union. Proxy request-free intervals require no request between P-ready and D HTTP completion. "
              "The recent-empty engine measure intersects observed kernel idle with bounded scheduler samples of an empty queue and no running requests. "
              "It covers only the time reported by scheduler samples and can miss a truly quiet engine between schedules. "
              "An interval with no CUPTI copy activity can still include DMA handled by the native connector. "
              "Native transfer-free and deadline-safe availability require separate evidence. All raw capture classifications below retain their observed/diagnostic labels.", "",
              "## Same-window idle decomposition", ""]
    result += table(("Epoch and cohort", "D kernel idle", "Also proxy request free", "Also sampled engine empty",
                     "Also CUPTI copy-free", "Also outside accounted NIXL READ", "P busy and D kernel idle",
                     "P busy and D proxy-quiet", "Evidence"), [
        (label(row), *(percent(trace_fraction(row, key)) for key in ("D_kernel_idle",
          "D_kernel_idle_proxy_request_free", "D_kernel_idle_recent_idle_snapshot", "D_kernel_idle_CUPTI_copy_free",
          "D_kernel_idle_outside_accounted_native_READ",
          "P_busy_D_kernel_idle", "P_busy_D_kernel_idle_proxy_request_free")), trace_class(row)) for row in rows])
    if rows:
        case = min(rows, key=trace_priority)
        result.extend(["", "## Observable idle causes", "",
            f"In {label(case)}, {percent(trace_fraction(case, 'D_kernel_idle_proxy_request_free'))} of the window is both kernel-idle and "
            f"without eligible D proxy work. The overlap with an active E HTTP request is {percent(trace_fraction(case, 'D_kernel_idle_proxy_free_with_E_HTTP'))} "
            f"and with an active P HTTP request {percent(trace_fraction(case, 'D_kernel_idle_proxy_free_with_P_HTTP'))}. "
            f"Kernel idle inside an active D HTTP request covers {percent(trace_fraction(case, 'D_kernel_idle_inside_D_HTTP'))}. "
            "Upstream HTTP overlap shows where requests are pending in the pipeline. E and P can overlap each other and the two shares can describe different requests.", ""])
        result.extend(["", "## Fragment length and hypothetical work fit", "",
            "The detail view prioritizes a same-window M2 pass, then M1, then the largest qualified capture. "
            f"The largest diagnostic is used when there is no qualified capture. Selected {label(case)} · `{trace_class(case)}`.", ""])
        result += table(("D idle ms", "Count", "Time s", "Share of idle", "P overlap s", "E overlap s",
                         "P-ready overlap s", "Next P-ready proxy p50 ms"), [
            (f"[{item['lower_ms']}, {item['upper_ms']})" if item["upper_ms"] is not None else f"≥{item['lower_ms']}",
             item["count"], number(item["seconds"], 3), percent(item["fraction_idle_time"]),
             number(item["intersection_seconds"].get("prefill"), 3),
             number(item["intersection_seconds"].get("encoder"), 3),
             number(item["intersection_seconds"].get("prefill_known_compute_ready_waiting"), 3),
             number(item["next_P_ready_proxy_ms"]["p50"], 2)) for item in case["trace"]["kernel_idle_histogram"]])
        result.extend(["", "The future P-ready proxy timestamp is retrospective and may follow the selected idle interval. "
                       "The fit table is an `OFFLINE_ORACLE` over contiguous D-kernel-idle and proxy-request-free intervals. "
                       "The P-ready column asks if a known compute-ready sample leaves a possible launch time inside the interval. "
                       "Fits do not account for unknown future arrivals, scheduler delay, staging, prefill compute shape, transport, or interruption.", ""])
        fits = case["trace"].get("retrospective_quantum_fit")
        if fits:
            result += table(("Hypothetical contiguous work ms", "Fits observed intervals",
                             "Fits with recent P-ready sample at possible start", "Total eligible interval s"), [
                (item["required_ms"], item["fit_windows"], number(item["fit_windows_with_recent_P_ready_possible_launch"], 0),
                 number(item["eligible_interval_seconds"], 4)) for item in fits])
            result.extend(["", "The 1,150 and 1,250 ms rows approximate the separately measured natural N24 and N64 naïve D-local "
                               "median HTTP submit-to-first-token enclosures. These comparisons apply measured work quanta to the native trace post hoc."])
        if durations := case["trace"].get("observed_opportunity_durations"):
            result.extend(["", "Measured contiguous intervals under the proxy and scheduler scopes.", ""])
            result += table(("Interval condition", "p50 ms", "p95 ms", "Maximum ms"), [
                (name, *(number(stats[q], 1) for q in ("p50", "p95", "max")))
                for name, stats in durations.items() if stats])
        result.extend(["", references(prefix, (("Figure C", "figures/figure_c_decode_opportunity.svg"),
                                              ("Idle window distribution", "figures/decode_idle_window_distribution.svg")))])
        result.extend(["", "## Arrival timing through the chain", "",
            "Elapsed completion gaps measure the actual arrival transformation. The stages are dependent and the cohorts can differ in model and request geometry.", ""])
        result += table(("Stream in the shared ROI", "Count", "p50 gap ms", "p95 gap ms", "Gap CV"), [
            (phase, row["events"], number(row["interevent_gap_ms"]["p50"], 1),
             number(row["interevent_gap_ms"]["p95"], 1), number(row["gap_CV"], 2))
            for phase, row in case["trace"]["arrival_transform"].items()])

    reads = [row for row in report["cohorts"] if row.get("native_NIXL_READ")]
    if reads:
        result.extend(["", "## Physical NIXL handoff readiness", "",
                       "The opt-in D observer matches each original P/H connector descriptor to the native async READ by its exact producer engine and request identity. "
                       "The post call entry and first DONE return bound possible READ activity. The last PROC entry bounds the interval still outstanding. "
                       "The read complement is reported only when every recorded successful request has a valid READ or an explicit destination-local path and capture reports zero drops. "
                       "This captures D-worker NIXL state in host monotonic time. Exclusive DMA utilization and other transports require additional observations. "
                       "These rows retain a diagnostic observer classification.", ""])
        result += table(("Epoch and cohort", "Native READ requests", "D-local no handoff", "Successful clients", "Whole-cohort accounted",
                         "Native bytes GiB", "Possible READ ROI", "Definite outstanding ROI",
                         "Proxy quiet outside accounted READs", "Kernel idle outside accounted READs",
                         "Kernel idle + proxy quiet outside accounted READs"), [
            (label(row), count.get("matched_native_READ_requests", 0), count.get("destination_local_without_native_READ", 0),
             count.get("successful_proxy_requests", 0), "yes" if data["whole_cohort_READ_accounted"] else "open",
             number(data["reported_native_READ_bytes"] / (1024**3)) if data["operations_with_telemetry"] else "—",
             percent(trace_fraction(row, "D_native_nixl_READ_possible")),
             percent(trace_fraction(row, "D_native_nixl_READ_definite_outstanding")),
             percent(trace_fraction(row, "D_proxy_request_free_and_no_accounted_native_READ")),
             percent(trace_fraction(row, "D_kernel_idle_outside_accounted_native_READ")),
             percent(trace_fraction(row, "D_kernel_idle_proxy_free_outside_accounted_native_READ")))
            for row in reads if (data := row["native_NIXL_READ"]) and (count := data["counts"])])
        if any(row["native_NIXL_READ"]["counts"].get("matched_native_READ_requests") for row in reads):
            result.extend(["", "The worker-ready timestamp is the return of D's original get-finished routine. "
                               "Request-level ready intervals may include NIXL state polling. Native telemetry duration is available separately.", ""])
            result += table(("Epoch and cohort", "Native transfer p50/p95 ms", "First post to worker ready p50/p95 ms",
                             "Latest DONE to worker ready median μs", "E/P busy, D idle outside accounted READs"), [
                (label(row), "/".join(number(data["native_transfer_duration_us"][field] / 1000, 3)
                                      if data["native_transfer_duration_us"][field] is not None else "—" for field in ("p50", "p95")),
                 "/".join(number(data["observed_worker_ready_after_first_post_us"][field] / 1000, 3)
                          if data["observed_worker_ready_after_first_post_us"][field] is not None else "—" for field in ("p50", "p95")),
                 number(data["observed_worker_ready_after_latest_DONE_us"]["p50"], 2),
                 percent(trace_fraction(row, "E_P_busy_D_kernel_idle_outside_accounted_native_READ")))
                for row in reads if (data := row["native_NIXL_READ"]) and data["counts"].get("matched_native_READ_requests")])
            result.extend(["", references(prefix, (("Native NIXL READ timeline", "figures/native_nixl_read_timeline.svg"),
                                                   ("Operation-level timestamps and telemetry", "native_nixl_READ_intervals.csv")))])

    replays = [row for row in report.get("arrival_replay", ()) if row.get("stream_gaps")]
    if replays:
        result.extend(["", "## Arrival-matched text control", "",
                       "This test reuses the observed E-ready offsets from a completed multimodal cohort with the same Language Model checkpoint. "
                       "Text requests enter a native 1P/1D NIXL path at those offsets. The test isolates arrival spacing with comparable prompt length, "
                       "while input content and the live encoder remain different."])
        for replay in replays:
            result.extend(["", f"{label(replay['source'])} → {label(replay['replay'])} · `{replay['evidence']}`. "
                           f"Largest scheduled offset error {number(replay.get('client_source_offsets_max_error_us'), 3)} μs. "
                           f"Largest difference between sorted input lengths "
                           f"{number(replay['max_sorted_input_token_count_difference'], 0)} token(s).", ""])
            result += table(("Full cohort stream", "Count", "Observed span rate /s", "Median gap ms", "p95 gap ms", "Gap CV"), [
                (phase, item["count"], number(item["finite_event_span_rate_per_second"], 3), number(item["gap_ms"]["p50"], 1),
                 number(item["gap_ms"]["p95"], 1), number(item["gap_CV"], 2))
                for phase, item in replay["stream_gaps"].items()])
            if clocks := replay.get("live_matched_stage_clocks"):
                result.extend(["", "Live stage activity in the central 60% of the matched E-ready and text replay arrival clocks.", ""])
                result += table(("Clock", "ROI s", "P submits/completes", "P-completed input tokens/s",
                                 "D proxy request-free", "P-submit to first client token p95 ms"), [
                    (arm, number(figure["duration_seconds"]), f"{figure['P_submits']}/{figure['P_completes']}",
                     number(figure["P_completion_input_tokens_per_second"], 0), percent(figure["D_proxy_request_free_fraction"]),
                     number(figure["P_submit_to_first_client_token_ms"]["p95"], 1))
                    for arm, figure in clocks.items() if figure])
            if peer := replay.get("nearest_text_Poisson"):
                result.extend(["", f"Nearest same-checkpoint text Poisson cohort {label(peer)} offered "
                               f"{number(peer['offered_qps'], 3)} req/s. Its offered rate differs by "
                               f"{percent(peer['relative_offered_rate_difference'])} from the recorded E-ready finite rate. "
                               "Its finite realized schedule is reported separately from its nominal rate."])
        result.extend(["", references(prefix, (("E-ready arrival replay figure", "figures/encoder_ready_arrival_replay.svg"),))])

    historical = report.get("historical_AR")
    if historical:
        trace = historical["trace"]
        fractions = trace["fraction_of_window"]
        result.extend(["", "## Historical AR context", "", historical["scope"], "",
            "The prior raw streams are re-binned over this study's central p20–p80 interval while keeping the published whole-cohort observer result. "
            f"The AR comparison gives P kernel busy {percent(fractions['prefill'])}, D kernel busy {percent(fractions['decode'])}, "
            f"P-busy and D-kernel-idle overlap {percent(fractions['P_busy_D_kernel_idle'])}, "
            f"and D kernel idle and proxy request free {percent(fractions['D_kernel_idle_proxy_request_free'])}. "
            "This supplies a descriptive reference. Model-specific or modality-specific explanations require a controlled comparison. "
            "[Published AR baseline](https://github.com/jyssys/MLLM-EP/blob/5bdd6ef5e91840e37c776c2ad937866531e6a391/poc_flashvep/reports/pd_prefill_heavy_resource_motivation.md)"])
    return result


def compare_assistance(report, prefix):
    probes = report.get("matched_E_D_interventions", [])
    result = ["# Matched encoder and decode assistance", ""]
    if not probes:
        return result + ["Paired E and D work was unavailable in the archived runs. `NOT_RUN`."]
    result.extend([
        "The two intervention arms select the same request indices. E routes the selected requests through the extra LM worker colocated with vision encoding and hands off its KV through NIXL. "
        "D runs the selected prefills on the decode worker and begins decode from its local KV. The same-runtime control keeps independent P and D and uses the same routed HTTP gateway. "
        "The E LM helper entails extra model weight residency. The D gateway holds pending decode requests during the local prefill to first generated token interval. "
        "The HTTP intervals include scheduling, stage dispatch and protocol overhead. Native NIXL counters identify connector traffic with the reported operation counts. "
        "Three arms execute sequentially. Repeated independent starts and task-quality comparisons would be required for an efficacy claim.", "",
    ])
    for probe in probes:
        routes = probe["routes"]
        comparison = probe.get("same_runtime_control_exact_generated_sequences")
        count = routes["probe_d"]["serving"]["planned"]
        result.extend([f"## N{count} paired arms", "",
                       f"Observed matched helper slots {len(probe['matched_assisted_slots'])}/{len(probe['expected_assisted_slots'])}. "
                       f"{probe['E_vs_D_exact_generated_sequences']}/{count} greedy sequences agree exactly E versus D. "
                       f"Evidence `{probe['evidence']}`.", ""])
        if comparison:
            result.extend([f"For the same-runtime no-assist arm, exact agreement is "
                           f"{comparison['probe_e']}/{count} for E assist and {comparison['probe_d']}/{count} for D-local.", ""])
        result += table(("Arm", "Completed", "Assisted", "Assisted LM input tokens", "E HTTP p95 ms",
                         "TTFT p95 ms", "TPOT p95 ms", "ITL p99 ms", "Decode SLO", "Joint goodput req/s",
                         "Non-assisted Decode SLO", "NIXL GiB (ops)"), [
            (label(row), f"{row['serving']['completed']}/{count}", row["completed_assisted"],
             number(row["completed_assisted_prompt_tokens"], 0), number(row["E_HTTP_ms"]["p95"], 1),
             number(row["serving"]["TTFT_p95_ms"], 1), number(row["serving"]["TPOT_p95_ms"]),
             number(row["serving"]["ITL_p99_ms"]), f"{row['serving']['absolute_decode_pass']}/{count}",
             number(row["serving"]["joint_goodput_req_s"], 3),
             f"{row['other_requests_decode_SLO_pass']}/{row['other_requests']}",
             f"{number(nixl['vllm:nixl_bytes_transferred_sum'] / (1024**3))} ({nixl['vllm:nixl_bytes_transferred_count']:.0f})"
             if (nixl := row["native_NIXL_D_counter_delta"]) and
             {"vllm:nixl_bytes_transferred_sum", "vllm:nixl_bytes_transferred_count"} <= nixl.keys() else "—")
            for row in routes.values()])
        result.extend(["", f"E-helper P-stage HTTP median {number(routes['probe_e']['H_HTTP_ms']['p50'], 1)} ms. "
                       f"D-local submit to first output median {number(routes['probe_d']['D_local_prefill_and_first_token_HTTP_ms']['p50'], 1)} ms. "
                       f"D proxy reported HTTP-empty admission for {routes['probe_d']['D_proxy_empty_on_local_admission']}/"
                       f"{routes['probe_d']['completed_assisted']} assisted requests.", ""])
        if "ondemand" in routes:
            native = routes["ondemand"]["serving"]
            strict_fields = ("TPOT_p95_ms", "ITL_p99_ms")
            result += table(("Relative to same-runtime control", "TTFT p95", "TPOT p95", "ITL p99", "Joint goodput",
                             "Strict +5% decode tail protection"), [
                (name, *(percent(arm[key] / native[key] - 1) if native[key] and arm[key] is not None else "—"
                         for key in ("TTFT_p95_ms", "TPOT_p95_ms", "ITL_p99_ms", "joint_goodput_req_s")),
                 "—" if any(arm[key] is None or native[key] is None for key in strict_fields)
                 else "Pass" if all(arm[key] <= 1.05 * native[key] for key in strict_fields) else "Fail")
                for name, arm in (("E helper", routes["probe_e"]["serving"]), ("D-local", routes["probe_d"]["serving"]))])
            result.append("")
    if any(probe["matched_assisted_work"] for probe in probes):
        result.append(references(prefix, (("Matched outcomes figure", "figures/encoder_vs_decode_matched_assistance.svg"),)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--link-prefix", default="../mllm_epd_prefill_heavy_modality_discovery/results/region")
    parser.add_argument("--include-assistance", action="store_true", help="Also write the paired E vs D assistance report")
    args = parser.parse_args()
    data = read_json(args.reduction)
    reports = [("prefill_heavy_operating_region", operating_region),
               ("encoder_prefill_joint_saturation", joint_saturation),
               ("decode_idle_fragmentation", fragmentation)]
    if args.include_assistance:
        reports.append(("encoder_vs_decode_assistance", compare_assistance))
    for key, render in reports:
        path = args.reports / f"mllm_epd_{key}.md"
        path.write_text("\n".join(render(data, args.link_prefix)) + "\n")


if __name__ == "__main__":
    main()
