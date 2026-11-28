"""Plot qualified and diagnostic EPD measurements with separate evidence markers."""

import argparse
from collections import defaultdict
from pathlib import Path

from .timeline import ROLES, read_json


ABBR = {"encoder": "E", "prefill": "P", "decode": "D"}
COLOR = {"encoder": "#315f9a", "prefill": "#ba5d27", "decode": "#398579"}


def render(report, out):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "mllm_epd_prefill_heavy_modality_discovery"
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    out.mkdir(exist_ok=True, parents=True)
    cases = [row for row in report["cohorts"] if "serving" in row and row["serving"]["complete"]]
    profiled = [row for row in cases if (row.get("trace") or {}).get("captures")
                and row["metadata"].get("route", "native") == "native"]
    selected = sorted(profiled, key=lambda row: (not row.get("M2_pass"), not row.get("M1_pass"),
        row["trace"]["evidence"] != "TRACE_ANALYSIS",
        -row["serving"]["completed"], row["epoch"], row["tag"]))
    names = []

    def save(fig, name, *, rect=None):
        path = out / (name + ".svg")
        if fig.get_layout_engine() is None:
            fig.tight_layout(rect=rect)
        fig.savefig(path, metadata={"Date": None}, bbox_inches="tight")
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        names.append(name)
        plt.close(fig)

    def source(row):
        return row["metadata"].get("cohort_id") or row["metadata"].get("source") or row["tag"]

    def measured(row, role):
        return (row.get("trace") or {}).get("fraction_of_window", {}).get(role)

    pairs = [(row, event) for row in cases if row["metadata"].get("route", "native") == "native"
             and row["tag"] == row["metadata"].get("group", row["tag"]) for event in row["request_event_rows"]]
    if pairs:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.7))
        for ax, role, event, x in zip(axs, ROLES, ("encoder_http_ms", "prefill_http_ms", "decode_http_ms"),
                                      ("image_count", "input_tokens", "output_tokens")):
            data = [(req[x], req[event]) for _, req in pairs if req.get(x) is not None and req.get(event, 0) > 0]
            if data:
                ax.scatter(*zip(*data), color=COLOR[role], alpha=.35, s=12, linewidths=0)
                ax.set_yscale("log")
            ax.set(xlabel=x.replace("_", " "), ylabel="Elapsed ms", title=ABBR[role] + " HTTP enclosure")
        fig.suptitle("Online stage HTTP elapsed time includes queueing, dispatch and transfers")
        save(fig, "stage_service_geometry")

    capacity = report["capacity"]
    if capacity:
        fig, ax = plt.subplots(figsize=(max(7, len(capacity) * 1.65), 3.8))
        for slot, role in enumerate(ROLES):
            points = [(i + (slot - 1) * .24, max(x["requests_per_second"] for x in rows if x["stage"] == role))
                      for i, rows in enumerate(capacity.values()) if any(x["stage"] == role for x in rows)]
            if points:
                ax.bar(*zip(*points), width=.22, label=role, color=COLOR[role])
        ax.set(xticks=list(range(len(capacity))), xticklabels=list(capacity), ylabel="Completed requests/s",
               title="Independent E/P/D finite service witnesses")
        ax.legend(frameon=False)
        save(fig, "independent_stage_capacity")

    gpm_rows = [row for row in cases if row.get("gpm") and row["metadata"].get("observer", "clean") == "clean"
                and row["metadata"].get("route", "native") in ("native", "text")
                and not row.get("auxiliary_observers")]
    if gpm_rows:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, role in zip(axs, ROLES):
            groups = defaultdict(list)
            for row in gpm_rows:
                sm = row["gpm"]["by_role"].get(role, {}).get("SM_UTIL")
                if sm is not None and row.get("offered_qps") is not None:
                    groups[source(row)].append((row["offered_qps"], sm))
            for label, values in sorted(groups.items()):
                ax.plot(*zip(*sorted(values)), "o", ms=4, alpha=.75, label=label)
            ax.set(xlabel="Offered requests/s", ylim=(0, 100), title=role)
        axs[0].set_ylabel("Window-weighted NVML SM activity (%)")
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=7, frameon=False)
        fig.suptitle("Clean EPD survey counters")
        save(fig, "sm_activity_vs_offered_load", rect=(0, .045 * ((len(labels) + 4) // 5) + .02, 1, 1))

        controlled = [row for row in gpm_rows if row["metadata"].get("source") == "controlled" and row["offered_qps"] is not None
                      and row["serving"]["geometry"]["visual_ratio"]["p50"] is not None]
        if controlled:
            fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True, layout="constrained")
            rates = [row["offered_qps"] for row in controlled]
            levels = Normalize(min(rates), max(rates))
            mesh = None
            for ax, role in zip(axs, ROLES):
                data = [(100 * row["serving"]["geometry"]["visual_ratio"]["p50"],
                         row["gpm"]["by_role"].get(role, {}).get("SM_UTIL"), row["offered_qps"])
                        for row in controlled]
                points = [(a, b, c) for a, b, c in data if b is not None]
                if points:
                    x, y, colors = zip(*points)
                    mesh = ax.scatter(x, y, c=colors, cmap="viridis", norm=levels, s=27, alpha=.80)
                ax.set(xlabel="Median visual token share (%)", ylim=(0, 100), title=role)
            axs[0].set_ylabel("Window-weighted NVML SM activity (%)")
            if mesh is not None:
                cbar = fig.colorbar(mesh, ax=list(axs.flat), fraction=.045, pad=.045)
                cbar.set_label("Offered requests/s")
            fig.suptitle("Visual ratio and device activity in the controlled input-length sweep")
            save(fig, "controlled_visual_ratio_sm_activity")

    if profiled:
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
        for ax, role in zip(axs, ROLES):
            for row in profiled:
                y = measured(row, role)
                if y is not None and row.get("offered_qps") is not None:
                    verified = row["trace"]["evidence"] == "TRACE_ANALYSIS"
                    ax.scatter(row["offered_qps"], 100 * y, s=42, c=COLOR[role], marker="o" if verified else "x")
            ax.set(xlabel="Offered requests/s", ylim=(0, 100), title=role)
        axs[0].set_ylabel("Physical kernel-busy interval union (%)")
        fig.suptitle("CUPTI against offered load   ○ qualified   × diagnostic")
        save(fig, "kernel_busy_vs_offered_load")

        ratio_points = [row for row in profiled if row["serving"]["geometry"]["visual_ratio"]["p50"] is not None]
        if ratio_points:
            fig, ax = plt.subplots(figsize=(7.4, 4.2))
            for row in ratio_points:
                v = row["serving"]["geometry"]["visual_ratio"]["p50"]
                for role in ROLES:
                    y = measured(row, role)
                    if y is not None:
                        ax.scatter(100 * v, 100 * y, color=COLOR[role], s=38,
                                   marker="o" if row["trace"]["evidence"] == "TRACE_ANALYSIS" else "x")
            for role in ROLES:
                ax.plot([], [], "o", color=COLOR[role], label=role)
            ax.set(xlabel="Median visual token share (%)", ylabel="CUPTI kernel-busy (%)",
                   title="Cross-workload visual ratio   ○ qualified   × diagnostic")
            ax.legend(frameon=False)
            save(fig, "visual_ratio_kernel_activity")

        cols = sorted({(source(row), row["serving"]["geometry"]["input_tokens_manifest"]["p50"] or 0)
                       for row in profiled}, key=lambda point: (point[1], point[0]))
        loads = sorted({row["offered_qps"] for row in profiled})
        grid = defaultdict(list)
        for row in selected:
            grid[row["offered_qps"], (source(row),
                 row["serving"]["geometry"]["input_tokens_manifest"]["p50"] or 0)].append(row)
        fig, ax = plt.subplots(figsize=(max(7, 2.15 * len(cols)), 1.5 + .95 * len(loads)))
        for i, load in enumerate(loads):
            for j, case in enumerate(cols):
                repetitions = grid[load, case]
                if not repetitions:
                    note, color = "No capture", "#f2f2f2"
                else:
                    row = repetitions[0]
                    evidence = row["trace"]["evidence"]
                    busy = ["—" if measured(row, role) is None else f"{100 * measured(row, role):.0f}"
                            for role in ROLES]
                    metrics = row["serving"]
                    gate = "M2 pass" if row["M2_pass"] else "M1 pass" if row["M1_pass"] else "M1 open"
                    note = (f"{' / '.join(busy)}%\n{row['observed_region']} · {evidence}\n"
                            f"{gate} · D SLO {metrics['absolute_decode_pass']}/{metrics['planned']}")
                    if len(repetitions) > 1:
                        note += f" · {len(repetitions)} starts"
                    color = "#d9ead9" if row["M1_pass"] else "#e7ddd0" if evidence == "DIAGNOSTIC" else "#dfe8ee"
                ax.add_patch(Rectangle((j - .49, i - .48), .98, .96, facecolor=color, edgecolor="white"))
                ax.text(j, i, note, va="center", ha="center", fontsize=8)
        ax.set(xlim=(-.5, len(cols) - .5), ylim=(-.5, len(loads) - .5), xticks=range(len(cols)),
               xticklabels=[f"{name}\n{tokens:,.0f} tokens" for name, tokens in cols], yticks=range(len(loads)),
               yticklabels=[f"{qps:g}" for qps in loads], xlabel="Cohort and median LM input length",
               ylabel="Offered requests/s", title="EPD operating region   E / P / D kernel busy")
        save(fig, "figure_a_operating_region")

    if selected:
        case = selected[0]
        trace = case["trace"]
        series = trace["bins"]
        window_s = trace["duration_seconds"]
        fig, (ax, queue_ax, request_ax) = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True,
                                                      gridspec_kw={"height_ratios": [1, .7, 2.1]})
        for role in ROLES:
            if role in series:
                values = series[role]
                x = [i * window_s / len(values) for i in range(len(values))]
                ax.step(x, values, where="post", label=ABBR[role], color=COLOR[role], linewidth=1.1)
        for key, label, color in (("E_busy_P_busy_D_kernel_idle", "E ∩ P ∩ D kernel idle", "#8c62a1"),
                                  ("prefill_known_compute_ready_waiting", "Sampled P compute-ready", "#cc3e3a"),
                                  ("prefill_plain_waiting", "Sampled P WAITING", "#cc3e3a"),
                                  ("decode_CUPTI_copy", "D CUPTI copies", "#333333"),
                                  ("D_native_nixl_READ_possible", "D NIXL possible READ", "#7c432e")):
            if key in series and (key != "prefill_known_compute_ready_waiting" or any(series[key])):
                values = series[key]
                x = [i * window_s / len(values) for i in range(len(values))]
                ax.step(x, values, where="post", color=color, linestyle="--", linewidth=1, label=label)
        ax.set(ylabel="Fraction of bin", ylim=(0, 1.05))
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        queue = (case.get("queue") or {}).get("samples", [])
        for role, field, label, color in (
            ("prefill", "vllm:num_requests_waiting_by_reason:capacity", "P capacity waiting", COLOR["prefill"]),
            ("prefill", "vllm:num_requests_waiting_by_reason:deferred", "P deferred", "#9d617c"),
            ("decode", "vllm:num_requests_waiting_by_reason:capacity", "D capacity waiting", COLOR["decode"]),
        ):
            points = [(item["seconds_from_roi"], item["values"][role][field]) for item in queue
                      if field in item["values"].get(role, {})]
            if points:
                queue_ax.plot(*zip(*points), ".", markersize=2.5, label=label, color=color)
        queue_ax.set(ylabel="Queue count")
        if queue_ax.lines:
            queue_ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        else:
            queue_ax.text(.5, .5, "No queue-reason snapshots in this window", transform=queue_ax.transAxes,
                          ha="center", fontsize=8)
        events = case["request_event_rows"]
        left, right = trace["window_monotonic_s"]
        events = [event for event in events if left <= event["arrival"] < right]
        events = sorted(events, key=lambda event: event["arrival"])
        every = max(1, (len(events) + 29) // 30)
        for index, event in enumerate(events[::every]):
            for role, (begin, end) in zip(ROLES, (("E_submit", "E_complete"), ("P_submit", "P_ready"),
                                                 ("D_submit", "D_http_complete"))):
                if begin in event and end in event:
                    a, b = max(0, event[begin] - left), min(window_s, event[end] - left)
                    if b > a:
                        request_ax.barh(index, b - a, left=a, color=COLOR[role], height=.72)
            if "E_renderer_processor_handoff_begin" in event:
                a = max(0, event["E_renderer_processor_handoff_begin"] - left)
                b = min(window_s, event["E_renderer_processor_handoff_end"] - left)
                if b > a:
                    request_ax.barh(index, b - a, left=a, color="#263650", height=.28)
            for begin, end in event.get("D_native_nixl_READ_possible", ()):
                a, b = max(0, begin - left), min(window_s, end - left)
                if b > a:
                    request_ax.barh(index, b - a, left=a, color="#7c432e", height=.25)
            for time, marker in (("E_complete", "^"), ("P_ready", "|")):
                if time in event and 0 <= event[time] - left < window_s:
                    request_ax.scatter(event[time] - left, index, marker=marker, color="#242424", s=13, zorder=4)
        request_ax.set(xlim=(0, window_s), xlabel="Monotonic seconds from common p20 arrival", ylabel="Requests sampled by index")
        request_ax.set_title("HTTP enclosures   E renderer in dark blue   ▲ whole E ready   │ P ready", fontsize=9)
        fig.suptitle(f"Common E/P/D timeline   {case['tag']}   {trace['evidence']}")
        save(fig, "figure_b_joint_timeline")

        if trace["kernel_idle_histogram"]:
            labels = [f"{r['lower_ms']}–{r['upper_ms']}" if r["upper_ms"] is not None else f"≥{r['lower_ms']}"
                      for r in trace["kernel_idle_histogram"]]
            fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
            for label, data, offset, color in (("D kernel idle", trace["kernel_idle_histogram"], -.18, COLOR["decode"]),
                                               ("Also proxy request free", trace["proxy_quiet_idle_histogram"], .18, "#6376a5")):
                if data:
                    centers = [i + offset for i in range(len(data))]
                    axs[0].bar(centers, [row["count"] for row in data], width=.35, color=color, label=label)
                    axs[1].bar(centers, [row["seconds"] for row in data], width=.35, color=color, label=label)
            for ax in axs:
                ax.set(xticks=range(len(labels)), xticklabels=labels, xlabel="Consecutive idle window size (ms)")
                ax.tick_params(axis="x", labelsize=8, rotation=30)
            axs[0].set_ylabel("Window count")
            axs[1].set_ylabel("Time in bucket (s)")
            axs[0].set_yscale("symlog", linthresh=1)
            axs[1].legend(frameon=False, fontsize=8)
            fig.suptitle(f"D idle fragmentation   {case['tag']}   {trace['evidence']}")
            save(fig, "decode_idle_window_distribution")

            opportunities = (("D kernel idle", "D_kernel_idle"),
                             ("Idle ∩ proxy request free", "D_kernel_idle_proxy_request_free"),
                             ("Idle ∩ recent empty engine sample", "D_kernel_idle_recent_idle_snapshot"),
                             ("Idle ∩ no recorded CUPTI copy", "D_kernel_idle_CUPTI_copy_free"),
                             ("Idle ∩ outside accounted NIXL READ", "D_kernel_idle_outside_accounted_native_READ"),
                             ("All native transport-free idle", None), ("Deadline-safe idle", None),
                             ("Native idle mapped to candidate execution", None))
            fits = trace.get("retrospective_quantum_fit")
            executed = [row for row in cases if row["metadata"].get("route") in ("naive", "naive_isolated", "cost", "cost_blind")
                        and (row.get("route_accounting") or {}).get("D_local_completed")]
            assisted = max(executed, key=lambda row: (source(row) == source(case),
                row["offered_qps"] == case["offered_qps"],
                row["metadata"].get("d_budget") == case["metadata"].get("d_budget"),
                row["metadata"].get("output") == case["metadata"].get("output"), row["serving"]["planned"],
                row["route_accounting"]["D_local_completed"]), default=None)
            height_ratios = (3.5, .9) if assisted else (3.5,)
            width_ratios = (2.1, 1) if fits else (1,)
            fig = plt.figure(figsize=(12 if fits else 8.7, 4.75 if assisted else 3.9))
            layout = fig.add_gridspec(len(height_ratios), len(width_ratios),
                                      width_ratios=width_ratios, height_ratios=height_ratios)
            ax = fig.add_subplot(layout[0, 0])
            rendered = trace["fraction_of_window"]
            for i, (name, key) in enumerate(opportunities):
                if key in rendered:
                    value = 100 * rendered[key]
                    ax.barh(i, value, color=COLOR["decode"] if i == 0 else "#6376a5", height=.59)
                    ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=8)
                else:
                    ax.text(1, i, "NOT MEASURED", va="center", color="#676767", fontsize=8)
            ax.set(xlim=(0, 110), yticks=range(len(opportunities)),
                   yticklabels=[label for label, _ in opportunities], xlabel="Fraction of shared ROI (%)",
                   title=f"D-side useful opportunity   {trace['evidence']}")
            ax.invert_yaxis()
            if fits:
                oracle = fig.add_subplot(layout[0, 1])
                oracle.plot([item["required_ms"] for item in fits],
                            [item["fit_windows"] for item in fits], "o--", color="#8c62a1", label="Proxy-quiet D")
                if all(item["fit_windows_with_recent_P_ready_possible_launch"] is not None for item in fits):
                    oracle.plot([item["required_ms"] for item in fits],
                                [item["fit_windows_with_recent_P_ready_possible_launch"] for item in fits],
                                "s:", color="#52749c", label="P-ready sample allows a start")
                oracle.legend(frameon=False, fontsize=7)
                oracle.set(xlabel="Hypothetical work quantum (ms)", ylabel="Observed intervals with enough span",
                           title="Offline fit bound", xscale="log")
            if assisted:
                actual = fig.add_subplot(layout[1, :])
                actual.axis("off")
                work = assisted["route_accounting"]
                duration = work["D_local_prefill_to_first_token_HTTP_ms"]["p50"]
                token_total = work["D_local_total_LM_input_tokens"]
                local_tokens = f"{token_total:,} prompt tokens" if token_total is not None else "prompt token count unavailable"
                first_token = f"   ·   D submit→first token p50 {duration:,.0f} ms" if duration is not None else ""
                actual.text(0, .86, "Completed D-local work in a separate online route arm   MEASURED_ONLINE",
                            transform=actual.transAxes, fontsize=10, fontweight="bold", va="top")
                actual.text(0, .47, f"{assisted['epoch']}/{assisted['tag']}    QPS {assisted['offered_qps']:g}    "
                            f"D budget {assisted['metadata'].get('d_budget', 'unrecorded')}    "
                            f"{work['D_local_completed']}/{assisted['serving']['planned']} local prefills    {local_tokens}"
                            f"{first_token}", transform=actual.transAxes, fontsize=9, va="top")
            save(fig, "figure_c_decode_opportunity")

    deflection_routes = {"native", "ondemand", "naive", "naive_isolated", "cost", "cost_blind", "force"}
    deflections = [row for row in cases if row["metadata"].get("route", "native") in deflection_routes
                   and any(event["D_local"] for event in row["request_event_rows"])]
    if deflections:
        cohorts = {source(row) for row in deflections}
        relevant = [row for row in cases if source(row) in cohorts and row["metadata"].get("observer", "clean") == "clean"
                    and row["metadata"].get("route", "native") in deflection_routes]
        groups = defaultdict(list)
        for row in relevant:
            meta = row["metadata"]
            budget = meta.get("d_budget")
            label = meta.get("route", "native") + (f" (D B{budget})" if budget else " (D budget unrecorded)")
            groups[label].append(row)
        fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
        for ax, (field, label) in zip(axs.flat, (("TTFT_p95_ms", "TTFT p95 (ms)"),
                ("TPOT_p95_ms", "TPOT p95 (ms)"), ("ITL_p99_ms", "ITL p99 (ms)"),
                ("joint_goodput_req_s", "Joint SLO goodput (req/s)"))):
            for route, rows in sorted(groups.items()):
                values = [(row["offered_qps"], row["serving"][field]) for row in rows if row["serving"][field] is not None]
                if values:
                    ax.scatter(*zip(*values), label=route, s=22)
            ax.set(xlabel="Offered requests/s", ylabel=label)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8, frameon=False)
        save(fig, "whole_request_deflection_tradeoff", rect=(0, 0, 1, 1 - .046 * ((len(labels) + 3) // 4)))

    replay_data = [row for row in report.get("arrival_replay", []) if row.get("stream_gaps")]
    if replay_data:
        row = max(replay_data, key=lambda record: (record["evidence"] == "MEASURED_ONLINE",
                                                   record["stream_gaps"]["replay_client_arrival"]["count"]))
        values = row["stream_gaps"]
        fig, (ax, cv) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [2, 1]})
        masks = (("Client arrived", "source_client_arrival"), ("E ready", "source_E_complete"),
                 ("Multimodal P submitted", "source_P_submit"), ("Multimodal P ready", "source_P_ready"),
                 ("Text replay arrived", "replay_client_arrival"),
                 ("Text replay P submitted", "replay_P_submit"), ("Text replay P ready", "replay_P_ready"),
                 ("Nearest-rate text Poisson", "nearest_text_Poisson_arrival"))
        plotted = [(label, values[key]) for label, key in masks if key in values and values[key]["count"] > 1]
        starts = [r["gap_ms"]["p50"] for _, r in plotted]
        ends = [r["gap_ms"]["p95"] - s for (_, r), s in zip(plotted, starts)]
        ax.barh(range(len(plotted)), starts, color="#6e94bb", label="p50")
        ax.barh(range(len(plotted)), ends, left=starts, color="#b6c7de", label="p50 to p95")
        ax.set(yticks=range(len(plotted)), yticklabels=[label for label, _ in plotted], xlabel="Inter-event gap (ms)")
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=8)
        cv.barh(range(len(plotted)), [r["gap_CV"] for _, r in plotted], color="#78a393")
        cv.set(yticks=[], xlabel="Gap coefficient of variation")
        cv.set_ylim(ax.get_ylim())
        fig.suptitle(f"E-ready arrival replay on the text-only backbone   {row['evidence']}")
        save(fig, "encoder_ready_arrival_replay")

    probes = [entry for entry in report.get("matched_E_D_interventions", []) if entry["matched_assisted_work"]]
    if probes:
        entry = max(probes, key=lambda row: (row["evidence"] == "MEASURED_ONLINE",
                                             row["routes"]["probe_d"]["serving"]["planned"]))
        arms = entry["routes"]
        names_by_route = {"ondemand": "Same-runtime control", "probe_e": "E helper", "probe_d": "D-local"}
        shades = {"ondemand": "#8995a1", "probe_e": COLOR["encoder"], "probe_d": COLOR["decode"]}
        fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
        for ax, (key, title) in zip(axs.flat, (("E_HTTP_p95_ms", "E HTTP p95 (ms)"),
                ("TTFT_p95_ms", "TTFT p95 (ms)"), ("ITL_p99_ms", "ITL p99 (ms)"),
                ("TPOT_p95_ms", "TPOT p95 (ms)"), ("joint_goodput_req_s", "Joint SLO goodput (req/s)"),
                ("NIXL_GiB", "D NIXL reported GiB"))):
            for position, (role, data) in enumerate(arms.items()):
                if key == "E_HTTP_p95_ms":
                    value = data["E_HTTP_ms"]["p95"]
                elif key == "NIXL_GiB":
                    delta = (data["native_NIXL_D_counter_delta"] or {}).get("vllm:nixl_bytes_transferred_sum")
                    value = delta / (1024 ** 3) if delta is not None else None
                else:
                    value = data["serving"].get(key)
                if value is not None:
                    ax.bar(position, value, color=shades[role], width=.6)
                    ax.annotate(f"{value:.2f}", (position, value), xytext=(0, 5), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)
            ax.set(xlim=(-.65, len(arms) - .35), xticks=range(len(arms)),
                   xticklabels=[names_by_route[route] for route in arms], title=title)
            ax.tick_params(axis="x", labelrotation=12, labelsize=8)
            ax.margins(y=.18)
        fig.suptitle(f"Matched assisted work slots   {len(entry['matched_assisted_slots'])} per route   "
                     f"N{arms['probe_d']['serving']['planned']}   {entry['evidence']}")
        save(fig, "encoder_vs_decode_matched_assistance")

    image_witnesses = [case for case in cases if (case.get("image_readiness") or {}).get("requests")]
    if image_witnesses:
        case = max(image_witnesses, key=lambda row: row["image_readiness"]["multimodal_complete_requests"])
        requests = case["image_readiness"]["requests"]
        preferred = [row for row in requests if row["images"] > 1] or requests
        fig, (ax, full) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for i, record in enumerate(preferred):
            for plot, prefix, color in ((ax, "first_image_before_all_ms", "#7a517b"),
                                        (full, "full_request_before_E_ack_ms", COLOR["encoder"])):
                lo, hi = record[prefix + "_lower"], record[prefix + "_upper"]
                plot.hlines(i, lo, hi, linewidth=1.5, color=color)
                plot.plot([lo, hi], [i, i], "|", color=color, markersize=6)
        ax.set(xlabel="First prompt image lead over all images (ms)", ylabel="Joined request index")
        full.set(xlabel="All images ready before E HTTP acknowledgment (ms)")
        ax.set_xscale("symlog", linthresh=10)
        full.set_xscale("symlog", linthresh=10)
        full.axvline(0, color="#555555", linestyle=":", lw=1)
        fig.suptitle("Native CUDA IPC publish and GPU-event readiness bounds   DIAGNOSTIC")
        save(fig, "native_per_image_readiness_bounds")

    nixl_witnesses = [case for case in cases if case.get("trace")
                      and (case.get("native_NIXL_READ") or {}).get("counts", {}).get("matched_native_READ_requests")]
    if nixl_witnesses:
        case = max(nixl_witnesses, key=lambda row: (row["native_NIXL_READ"]["whole_cohort_READ_accounted"],
                                                    row["native_NIXL_READ"]["counts"]["matched_native_READ_requests"]))
        trace = case["trace"]
        left, right = trace["window_monotonic_s"]
        events = sorted((row for row in case["request_event_rows"] if row.get("D_native_nixl_READ_possible")
                         and left <= row["D_submit"] <= right), key=lambda row: row["D_submit"])
        fig, (ax, request_ax) = plt.subplots(2, 1, figsize=(9.5, 4.9), sharex=True,
                                           gridspec_kw={"height_ratios": [1, 3]})
        for key, role, color in (("D_native_nixl_READ_possible", "Possible READ outstanding", "#a55e45"),
                                 ("D_native_nixl_READ_definite_outstanding", "Definite outstanding", "#4f2d2d"),
                                 ("decode", "D GPU kernels", COLOR["decode"])):
            if values := trace["bins"].get(key):
                x = [i * (right - left) / len(values) for i in range(len(values))]
                ax.step(x, values, where="post", linewidth=1.05, label=role, color=color)
        ax.set(ylabel="Fraction of bin", ylim=(0, 1.04))
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        for index, row in enumerate(events):
            a, b = row["D_submit"] - left, min(right, row["D_http_complete"]) - left
            request_ax.barh(index, b - a, left=a, color="#a0b7b0", height=.66)
            for begin, end in row["D_native_nixl_READ_possible"]:
                a, b = begin - left, min(right, end) - left
                if b >= a:
                    request_ax.barh(index, b - a, left=a, color="#a55e45", height=.34)
            ready = row["observed_D_worker_KV_ready"] - left
            if ready <= right - left:
                request_ax.scatter(ready, index, s=14, marker="|", color="#232323")
        request_ax.set(xlim=(0, right - left), xlabel="Monotonic seconds from common p20 arrival",
                       ylabel="Decode submissions", title="Native HTTP in green   possible READ in rose   │ D KV-ready")
        fig.suptitle(f"Native NIXL READ bounds   {case['tag']}   DIAGNOSTIC")
        save(fig, "native_nixl_read_timeline")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("Rendered", ", ".join(render(read_json(args.reduction), args.out)))


if __name__ == "__main__":
    main()
