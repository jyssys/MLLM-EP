"""Reproduce H200 wide-decode-cap online tables, heatmaps, and idle CDFs."""

import argparse
from collections import Counter, defaultdict
from functools import cached_property
import json
from pathlib import Path
from statistics import mean

from ..mllm_epd_h200_prefill_heavy_batch_matched import analysis as shared
from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import complement, elapsed, intersect, merged, scheduler_intervals


CAPS = ("1", "2", "4", "8", "16", "32", "64", "free")
NATIVE_LOADS = (2, 3, 4, 5, 6, 8)
SMALL_CAP_LOADS = (2, 4, 5, 6)
ROLES = ("encoder", "prefill", "decode")
CLEAN_OBSERVATIONS = {"clean", "batch_clean_before", "batch_clean_after", "clean_before", "clean_after"}
CONTEXT = ("output_mode", "Main100_cycles", "burner_condition", "client_raw_policy", "subset")
QUIET_MS = (160, 225)
OUTCOMES = ("TTFT_p95_ms", "ITL_p99_ms", "TPOT_p95_ms", "D_absolute_SLO_attainment",
            "D_queue_mean", "D_max_prepared_decode_requests")


def burner_condition(run):
    lease = run / "state/lease.json"
    before = run / "state/burners_before.json"
    if not lease.is_file() or not before.is_file():
        return "UNKNOWN"
    details = shared.read(lease)
    occupants = details["foreign_gpu_occupancy"]
    known = {(str(index), str(pid)) for index, pid in shared.read(before)}
    indices = {row[1]: str(row[0]) for row in details["gpu_inventory"]}
    verified = sum((indices.get(row[0]), str(row[1])) in known for row in occupants)
    if not occupants:
        return "OFF"
    if verified == len(occupants):
        return f"ON_{verified}"
    return f"ON_{verified}_WITH_OTHER"


class WideCohort(shared.Cohort):
    @property
    def cap(self):
        workload = self.config["workload"]
        return str(workload.get("decode_cap", "1" if workload.get("batch_one") else "free"))

    @property
    def cycles(self):
        return self.config["workload"].get("Main100_cycles", 1)

    @cached_property
    def burner(self):
        return burner_condition(self.run)

    @property
    def method(self):
        return f"{self.config['workload']['method']}-{'B' + self.cap if self.cap != 'free' else 'free'}"

    @property
    def base(self):
        return {**super().base, "decode_cap": self.cap, "Main100_cycles": self.cycles,
                "burner_condition": self.burner,
                "client_raw_policy": "DEFERRED" if self.config["defer_client_full_raw"] else "INLINE",
                "subset": self.config["workload"].get("subset", "all"),
                "method_family": self.config["workload"]["method"]}

    @cached_property
    def match(self):
        return (*super().match, self.cycles, self.burner, self.config["defer_client_full_raw"],
                self.base["subset"], json.dumps(self.config.get("output_count"), sort_keys=True),
                self.config["workload"].get("paired_start_id"))


def group(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def weighted_quantile(depth, fraction):
    ordered = sorted((count, end - begin) for begin, end, count in depth)
    target = sum(duration for _, duration in ordered) * fraction
    cumulative = 0.
    for count, duration in ordered:
        cumulative += duration
        if cumulative >= target:
            return count
    return None


def coord_depth(cohort, main_requests):
    left, right = shared.cutoff(cohort.client_rows)
    active, queued = [], []
    blockers = Counter()
    enqueued = 0
    selected = isolated = audited = 0
    cap = None if cohort.cap == "free" else int(cohort.cap)
    for row in main_requests:
        raw = cohort.proxy_rows.get(row["runtime_request_id"])
        if raw is None:
            continue
        buckets = shared.stage_intervals(raw)
        if "D HTTP" in buckets:
            active.append(buckets["D HTTP"])
        if "P ready to D submit" in buckets:
            queued.append(buckets["P ready to D submit"])
        event = raw.get("D_admission", {})
        ready = raw.get("D_at_P_ready", {})
        why = event.get("blocked_on_enqueue", ())
        blockers.update(why)
        enqueued += "blocked_on_enqueue" in event
        selected += bool(raw["token_enabled"])
        if raw["token_enabled"] and "active_http_requests" in event.get("before", {}):
            audited += 1
            isolated += event["before"]["active_http_requests"] == 0
        first_client = row["scheduled_arrival_monotonic"] + row["TTFT_ms"] / 1000
        d_submit = raw.get("D_submit")
        row.update(P_ready_monotonic=raw.get("P_ready"), D_submit_monotonic=d_submit,
                   D_first_client_token_monotonic=first_client,
                   D_first_generated_token_observed_monotonic=raw.get("D_first_generated_token"),
                   D_submit_to_first_client_token_ms=(first_client - d_submit) * 1000
                   if d_submit is not None else None,
                   realized_prefix_tokens=raw.get("token_partition", {}).get("prefix_tokens"),
                   D_cap_binding_at_P_ready=ready.get("cap_binding"),
                   D_admission_enqueue_blockers=",".join(why),
                   D_HTTP_at_admit=event.get("after", {}).get("active_http_requests"),
                   D_suffix_isolated=event.get("suffix_isolated"))
    complete = len(active) == len(cohort.client_rows) and len(queued) == len(cohort.client_rows)
    if not complete:
        return {"HTTP_depth_status": "NOT_MEASURED", "suffix_selected": selected if cohort.proxy_rows else None}
    d_http = list(shared.depth_series(active, left, right))
    d_queue = list(shared.depth_series(queued, left, right))
    full = list(shared.depth_series(active, min(a for a, _ in active), max(b for _, b in active)))
    duration = right - left
    if cap is not None and any(count > cap for _, _, count in full):
        raise ValueError(f"{cohort.run.name}/{cohort.tag} exceeds its claimed coordinator cap")
    return {
        "HTTP_depth_status": "MEASURED_ONLINE",
        "D_HTTP_max_all": max(count for _, _, count in full),
        "D_HTTP_max_central": max(count for _, _, count in d_http),
        "D_HTTP_mean_central": sum(count * (b - a) for a, b, count in d_http) / duration,
        "D_HTTP_cap_binding_fraction_time":
            sum(b - a for a, b, count in d_http if count == cap) / duration if cap is not None else None,
        "D_HTTP_cap_reached": any(count == cap for _, _, count in full) if cap is not None else None,
        "D_queue_mean": sum(count * (b - a) for a, b, count in d_queue) / duration,
        "D_queue_p95": weighted_quantile(d_queue, .95),
        "D_queue_max": max(count for _, _, count in d_queue),
        "D_queue_nonempty_fraction": sum(b - a for a, b, count in d_queue if count > 0) / duration,
        "D_cap_at_enqueue_fraction": blockers["CAP_FULL"] / enqueued if enqueued else None,
        "D_fifo_at_enqueue_fraction": blockers["FIFO_AHEAD"] / enqueued if enqueued else None,
        "D_suffix_wait_at_enqueue_fraction": blockers["SUFFIX_NEEDS_EMPTY_D"] / enqueued if enqueued else None,
        "D_suffix_owner_at_enqueue_fraction": blockers["SUFFIX_PREFILL_IN_PROGRESS"] / enqueued if enqueued else None,
        "D_enqueue_observations": enqueued,
        "suffix_selected": selected, "suffix_isolation_observations": audited,
        "suffix_zero_active_D": isolated,
    }


def geometry_summary(cohort, histogram):
    prepared = [r for r in histogram
                if r["series"] == "prepared_decode_padding_histogram" and r["counter_valid"]]
    if not prepared:
        return {"batch_evidence": "NOT_MEASURED", "attains_cap": None}
    calls = Counter()
    for row in prepared:
        if row["decode_requests"]:
            calls[row["decode_requests"]] += row["calls"]
    if not calls:
        return {"batch_evidence": "NOT_MEASURED", "attains_cap": None}
    count = sum(calls.values())
    cap = None if cohort.cap == "free" else int(cohort.cap)
    return {
        "batch_evidence": cohort.evidence,
        "decode_forward_calls": count,
        "D_prepared_decode_mode": min(calls, key=lambda n: (-calls[n], n)),
        "D_prepared_decode_mean": sum(n * c for n, c in calls.items()) / count,
        "D_prepared_decode_max": max(calls),
        "D_prepared_single_request_fraction": calls[1] / count,
        "D_prepared_decode_call_histogram": json.dumps(calls, sort_keys=True),
        "attains_cap": (max(calls) == cap) if cap is not None else None,
        "D_prepared_graph_modes": ",".join(sorted({row["graph_mode"] for row in prepared})),
    }


def profile_rows(cohort, tables):
    if cohort.observation != "profile":
        return
    tag = cohort.tag.removesuffix("_profile")
    path = cohort.run / "analysis" / f"{tag}_trace.json"
    if not path.is_file():
        return
    trace = shared.read(path)
    evidence = trace["evidence"]
    for role, row in trace.get("scheduler_samples", {}).items():
        tables["scheduler"].append({**cohort.base, "evidence": evidence, "role": role, **row})

    left, right = trace["window_monotonic"]
    p_busy = trace["kernel_intervals"]["prefill"]
    observed_decode_idle = complement(trace["kernel_intervals"]["decode"], left, right)
    d_active, p_ready_wait = [], []
    for raw in cohort.proxy_rows.values():
        intervals = shared.stage_intervals(raw)
        if "D HTTP" in intervals:
            d_active.append(intervals["D HTTP"])
        if "P ready to D submit" in intervals:
            p_ready_wait.append(intervals["P ready to D submit"])
    if len(d_active) != len(cohort.client_rows) or len(p_ready_wait) != len(cohort.client_rows):
        tables["missing"].append({**cohort.base, "artifact": "quiet_windows", "status": "NOT_MEASURED",
                                  "reason": "Incomplete D HTTP or P-ready wait receipt coverage"})
        return
    d_available_http = complement(merged(d_active + p_ready_wait, left, right), left, right)
    quiet = intersect(observed_decode_idle, d_available_http)
    for begin, end in quiet:
        tables["quiet_windows"].append({**cohort.base, "evidence": evidence,
                                         "begin_seconds": begin - left, "end_seconds": end - left,
                                         "duration_ms": (end - begin) * 1000,
                                         "boundary_censored": begin <= left or end >= right})
    profile_dir = cohort.run / "raw/profiles" / cohort.tag / "prefill"
    p_waiting, scheduler = scheduler_intervals(profile_dir, left, right)
    p_busy_and_waiting = intersect(p_busy, p_waiting)
    p_busy_time = elapsed(p_busy)
    for threshold in QUIET_MS:
        qualifying = [(a, b) for a, b in quiet if (b - a) * 1000 >= threshold]
        overlapping = [window for window in qualifying if intersect([window], p_busy_and_waiting)]
        overlap_time = elapsed(intersect(overlapping, p_busy_and_waiting))
        tables["quiet_overlap"].append({
            **cohort.base, "evidence": evidence, "minimum_window_ms": threshold,
            "quiet_windows": len(qualifying), "overlapping_windows": len(overlapping),
            "overlap_seconds": overlap_time, "fraction_central": overlap_time / (right - left),
            "fraction_P_busy": overlap_time / p_busy_time if p_busy_time else None,
            "P_scheduler_sample_coverage": scheduler["window_fraction_with_recent_sample"],
            "scope": "Observed physical D quiet and no proxy D HTTP or P-ready wait, intersected with "
                     "P kernel activity and recent sampled engine waiting. Central p20–p80 only.",
        })


def nvml_rows(cohort, tables):
    path = cohort.run / "raw" / f"{cohort.tag}_nvml.json"
    if not path.is_file():
        return {}
    details = shared.read(path)
    left, right = shared.cutoff(cohort.client_rows)
    central = [row for row in details["rows"] if left <= row["monotonic"] < right]
    sampler_valid = details["sampler_drained"] and not details["errors"]
    metrics = ("gpu_utilization_percent", "memory_utilization_percent", "memory_used_bytes",
               "temperature_c", "power_w", "sm_clock_mhz")
    output = {}
    for role in ROLES:
        readings = [row["roles"][role] for row in central if role in row["roles"]]
        if not readings:
            continue
        means = {f"{key}_mean": mean(row[key] for row in readings) for key in metrics}
        tables["nvml_utilization"].append({
            **cohort.base, "evidence": "DIAGNOSTIC", "role": role,
            "central_samples": len(readings), "interval_seconds": details["interval_seconds"],
            "sampler_valid": sampler_valid, **means,
            "gpu_utilization_percent_p95": shared.percentile([r["gpu_utilization_percent"] for r in readings], .95),
            "scope": "NVML sample-window utilization and telemetry, distinct from CUPTI kernel activity.",
        })
        if sampler_valid:
            output[f"{role}_nvml_gpu_utilization_percent_mean"] = means["gpu_utilization_percent_mean"]
    return output


def capacity_rows(campaign):
    result = []
    for path in sorted(campaign.glob("*/analysis/capacity_decode_wide_*_r*.json")):
        raw = shared.read(path)
        if "maximum_actual_D_geometry" not in raw:
            continue
        row = {k: v for k, v in raw.items() if isinstance(v, (str, int, bool, float)) or v is None}
        row.update({f"actual_{key}": value for key, value in raw["maximum_actual_D_geometry"].items()})
        row.update(start=path.parent.parent.name, burner_condition=burner_condition(path.parent.parent))
        result.append(row)
    return result


def clean_start_mean(rows, metric):
    by_start = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            by_start[row["start"]].append(row[metric])
    return mean(mean(samples) for samples in by_start.values()) if by_start else None


def grid_attempts(campaign):
    attempts = {}
    for path in sorted(campaign.glob("*/state/environment.json")):
        run = path.parent.parent
        env = shared.read(path)
        argv = env.get("argv", [])
        if "--mode" not in argv or argv[argv.index("--mode") + 1] != "native-common" or "--qps" not in argv:
            continue
        loads = []
        for arg in argv[argv.index("--qps") + 1:]:
            if arg.startswith("--"):
                break
            loads.append(float(arg))
        cap = argv[argv.index("--decode-cap") + 1] if "--decode-cap" in argv else "free"
        cycles = int(argv[argv.index("--repeat-main100") + 1]) if "--repeat-main100" in argv else 1
        mode = "Per-request equal-output replay" if "--replay" in argv else "Natural EOS"
        client_raw = "DEFERRED" if "--defer-client-raw" in argv else "INLINE"
        subset = argv[argv.index("--subset") + 1] if "--subset" in argv else "all"
        failed = (run / "state/failure.json").exists()
        observed_cfgs = list((run / "configs").glob("native_common*_arrival.json"))
        tried_loads = {shared.read(p)["QPS"] for p in observed_cfgs}
        for load in loads:
            key = (mode, cycles, burner_condition(run), client_raw, subset, cap, load)
            if failed and (load in tried_loads or not tried_loads and load == loads[0]):
                attempts[key] = "FAILED_EXECUTION"
            elif key not in attempts:
                attempts[key] = "NOT_MEASURED" if load in tried_loads else "NOT_RUN"
    return attempts


def planned_cell(cap, load):
    if cap in ("4", "8", "16", "32", "64", "free") and load in NATIVE_LOADS:
        return "PRESPECIFIED"
    if cap in ("1", "2") and load in SMALL_CAP_LOADS:
        return "PRESPECIFIED"
    return "OPTIONAL_SCREEN"


def native_grid(campaign, cohorts, kernel_busy):
    keys = (*CONTEXT, "decode_cap", "QPS")
    native = [r for r in cohorts if r["method_family"] == "native-common"]
    grouped = group(native, keys)
    kernels = group([r for r in kernel_busy if r["method_family"] == "native-common"], keys)
    attempts = grid_attempts(campaign)
    contexts = {key[:-2] for key in {*grouped, *attempts}}
    if not contexts:
        contexts.add(("Natural EOS", 1, "UNKNOWN", "UNKNOWN", "all"))
    rows = []
    for context in sorted(contexts):
        cells = {(key[-2], key[-1]) for key in {*grouped, *attempts} if key[:-2] == context}
        full_natural = context[:2] == ("Natural EOS", 1) and context[-1] == "all"
        if full_natural:
            cells |= {(cap, qps) for cap in CAPS for qps in NATIVE_LOADS}
        for cap, qps in sorted(cells, key=lambda pair: (CAPS.index(pair[0]), pair[1])):
            key = (*context, cap, qps)
            all_rows = grouped.get(key, [])
            clean = [row for row in all_rows if row["observation"] in CLEAN_OBSERVATIONS]
            counters = [row for row in all_rows if row["batch_evidence"] != "NOT_MEASURED"]
            qualified_counters = [row for row in counters if row["batch_evidence"] == "TRACE_ANALYSIS"]
            selected = qualified_counters or counters
            kernel = kernels.get(key, [])
            q_kernel = [row for row in kernel if row["evidence"] == "TRACE_ANALYSIS"]
            report_kernel = q_kernel or kernel
            calls = Counter()
            for summary in selected:
                calls.update({int(n): count for n, count in
                              json.loads(summary["D_prepared_decode_call_histogram"]).items()})
            row = dict(zip(keys, key, strict=True))
            row.update(
                design=planned_cell(cap, qps) if full_natural else "SEPARATE_DIAGNOSTIC",
                latency_status="MEASURED_ONLINE" if clean else attempts.get(key, "NOT_RUN"),
                independent_starts=len({item["start"] for item in clean}),
                clean_cohorts=len(clean), batch_cohorts=len(counters),
                batch_evidence="TRACE_ANALYSIS" if qualified_counters else "DIAGNOSTIC" if counters else "NOT_MEASURED",
                kernel_evidence="TRACE_ANALYSIS" if q_kernel else "DIAGNOSTIC" if kernel else "NOT_MEASURED",
                D_max_scheduled_requests=max((r.get("D_max_scheduled_requests", 0) for r in selected), default=None),
                D_max_prepared_decode_requests=max((r["D_prepared_decode_max"] for r in selected), default=None),
                D_prepared_decode_mode=min(calls, key=lambda n: (-calls[n], n)) if calls else None,
                D_decode_forward_calls=sum(r["decode_forward_calls"] for r in selected) if selected else None,
                attains_cap=(any(r["attains_cap"] for r in selected) if selected and cap != "free" else None),
            )
            for metric in ("TTFT_p95_ms", "TTFT_p50_ms", "ITL_p99_ms", "TPOT_p95_ms", "D_absolute_SLO_attainment",
                           "D_queue_mean", "SLO_goodput", "completed_requests_per_second", "D_HTTP_cap_binding_fraction_time"):
                row[metric] = clean_start_mean(clean, metric)
            for role in ROLES:
                values = [item["busy_fraction"] for item in report_kernel if item["role"] == role]
                row[f"{role}_kernel_busy_fraction"] = mean(values) if values else None
                key_nvml = f"{role}_nvml_gpu_utilization_percent_mean"
                row[key_nvml] = clean_start_mean(clean, key_nvml)
            row["nvml_evidence"] = "DIAGNOSTIC" if any(
                row[f"{role}_nvml_gpu_utilization_percent_mean"] is not None for role in ROLES) else "NOT_MEASURED"
            rows.append(row)
    return rows


def stratified_intervals(contrasts):
    rows = []
    fields = ("Main100_cycles", "burner_condition", "client_raw_policy", "subset")
    for key, members in group(contrasts, fields).items():
        stratum = dict(zip(fields, key, strict=True))
        rows.extend({**stratum, **row} for row in shared.paired_intervals(members))
    return rows


def draw_figures(output, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=180)
        fig.savefig(output / (name + ".pdf"))
        plt.close(fig)

    def context_name(mode, cycles, burner, raw_policy, subset, extra=""):
        workload = "natural" if mode == "Natural EOS" else "replay"
        return f"{workload}_{cycles}x_{burner.lower()}_{raw_policy.lower()}_{subset}{extra}"

    def cap_text(cap):
        return "Free" if cap == "free" else f"B{cap}"

    def heatmap(fig, ax, rows, field, label, status_field, scale):
        loads = sorted({r["QPS"] for r in rows})
        caps = [cap for cap in CAPS if any(r["decode_cap"] == cap for r in rows)]
        keyed = {(r["decode_cap"], r["QPS"]): r for r in rows}
        data = np.full((len(caps), len(loads)), np.nan)
        for i, cap in enumerate(caps):
            for j, load in enumerate(loads):
                value = keyed.get((cap, load), {}).get(field)
                if value is not None:
                    data[i, j] = value * scale
        good = ~np.isnan(data)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#ececec")
        image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto")
        if good.any():
            fig.colorbar(image, ax=ax, fraction=.05, pad=.04)
        for i, cap in enumerate(caps):
            for j, load in enumerate(loads):
                row = keyed.get((cap, load), {})
                status = row.get(status_field, "NOT_RUN")
                if good[i, j]:
                    suffix = "*" if status == "DIAGNOSTIC" else ""
                    text = f"{data[i, j]:.3g}{suffix}"
                    color = "white" if image.norm(data[i, j]) < .55 else "black"
                else:
                    if status_field != "latency_status" and row.get("latency_status") == "NOT_RUN":
                        status = "NOT_RUN"
                    text = {"FAILED_EXECUTION": "FAIL", "NOT_RUN": "NR"}.get(status, "NM")
                    color = "#444444"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
        ax.set(xticks=range(len(loads)), xticklabels=[f"{q:g}" for q in loads],
               yticks=range(len(caps)), yticklabels=[cap_text(cap) for cap in caps],
               xlabel="Offered QPS", title=label)

    context = CONTEXT
    titles = ("TTFT p95 (ms)", "ITL p99 (ms)", "TPOT p95 (ms)", "Absolute decode SLO (%)",
              "Mean D coordinator queue", "Maximum prepared decode requests")
    for key, rows in sorted(group(tables["native_grid"], context).items()):
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 7))
        for ax, metric, title in zip(axes.flat, OUTCOMES, titles, strict=True):
            observed = metric == "D_max_prepared_decode_requests"
            heatmap(fig, ax, rows, metric, title, "batch_evidence" if observed else "latency_status",
                    100 if metric == "D_absolute_SLO_attainment" else 1)
        fig.suptitle(f"NativeCommon · {key[0]} · {key[1]}× Main100 · burners {key[2]}\n"
                     "NR not run   NM not measured   FAIL failed execution   * diagnostic")
        save(fig, "native_cap_qps_heatmap_" + context_name(*key))

        if any(row["kernel_evidence"] != "NOT_MEASURED" for row in rows):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
            for ax, role in zip(axes, ROLES, strict=True):
                heatmap(fig, ax, rows, f"{role}_kernel_busy_fraction",
                        f"{role.title()} kernel busy (%)", "kernel_evidence", 100)
            fig.suptitle(f"NativeCommon CUPTI · {key[0]} · {key[1]}× Main100 · burners {key[2]}\n"
                         "NR not run   NM not measured   * observer gate unqualified")
            save(fig, "kernel_busy_cap_qps_" + context_name(*key))
        if any(row["nvml_evidence"] != "NOT_MEASURED" for row in rows):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
            for ax, role in zip(axes, ROLES, strict=True):
                heatmap(fig, ax, rows, f"{role}_nvml_gpu_utilization_percent_mean",
                        f"{role.title()} mean NVML utilization (%)", "nvml_evidence", 1)
            fig.suptitle(f"NVML telemetry · {key[0]} · {key[1]}× Main100 · burners {key[2]}\n"
                         "NR not run   NM not measured   * diagnostic")
            save(fig, "nvml_utilization_cap_qps_" + context_name(*key))

    def context_at_load(key):
        return context_name(*key[:-1], extra=f"_q{key[-1]:g}")

    for key, rows in sorted(group(tables["quiet_windows"], (*context, "QPS")).items()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, series in group(rows, ("start", "cohort")).items():
            durations = sorted(r["duration_ms"] for r in series if not r["boundary_censored"])
            if durations:
                first = series[0]
                caption = f"{first['method']} · {first['start']} · {first['evidence']}"
                ax.step(durations, [(i + 1) / len(durations) for i in range(len(durations))],
                        where="post", label=caption)
        if ax.lines:
            for threshold in QUIET_MS:
                ax.axvline(threshold, color="#555555", linestyle=":", linewidth=.9)
            ax.set(xscale="log", xlabel="Observed quiet window (ms)", ylabel="Empirical CDF",
                   title="CUPTI D idle with no active HTTP or queued P-ready request")
            ax.legend(fontsize=7)
            save(fig, "D_observed_quiet_window_CDF_" + context_at_load(key))
        else:
            plt.close(fig)

    actual = [r for r in tables["actual_d_batch"] if r["series"] == "prepared_decode_padding_histogram"
              and r["counter_valid"] and r["decode_requests"] > 0]
    for key, rows in sorted(group(actual, (*context, "QPS")).items()):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for _, arm in group(rows, ("start", "cohort")).items():
            calls = Counter()
            for row in arm:
                calls[row["decode_requests"]] += row["calls"]
            x = sorted(calls)
            first = arm[0]
            ax.plot(x, [calls[n] / sum(calls.values()) for n in x], marker="o",
                    label=f"{first['method']} · {first['start']} · {first['evidence']}")
        ax.set(xlabel="Actual prepared decode requests", ylabel="Share of decode forwards")
        ax.legend(fontsize=7)
        save(fig, "decode_prepared_distribution_" + context_at_load(key))

    clean = [r for r in tables["requests"] if r["observation"] == "clean"]
    for key, rows in sorted(group(clean, (*context, "QPS")).items()):
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for _, arm in group(rows, ("start", "cohort")).items():
            values = sorted(row["TTFT_ms"] for row in arm)
            ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post",
                    label=f"{arm[0]['method']} · {arm[0]['start']}")
        ax.axvline(shared.SLO["TTFT_ms"], color="#555555", linestyle=":", linewidth=.9)
        ax.set(xlabel="TTFT (ms)", ylabel="Request CDF", title=f"{key[0]} · {key[1]}× Main100 · q{key[-1]:g}")
        ax.legend(fontsize=7)
        save(fig, "ttft_CDF_" + context_at_load(key))

    serving = [r for r in tables["cohorts"] if r["observation"] in CLEAN_OBSERVATIONS]
    for key, rows in sorted(group(serving, context).items()):
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for (method,), arm in group(rows, ("method",)).items():
            ax.scatter([row["TTFT_p95_ms"] for row in arm], [row["ITL_p99_ms"] for row in arm],
                       label=method, s=27)
        ax.set(xlabel="Client TTFT p95 (ms)", ylabel="Client ITL p99 (ms)",
               title=f"{key[0]} · {key[-1]} · burners {key[2]}")
        ax.legend(fontsize=7)
        save(fig, "TTFT_ITL_tradeoff_" + context_name(*key))

    selected = [r for r in serving if r["split_only_P_ready_D_observations"] > 0]
    for key, rows in sorted(group(selected, (*context, "QPS")).items()):
        fig, ax = plt.subplots(figsize=(max(7, .65 * len(rows)), 4.5))
        quiet = [r["split_only_P_ready_D_HTTP_inactive_no_pending_fraction"] for r in rows]
        idle = [r["split_only_P_ready_D_HTTP_inactive_fraction"] for r in rows]
        waiting = [i - q for i, q in zip(idle, quiet, strict=True)]
        positions = range(len(rows))
        ax.bar(positions, quiet, label="No active or pending D HTTP")
        ax.bar(positions, waiting, bottom=quiet, label="Pending D admission only")
        ax.bar(positions, [1 - i for i in idle], bottom=idle, label="At least one active D HTTP")
        labels = [f"{r['method']}\n{r['start']} · {r['observation']}" for r in rows]
        ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1), ylabel="Share of selected requests",
               title="Coordinator D state at observed P prefix completion")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.legend(fontsize=7)
        save(fig, "selected_prefix_ready_D_state_" + context_at_load(key))

    profiled = group(tables["kernel_timeline"], (*context, "QPS", "decode_cap"))
    summaries = {(r["start"], r["cohort"]): r for r in tables["cohorts"]}
    for key, rows in sorted(profiled.items()):
        # Prefer the lexicographically first finalized qualified start, then
        # show the central four seconds independently of kernel occupancy.
        epochs = group(rows, ("start", "cohort"))
        epoch = min(epochs)
        middle = summaries[epoch]["central_window_s"] / 2
        left, right = max(0., middle - 2), min(middle * 2, middle + 2)
        labels = (*ROLES, "P kernel and D physical idle")
        fig, ax = plt.subplots(figsize=(8.6, 3.5))
        for i, role in enumerate(labels):
            spans = merged([(r["begin_seconds"], r["end_seconds"])
                            for r in epochs[epoch] if r["series"] == role], left, right)
            ax.broken_barh([(a, b - a) for a, b in spans], (i - .3, .6))
        ax.set(yticks=range(len(labels)), yticklabels=labels, xlim=(left, right),
               xlabel="Seconds since central arrival window began",
               title=f"Qualified CUPTI activity · {epoch[0]} / {epoch[1]}")
        save(fig, "qualified_kernel_timeline_" + context_at_load(key[:-1]) + "_" + cap_text(key[-1]))

    capacity = tables["wide_decode_capacity"]
    if capacity:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for (burners,), rows in group(capacity, ("burner_condition",)).items():
            q = lambda cap, key: mean(row[key] for row in rows if str(row["cap"]) == cap)
            caps = [cap for cap in CAPS if any(str(row["cap"]) == cap for row in rows)]
            for ax, metric, label in zip(axes, ("requests_per_second", "actual_prepared_decode_requests"),
                                         ("Completion rate (req/s)", "Maximum prepared decode requests"),
                                         strict=True):
                ax.plot([cap_text(cap) for cap in caps], [q(cap, metric) for cap in caps],
                        marker="o", label=burners)
                ax.set(xlabel="Configured D HTTP cap", ylabel=label)
        axes[0].legend(fontsize=7)
        fig.suptitle("Independent D with actual prepared NIXL KV")
        save(fig, "independent_decode_capacity")


def write_outputs(tables, output):
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        shared.csv_file(output / f"{name}.csv", rows, minimum_fields=("start",) if not rows else ())


def aggregate(campaign, output, figures=True):
    cohorts, excluded = shared.cohort_records(campaign, WideCohort)
    tables = {name: [] for name in (
        "cohorts", "requests", "slo_breakdown", "host_timeline", "decode_idle_windows",
        "kernel_busy", "kernel_overlap", "kernel_timeline", "actual_d_batch", "critical_path",
        "decode_cap", "scheduler", "quiet_windows", "quiet_overlap", "missing", "nvml_utilization",
    )}
    for cohort in cohorts:
        host = shared.request_table(cohort, tables)
        current = tables["requests"][-len(cohort.client_rows):]
        for row, request in zip(current, cohort.client_rows, strict=True):
            row.update(runtime_request_id=request["runtime_request_id"], source=request["source"],
                       stratum=request["stratum"], image_count=request["image_count"],
                       scheduled_arrival_monotonic=request["scheduled_arrival"])
        depth = coord_depth(cohort, current)
        start = len(tables["actual_d_batch"])
        maxima = shared.batch_table(cohort, tables)
        geometry = geometry_summary(cohort, tables["actual_d_batch"][start:])
        caps = {**cohort.base, **depth, **maxima, **geometry}
        tables["decode_cap"].append(caps)
        shared.kernel_tables(cohort, tables)
        profile_rows(cohort, tables)
        telemetry = nvml_rows(cohort, tables)
        tables["cohorts"].append({
            **caps, "duration_seconds": cohort.metrics["duration_seconds"],
            **{metric: cohort.metrics.get(metric) for metric in shared.METRICS},
            **cohort.metrics["gate_attainment"], **host, **telemetry,
        })
        for artifact, exists in (("D_worker_counter", bool(maxima)), ("kernel_trace", cohort.observation == "profile"
                and (cohort.run / "analysis" / f"{cohort.tag.removesuffix('_profile')}_trace.json").is_file())):
            if not exists:
                tables["missing"].append({**cohort.base, "artifact": artifact, "status": "NOT_MEASURED"})
        if depth.get("suffix_selected", 0):
            for name in ("worker_suffix_kernel_cost", "batch_reformation_timed_after_suffix"):
                tables["missing"].append({**cohort.base, "artifact": name, "status": "NOT_MEASURED",
                    "reason": "Host boundaries and aggregate batch histograms do not isolate this worker interval"})
    tables["execution_failures"] = excluded
    tables["observer_qualification"] = shared.observer_qualification(campaign)
    tables["wide_decode_capacity"] = capacity_rows(campaign)
    tables["stage_capacity"] = shared.capacity_table(campaign)
    tables["service_proximity"] = shared.capacity_proximity(cohorts, tables["cohorts"], tables["stage_capacity"])
    contrasts = [row for row in shared.paired_contrasts(cohorts)
                 if row["baseline"].startswith("native-") and row["method_family"].startswith(("static", "idle"))]
    tables["same_cap_contrasts"] = [row for row in contrasts if row["baseline"].rsplit("-", 1)[1]
                                    == row["method"].rsplit("-", 1)[1]]
    tables["free_native_contrasts"] = [row for row in contrasts if row["baseline"].endswith("-free")]
    tables["contrast_ci"] = stratified_intervals(
        tables["same_cap_contrasts"] + [r for r in tables["free_native_contrasts"] if r["decode_cap"] != "free"])
    tables["native_grid"] = native_grid(campaign, tables["cohorts"], tables["kernel_busy"])
    write_outputs(tables, output)
    if figures:
        draw_figures(output, tables)
    receipt = {
        "source": str(campaign.resolve()), "cohorts": len(cohorts),
        "table_rows": {key: len(rows) for key, rows in tables.items()},
        "missing_cells": {"NR": "not run", "NM": "not measured", "FAIL": "failed execution"},
        "central_window": "Scheduled arrival p20–p80 for each complete frozen Main100 or sustained cohort.",
        "decode_cap": "Configured HTTP cap, its measured occupancy, and engine-forward geometry are separate.",
        "scope": "CUPTI quiet-window intersections are observations; coordinator no-HTTP is a host proxy.",
        "comparison": "Unique equal-arrival pairs and separate burner, output, repetition, observer and client write policies.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.runs, args.output, not args.no_figures), indent=2))


if __name__ == "__main__":
    main()
