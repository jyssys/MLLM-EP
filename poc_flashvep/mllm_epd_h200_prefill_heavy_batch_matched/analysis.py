"""Generate H200 Main100 evidence tables and figures from completed cohorts."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
from statistics import mean

from .trace import complement, elapsed, intersect, merged


SLO = {"TTFT_ms": 5000, "ITL_ms": 34, "TPOT_ms": 33}
METRICS = (
    "TTFT_mean_ms", "TTFT_p50_ms", "TTFT_p95_ms", "ITL_mean_ms", "ITL_p99_ms",
    "TPOT_p95_ms", "total_input_output_tokens_per_second", "raw_output_tokens_per_second",
    "completed_requests_per_second", "SLO_goodput", "total_output_tokens",
)
ROLES = ("encoder", "prefill", "decode")
PHASE_SUFFIXES = ("batch_clean_before", "batch_clean_after", "clean_before", "profile", "clean_after")
CRITICAL_PATH = ("Arrival to proxy", "Proxy to E submit", "E HTTP", "E complete to P submit",
                 "P HTTP", "P ready to D submit", "D submit to first client token")


def read(path):
    return json.loads(path.read_text())


def jsonl(path, appending=False):
    with path.open() as stream:
        return [json.loads(line) for line in stream if not appending or line.endswith("\n")]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(values, proportion):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * proportion
    left = int(rank)
    return values[left] + (rank - left) * (values[min(left + 1, len(values) - 1)] - values[left])


def csv_file(path, rows, minimum_fields=()):
    fields = list(dict.fromkeys((*minimum_fields, *(field for row in rows for field in row))))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cutoff(client_rows):
    arrivals = sorted(row["scheduled_arrival"] for row in client_rows)
    return arrivals[len(arrivals) // 5], arrivals[len(arrivals) * 4 // 5]


@dataclass(frozen=True)
class Cohort:
    run: Path
    tag: str
    config: dict
    environment: dict
    metrics: dict
    client_rows: list
    proxy_rows: dict
    observation: str
    evidence: str

    @property
    def method(self):
        parts = self.config["workload"]
        return f"{parts['method']}-{'B1' if parts['batch_one'] else 'free'}"

    @property
    def base(self):
        return {
            "start": self.run.name, "cohort": self.tag, "method": self.method,
            "QPS": self.config["QPS"], "seed": self.config["seed"],
            "output_mode": self.config["output_mode"], "observation": self.observation,
            "evidence": self.evidence,
        }

    @property
    def match(self):
        return (
            self.config["QPS"], self.config["output_mode"], self.config["seed"],
            self.config["arrival_SHA256"], self.config["workload"]["manifest_SHA256"],
            self.environment["receipt_SHA256"],
            tuple(sorted(self.environment["authorized_uuid_by_gpu"].items())),
            self.environment["runtime"]["vllm_wheel_sha256"],
            self.observation,
        )


def cohort_records(campaign):
    cohorts, excluded = [], []
    for run in sorted(campaign.iterdir()):
        if not (run / "state/environment.json").is_file():
            continue
        environment = read(run / "state/environment.json")
        failure = run / "state/failure.json"
        if failure.exists():
            excluded.append({"start": run.name, "cohort": "", "evidence": "FAILED_EXECUTION",
                             "reason": read(failure).get("message", "")})
        proxy_path = run / "raw/proxy_epd.jsonl"
        # An active start can append logs during analysis. Consider finalized
        # cohorts whose raw client evidence can be validated independently.
        metrics_paths = list(sorted((run / "analysis").glob("*_metrics.json")))
        if not metrics_paths:
            continue
        proxy_rows = jsonl(proxy_path, appending=True) if proxy_path.exists() else []
        proxy_lookup = {row["runtime_request_id"]: row for row in proxy_rows}
        for metrics_path in metrics_paths:
            metrics = read(metrics_path)
            tag = metrics.get("run")
            if not tag or metrics.get("planned_count") != 100:
                continue
            config = read(run / "configs" / f"{tag}.json")
            if config.get("serial") or "method" not in config.get("workload", {}):
                continue
            completion = read(run / "state" / f"{tag}_complete.json")
            raw_path = run / "raw" / f"{tag}_requests.jsonl"
            raw_digest = digest(raw_path)
            arrival_path = run / "configs" / f"{tag}_arrival.json"
            if (not metrics["primary_execution_valid"] or not completion["passed"]
                    or metrics["completed_count"] != 100
                    or not raw_digest == metrics["raw_SHA256"] == completion["raw_SHA256"]
                    or digest(arrival_path) != config["arrival_SHA256"]):
                excluded.append({"start": run.name, "cohort": tag, "evidence": "FAILED_EXECUTION",
                                 "reason": "Incomplete cohort or raw client/arrival hash mismatch"})
                continue
            client_rows = jsonl(raw_path)
            if (len(client_rows) != 100 or len({row["request_id"] for row in client_rows}) != 100
                    or any(row["status"] != "MEASURED_ONLINE" for row in client_rows)):
                raise ValueError(f"Invalid Main100 roster in {run.name}/{tag}")
            arrival = read(arrival_path)["trace"]
            by_index = {row["index"]: row for row in client_rows}
            if len(arrival) != 100 or any(
                t["request_id"] != by_index[t["index"]]["request_id"]
                or abs(by_index[t["index"]]["scheduled_arrival"] -
                       completion["monotonic_start"] - t["offset_seconds"]) > 5e-6
                for t in arrival
            ):
                raise ValueError(f"Client arrivals do not match the frozen trace in {run.name}/{tag}")
            aligned = {row["runtime_request_id"]: proxy_lookup[row["runtime_request_id"]]
                       for row in client_rows if row["runtime_request_id"] in proxy_lookup
                       and proxy_lookup[row["runtime_request_id"]]["status"] == "MEASURED_ONLINE"}
            batch_observed = (run / "analysis" / f"{tag}_batch.json").exists()
            cpc_phase = next((phase for phase in PHASE_SUFFIXES if tag.endswith("_" + phase)), None)
            observation = cpc_phase or ("batch_observed" if batch_observed else "clean")
            evidence = "DIAGNOSTIC" if batch_observed or cpc_phase == "profile" else "MEASURED_ONLINE"
            if cpc_phase == "profile":
                gate_path = run / "analysis" / f"{tag.removesuffix('_profile')}_profile_gate.json"
                if gate_path.exists() and read(gate_path)["passed"]:
                    evidence = "TRACE_ANALYSIS"
            if batch_observed and tag.endswith("_observed"):
                gate_path = run / "analysis" / f"{tag.removesuffix('_observed')}_batch_qualification.json"
                if gate_path.exists() and read(gate_path)["passed"]:
                    evidence = "TRACE_ANALYSIS"
            cohorts.append(Cohort(run, tag, config, environment, metrics, client_rows,
                                  aligned, observation, evidence))
    return cohorts, excluded


def add_window(tables, cohort, series, spans, left, quantity=None):
    for begin, end in spans:
        tables.append({**cohort.base, "series": series,
                       "begin_seconds": begin - left, "end_seconds": end - left,
                       "duration_seconds": end - begin, "quantity": quantity})


def depth_series(intervals, left, right):
    events = Counter({left: 0, right: 0})
    for begin, end in intervals:
        begin, end = max(left, begin), min(right, end)
        if begin < end:
            events[begin] += 1
            events[end] -= 1
    depth = 0
    times = sorted(events)
    for begin, end in zip(times, times[1:]):
        depth += events[begin]
        yield begin, end, depth


def stage_intervals(row):
    time_pairs = {
        "E HTTP": ("E_submit", "E_complete"),
        "E complete to P submit": ("E_complete", "P_submit"),
        "P HTTP": ("P_submit", "P_ready"),
        "P ready to D submit": ("P_ready", "D_submit"),
        "D HTTP": ("D_submit", "D_http_complete" if "D_http_complete" in row else "finished"),
        "D submit to first SSE": ("D_submit", "D_first_SSE_bytes"),
        "D post-first-token HTTP": ("D_first_generated_token",
                                   "D_http_complete" if "D_http_complete" in row else "finished"),
    }
    return {name: (row[a], row[b]) for name, (a, b) in time_pairs.items()
            if a in row and b in row}


def critical_path(raw, proxy):
    timestamps = (raw["scheduled_arrival"], proxy["received"], proxy["E_submit"],
                  proxy["E_complete"], proxy["P_submit"], proxy["P_ready"],
                  proxy["D_submit"], raw["token_times"][0])
    if any(a > b for a, b in zip(timestamps, timestamps[1:])):
        return None
    durations = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]
    if abs(sum(durations) - raw["TTFT_ms"]) > .001:
        return None
    return dict(zip(CRITICAL_PATH, durations, strict=True))


def request_table(cohort, tables):
    categories = Counter()
    decode_pass = 0
    host_rows = defaultdict(list)
    ready_count = available_at_ready = available_without_pending = 0
    ready_split = ready_split_inactive = ready_split_quiet = 0
    ready_window = cutoff(cohort.client_rows)
    service = Counter()
    path_totals = Counter()
    path_count = 0
    wait = []
    split_wait = []
    for raw in cohort.client_rows:
        ttft, itl, tpot = (raw[key] for key in ("TTFT_ms", "ITL_ms", "TPOT_ms"))
        failures = [key for key, failed in (
            ("TTFT", ttft > SLO["TTFT_ms"]), ("ITL", any(i > SLO["ITL_ms"] for i in itl)),
            ("TPOT", tpot > SLO["TPOT_ms"]),
        ) if failed]
        category = "+".join(failures) + "_FAIL" if failures else "JOINT_PASS"
        categories[category] += 1
        decode_pass += not ("ITL" in failures or "TPOT" in failures)
        row = {**cohort.base, "request_id": raw["request_id"], "index": raw["index"],
               "TTFT_ms": ttft, "ITL_p99_ms": percentile(itl, .99), "TPOT_ms": tpot,
               "input_tokens": raw["usage"]["prompt_tokens"], "output_tokens": raw["output_tokens"],
               "SLO_category": category}
        proxy = cohort.proxy_rows.get(raw["runtime_request_id"])
        if proxy:
            row["split_selected"] = proxy["token_enabled"]
            row["suffix_tokens"] = proxy.get("token_partition", {}).get("suffix_tokens", 0)
            path = critical_path(raw, proxy)
            if path is not None:
                path_totals.update(path)
                path_count += 1
            for label, (begin, end) in stage_intervals(proxy).items():
                host_rows[label].append((begin, end))
                row[label + "_ms"] = (end - begin) * 1000
            ready = proxy.get("D_at_P_ready")
            if ready:
                ready_count += 1
                available_at_ready += ready["active_http_requests"] == 0
                available_without_pending += ready["active_http_requests"] == ready["pending_admissions"] == 0
                row["D_at_P_ready_active_http"] = ready["active_http_requests"]
                row["D_at_P_ready_pend_coordinator"] = ready["pending_admissions"]
                if row["split_selected"]:
                    ready_split += 1
                    ready_split_inactive += ready["active_http_requests"] == 0
                    ready_split_quiet += ready["active_http_requests"] == ready["pending_admissions"] == 0
            admission = proxy.get("D_admission")
            if admission:
                delay = (admission["admitted"] - admission["begin"]) * 1000
                wait.append(delay)
                if row["split_selected"]:
                    split_wait.append(delay)
                row["D_coordinator_wait_ms"] = delay
            left, right = ready_window
            if left <= proxy.get("E_complete", -1) < right:
                service["E_reqs"] += 1
                service["E_images"] += raw["image_count"]
            if left <= proxy.get("P_ready", -1) < right:
                service["P_reqs"] += 1
                service["P_input_tokens"] += raw["usage"]["prompt_tokens"] - row["suffix_tokens"]
            if left <= proxy.get("D_http_complete", proxy.get("finished", -1)) < right:
                service["D_reqs"] += 1
            service["D_output_tokens"] += sum(left <= t < right for t in raw["token_times"])
        tables["requests"].append(row)
    for category, count in sorted(categories.items()):
        tables["slo_breakdown"].append({**cohort.base, "category": category, "requests": count,
                                        "output_tokens": cohort.metrics["category_output_tokens"][category]})
    if categories != cohort.metrics["categories"]:
        raise ValueError(f"Recomputed SLO categories differ in {cohort.run.name}/{cohort.tag}")
    left, right = ready_window
    stages = ("E HTTP", "E complete to P submit", "P HTTP", "P ready to D submit", "D HTTP")
    totals = {}
    for stage in stages:
        intervals = host_rows[stage]
        if not intervals:
            continue
        depth = list(depth_series(intervals, left, right))
        totals[stage + "_central_nonempty_fraction"] = elapsed(
            [(begin, end) for begin, end, count in depth if count]) / (right - left)
        for begin, end, count in depth:
            add_window(tables["host_timeline"], cohort, stage, [(begin, end)], left, count)
    # A partial proxy log cannot support an idle measurement. A fully covered
    # coordinator view still cannot detect a scheduler-ready request by itself.
    if len(cohort.proxy_rows) == len(cohort.client_rows) and len(host_rows["D HTTP"]) == len(cohort.client_rows):
        for series, busy in (
            ("No active D HTTP", host_rows["D HTTP"]),
            ("No active D HTTP or P-ready wait", host_rows["D HTTP"] + host_rows["P ready to D submit"]),
        ):
            windows = complement(merged(busy, left, right), left, right)
            add_window(tables["decode_idle_windows"], cohort, series, windows, left)
            totals[series + "_central_fraction"] = elapsed(windows) / (right - left)
    n = len(cohort.client_rows)
    current = [row for row in tables["requests"][-n:]]
    totals.update({
        "proxy_rows": len(cohort.proxy_rows), "central_window_s": right - left,
        "P_ready_D_HTTP_inactive_fraction": available_at_ready / ready_count if ready_count else None,
        "P_ready_D_HTTP_inactive_no_pending_fraction":
            available_without_pending / ready_count if ready_count else None,
        "P_ready_D_observations": ready_count,
        "split_only_P_ready_D_observations": ready_split,
        "split_only_P_ready_D_HTTP_inactive_fraction": ready_split_inactive / ready_split if ready_split else None,
        "split_only_P_ready_D_HTTP_inactive_no_pending_fraction": ready_split_quiet / ready_split
        if ready_split else None,
        "D_coordinator_wait_p50_ms": percentile(wait, .50),
        "D_coordinator_wait_p95_ms": percentile(wait, .95),
        "D_coordinator_wait_p99_ms": percentile(wait, .99),
        "split_only_D_coordinator_wait_p50_ms": percentile(split_wait, .50),
        "split_only_D_coordinator_wait_p95_ms": percentile(split_wait, .95),
        "split_only_D_coordinator_wait_p99_ms": percentile(split_wait, .99),
        "split_requests": sum(bool(row.get("split_selected")) for row in current) if cohort.proxy_rows else None,
        "suffix_tokens": sum(row.get("suffix_tokens", 0) for row in current) if cohort.proxy_rows else None,
        "actual_offload_fraction": sum(row.get("suffix_tokens", 0) for row in current) /
        sum(row["input_tokens"] for row in current) if len(cohort.proxy_rows) == n else None,
        "D_absolute_SLO_attainment": decode_pass / n,
    })
    if len(cohort.proxy_rows) == n:
        totals.update({f"{key}_per_central_second": service[key] / (right - left)
                       for key in ("E_reqs", "E_images", "P_reqs", "P_input_tokens",
                                   "D_reqs", "D_output_tokens")})
    for label, values in host_rows.items():
        totals[label + "_mean_ms"] = mean((b - a) * 1000 for a, b in values)
    if path_count == n:
        tables["critical_path"].extend({**cohort.base, "stage": stage, "mean_ms": path_totals[stage] / n}
                                       for stage in CRITICAL_PATH)
    return totals


def batch_table(cohort, tables):
    path = cohort.run / "analysis" / f"{cohort.tag}_batch.json"
    if not path.exists():
        return {}
    receipt, = read(path)
    hist = read(Path(receipt["folder"]) / "histograms.json")
    for key in ("worker_schedule_histogram", "scheduler_generation_phase_histogram",
                "prepared_query_histogram", "prepared_decode_padding_histogram"):
        for row in hist.get(key, []):
            tables["actual_d_batch"].append({**cohort.base, "series": key,
                                             "counter_valid": hist["valid"], **row})
    return {f"D_max_{key}": value for key, value in receipt.get("decode_maxima", {}).items()}


def kernel_tables(cohort, tables):
    if cohort.observation != "profile":
        return
    path = cohort.run / "analysis" / f"{cohort.tag.removesuffix('_profile')}_trace.json"
    if not path.exists():
        return
    trace = read(path)
    for role in ROLES:
        tables["kernel_busy"].append({**cohort.base, "evidence": trace["evidence"], "role": role,
                                      "busy_fraction": trace["kernel_busy_fraction"][role]})
    for name, values in trace["simultaneous"].items():
        tables["kernel_overlap"].append({**cohort.base, "evidence": trace["evidence"],
                                         "condition": name, **values})
    if trace["evidence"] != "TRACE_ANALYSIS" or cohort.evidence != "TRACE_ANALYSIS":
        return
    left, right = trace["window_monotonic"]
    intervals = trace["kernel_intervals"]
    for role, values in intervals.items():
        add_window(tables["kernel_timeline"], cohort, role, values, left)
    idle = complement(intervals["decode"], left, right)
    simultaneous = intersect(intervals["prefill"], idle)
    add_window(tables["kernel_timeline"], cohort, "P kernel and D physical idle", simultaneous, left)
    add_window(tables["decode_idle_windows"], cohort, "D physical kernel idle", idle, left)


def observer_qualification(campaign):
    output = []
    patterns = (("*_batch_qualification.json", "_batch_qualification.json", "batch_counter"),
                ("*_profile_gate.json", "_profile_gate.json", "kernel_trace"))
    for pattern, suffix, observer in patterns:
        for path in sorted(campaign.glob(f"*/analysis/{pattern}")):
            value = read(path)
            for metric, comparison in value["metrics"].items():
                output.append({"start": path.parent.parent.name, "cohort": path.name.removesuffix(suffix),
                               "observer": observer, "metric": metric, **comparison,
                               "qualification_passed": value["passed"], "threshold": value["threshold"],
                               "evidence": value["evidence"]})
    return output


def capacity_table(campaign):
    output = []
    for path in sorted(campaign.glob("*/analysis/capacity_*.json")):
        value = read(path)
        if value.get("evidence") != "ISOLATED_GPU_MEASUREMENT" or "requests_per_second" not in value:
            continue
        environment = read(path.parent.parent / "state/environment.json")
        output.append({key: value.get(key) for key in (
            "evidence", "stage", "batch_one", "requests", "tokens", "seconds",
            "requests_per_second", "tokens_per_second", "scope", "observer_qualification",
        )} | {"start": path.parent.parent.name, "record": path.name,
              "receipt_SHA256": environment["receipt_SHA256"],
              "wheel_SHA256": environment["runtime"]["vllm_wheel_sha256"],
              "GPU_UUIDs": json.dumps(environment["authorized_uuid_by_gpu"], sort_keys=True)})
    return output


def capacity_proximity(cohorts, cohort_summaries, capacity_rows):
    estimates = []
    for cohort, summary in zip(cohorts, cohort_summaries, strict=True):
        if not cohort.method.startswith("native-common") or cohort.observation not in ("clean", "profile"):
            continue
        for isolated in capacity_rows:
            if (cohort.environment["receipt_SHA256"] != isolated["receipt_SHA256"]
                    or cohort.environment["runtime"]["vllm_wheel_sha256"] != isolated["wheel_SHA256"]
                    or json.dumps(cohort.environment["authorized_uuid_by_gpu"], sort_keys=True) != isolated["GPU_UUIDs"]):
                continue
            role = isolated["stage"]
            if role == "decode" and (summary["output_mode"] == "Natural EOS"
                                     or bool(isolated["batch_one"]) != cohort.method.endswith("-B1")):
                continue
            fields = {
                "encoder": ("E_reqs_per_central_second", None),
                "prefill": ("P_reqs_per_central_second", "P_input_tokens_per_central_second"),
                "decode": ("D_reqs_per_central_second", "D_output_tokens_per_central_second"),
            }[role]
            for metric, field in zip(("requests_per_second", "tokens_per_second"), fields, strict=True):
                if field is None or field not in summary or not isolated.get(metric):
                    continue
                estimates.append({**cohort.base, "evidence": "DIAGNOSTIC", "role": role,
                                  "capacity_start": isolated["start"], "units": metric,
                                  "online_rate": summary[field], "isolated_rate": isolated[metric],
                                  "rate_ratio": summary[field] / isolated[metric],
                                  "scope": "Central coordinator completions versus isolated finite cohorts. "
                                           "Geometry, KV transport and observer differences remain."})
    return estimates


def paired_contrasts(cohorts):
    groups = defaultdict(list)
    for cohort in cohorts:
        if cohort.observation in ("clean", "batch_observed"):
            groups[cohort.match].append(cohort)
    rows = []
    def method_rank(cohort):
        return (not cohort.method.startswith("native"), not cohort.method.endswith("-free"), cohort.method)

    for members in groups.values():
        for baseline, candidate in combinations(sorted(members, key=method_rank), 2):
            if baseline.method == candidate.method or baseline.run == candidate.run:
                continue
            for metric in METRICS:
                a, b = baseline.metrics.get(metric), candidate.metrics.get(metric)
                if a is None or b is None:
                    continue
                rows.append({**candidate.base, "baseline": baseline.method,
                             "baseline_start": baseline.run.name, "metric": metric,
                             "baseline_evidence": baseline.evidence,
                             "evidence": "DIAGNOSTIC" if "DIAGNOSTIC" in (baseline.evidence, candidate.evidence)
                             else candidate.evidence,
                             "baseline_value": a, "candidate_value": b, "delta": b - a,
                             "ratio": b / a if a else None,
                             "matching": "arrival SHA256, manifest SHA256, runtime wheel, GPU IDs and observation"})
    return rows


def paired_intervals(contrasts):
    groups = defaultdict(list)
    for row in contrasts:
        groups[(row["baseline"], row["method"], row["QPS"], row["output_mode"],
                row["observation"], row["metric"])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        # Multiple starts with the same arrival may not be matched uniquely.
        # Keep all transparent N1 comparisons and omit a CI on ambiguous pairs.
        left_starts = {row["baseline_start"] for row in rows}
        right_starts = {row["start"] for row in rows}
        n = len(rows)
        if n < 3 or len(left_starts) != n or len(right_starts) != n or left_starts & right_starts:
            continue
        if any(row["baseline_value"] <= 0 for row in rows):
            continue
        rng = random.Random(20261125)
        draws = []
        for _ in range(10000):
            sample = rng.choices(rows, k=n)
            draws.append(sum(row["candidate_value"] for row in sample) /
                         sum(row["baseline_value"] for row in sample))
        output.append(dict(zip(("baseline", "method", "QPS", "output_mode", "observation", "metric"), key),
                           pairs=n, ratio=mean(r["candidate_value"] for r in rows) /
                           mean(r["baseline_value"] for r in rows),
                           CI_low=percentile(draws, .025), CI_high=percentile(draws, .975)))
    return output


def save_figures(output, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    def groups(rows, keys):
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[key] for key in keys)].append(row)
        return sorted(grouped.items())

    def cdf(ax, values, label):
        values = sorted(values)
        ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", label=label)

    def label(members):
        first = members[0]
        return f"{first['method']} q{first['QPS']:g} · {first['start']}"

    for mode, view in groups(
        [r for r in tables["cohorts"] if r["observation"] in ("clean", "batch_observed")], ("output_mode",),
    ):
        output_tag = "natural" if mode[0] == "Natural EOS" else "replay"
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for (method, obs), members in groups(view, ("method", "observation")):
            for ax, metric in zip(axes, ("TTFT_p95_ms", "ITL_p99_ms"), strict=True):
                series = [(qps[0], mean(r[metric] for r in points if r[metric] is not None))
                          for qps, points in groups(members, ("QPS",))
                          if any(r[metric] is not None for r in points)]
                if series:
                    x, y = zip(*series)
                    ax.plot(x, y, marker="o", label=f"{method} · {obs}")
                    ax.set(xlabel="Offered QPS", ylabel=metric.replace("_", " "))
                    ax.grid(alpha=.2)
        if axes[0].lines:
            axes[0].legend(fontsize=7)
        fig.suptitle(mode[0])
        save(fig, f"native_qps_latency_{output_tag}")

        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        for row in view:
            ax.scatter(row["TTFT_p95_ms"], row["ITL_p99_ms"],
                       label=f"{row['method']} q{row['QPS']:g} · {row['observation']}")
        ax.set(xlabel="TTFT p95 (ms)", ylabel="ITL p99 (ms)", title=mode[0])
        ax.legend(fontsize=7)
        save(fig, f"ttft_itl_tradeoff_{output_tag}")

    for (qps, mode), selected in groups(tables["requests"], ("QPS", "output_mode")):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        standard = [r for r in selected if r["observation"] in ("clean", "batch_observed")]
        for _, members in groups(standard, ("start", "cohort")):
            cdf(ax, [row["TTFT_ms"] for row in members], label(members))
        if ax.lines:
            ax.set(xlabel="TTFT (ms)", ylabel="Request CDF",
                   title=f"Main100 · q{qps:g} · {mode}")
            ax.legend(fontsize=7)
            save(fig, f"ttft_cdf_{qps:g}_{'natural' if mode == 'Natural EOS' else 'replay'}")
        else:
            plt.close(fig)

    for (start, cohort), rows in groups(tables["host_timeline"], ("start", "cohort")):
        fig, axes = plt.subplots(5, 1, figsize=(9, 7), sharex=True)
        series = ("E HTTP", "E complete to P submit", "P HTTP", "P ready to D submit", "D HTTP")
        for ax, name in zip(axes, series, strict=True):
            members = [r for r in rows if r["series"] == name]
            if members:
                boundaries = [row["begin_seconds"] for row in members] + [members[-1]["end_seconds"]]
                ax.stairs([row["quantity"] for row in members], boundaries)
            ax.set_ylabel(name, fontsize=8)
        axes[-1].set_xlabel("Seconds after central arrival window begins")
        fig.suptitle(f"Coordinator HTTP and wait intervals · {start} / {cohort}")
        save(fig, f"host_timeline_{start}_{cohort}")

    for (start, cohort), rows in groups(tables["decode_idle_windows"], ("start", "cohort")):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for (series,), members in groups(rows, ("series",)):
            cdf(ax, [r["duration_seconds"] * 1000 for r in members], series)
        ax.set(xlabel="Central window duration (ms)", ylabel="Idle-window CDF",
               title=f"{start} / {cohort}")
        ax.legend(fontsize=8)
        save(fig, f"d_idle_cdf_{start}_{cohort}")

    if tables["kernel_busy"]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for (method, role, evidence), rows in groups(tables["kernel_busy"], ("method", "role", "evidence")):
            rows = sorted(rows, key=lambda r: r["QPS"])
            ax.plot([r["QPS"] for r in rows], [r["busy_fraction"] * 100 for r in rows],
                    linestyle="-" if evidence == "TRACE_ANALYSIS" else ":", marker="o",
                    label=f"{method} · {role} · {evidence}")
        ax.set(xlabel="Offered QPS", ylabel="CUPTI kernel busy (%)", ylim=(0, 100))
        ax.legend(fontsize=7)
        save(fig, "kernel_busy_by_qps")

    for (start, cohort), rows in groups(tables["kernel_timeline"], ("start", "cohort")):
        # Predeclared centered four-second crop for a readable exact kernel trace.
        center = max(r["end_seconds"] for r in rows) / 2
        left, right = max(0, center - 2), center + 2
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for i, role in enumerate((*ROLES, "P kernel and D physical idle")):
            spans = merged([(r["begin_seconds"], r["end_seconds"]) for r in rows if r["series"] == role],
                           left, right)
            ax.broken_barh([(begin, end - begin) for begin, end in spans], (i - .32, .64))
        ax.set(xlim=(left, right), yticks=range(4), yticklabels=(*ROLES, "P busy ∩ D idle"),
               xlabel="Seconds after central arrival window begins", title=f"Qualified CUPTI · {start} / {cohort}")
        save(fig, f"kernel_timeline_{start}_{cohort}")

    actual = [r for r in tables["actual_d_batch"] if r["series"] == "prepared_decode_padding_histogram"
              and r["counter_valid"] and r["decode_requests"] > 0]
    if actual:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for _, members in groups(actual, ("start", "cohort")):
            counts = Counter()
            for row in members:
                counts[row["decode_requests"]] += row["calls"]
            x = sorted(counts)
            ax.plot(x, [counts[i] / sum(counts.values()) for i in x], marker="o", label=label(members))
        ax.set(xlabel="Actual prepared decode requests per forward",
               ylabel="Fraction of decode-containing forward calls", title="Worker batch counter cohorts")
        ax.legend(fontsize=7)
        save(fig, "actual_decode_batch_histogram")

    if tables["slo_breakdown"]:
        for (mode,), rows in groups(tables["slo_breakdown"], ("output_mode",)):
            cohorts = groups(rows, ("start", "cohort"))
            categories = sorted({row["category"] for row in rows})
            fig, ax = plt.subplots(figsize=(max(7, len(cohorts) * .62), 4.5))
            bottoms = [0] * len(cohorts)
            for category in categories:
                values = [sum(r["requests"] for r in members if r["category"] == category)
                          for _, members in cohorts]
                ax.bar(range(len(cohorts)), values, bottom=bottoms, label=category)
                bottoms = [a + b for a, b in zip(bottoms, values, strict=True)]
            ax.set(xticks=range(len(cohorts)), xticklabels=[label(members) for _, members in cohorts],
                   ylabel="Requests", title=mode)
            ax.tick_params(axis="x", labelrotation=55, labelsize=7)
            ax.legend(fontsize=7)
            save(fig, f"slo_breakdown_{'natural' if mode == 'Natural EOS' else 'replay'}")

    scoped_cohorts = [r for r in tables["cohorts"] if r["observation"] in ("clean", "batch_observed")]
    for (mode,), rows in groups(scoped_cohorts, ("output_mode",)):
        selected = [row for row in rows if row["P_ready_D_observations"]]
        suffix = "natural" if mode == "Natural EOS" else "replay"
        if selected:
            fig, ax = plt.subplots(figsize=(max(7, len(selected) * .65), 4.2))
            quiet = [r["P_ready_D_HTTP_inactive_no_pending_fraction"] for r in selected]
            pending = [r["P_ready_D_HTTP_inactive_fraction"] - q for r, q in zip(selected, quiet, strict=True)]
            active = [1 - r["P_ready_D_HTTP_inactive_fraction"] for r in selected]
            positions = range(len(selected))
            ax.bar(positions, quiet, label="No active HTTP or pending coordinator admission")
            ax.bar(positions, pending, bottom=quiet, label="Pending coordinator admission")
            ax.bar(positions, active, bottom=[q + a for q, a in zip(quiet, pending, strict=True)],
                   label="Active D HTTP")
            ax.set(xticks=positions, xticklabels=[label([row]) for row in selected], ylim=(0, 1),
                   ylabel="Share of observed requests", title=f"P ready · coordinator state · {mode}")
            ax.tick_params(axis="x", labelrotation=55, labelsize=7)
            ax.legend(fontsize=7)
            save(fig, f"p_ready_d_state_{suffix}")

        path_rows = [r for r in tables["critical_path"] if r["output_mode"] == mode
                     and r["observation"] in ("clean", "batch_observed")]
        if path_rows:
            keys = groups(path_rows, ("start", "cohort"))
            positions = range(len(keys))
            fig, ax = plt.subplots(figsize=(max(7, len(keys) * .65), 4.8))
            bottom = [0] * len(keys)
            for stage in CRITICAL_PATH:
                values = [next(r["mean_ms"] for r in group if r["stage"] == stage) for _, group in keys]
                ax.bar(positions, values, bottom=bottom, label=stage)
                bottom = [x + y for x, y in zip(bottom, values, strict=True)]
            ax.set(xticks=positions, xticklabels=[label(group) for _, group in keys],
                   ylabel="Mean TTFT (ms)", title=f"Additive client and proxy timestamps · {mode}")
            ax.tick_params(axis="x", labelrotation=55, labelsize=7)
            ax.legend(fontsize=7)
            save(fig, f"critical_path_{suffix}")

        candidate = [r for r in rows if r["actual_offload_fraction"] is not None
                     and (r["actual_offload_fraction"] > 0 or r["method"].endswith("-B1"))]
        if any(row["actual_offload_fraction"] > 0 for row in candidate):
            for (qps,), members in groups(candidate, ("QPS",)):
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
                for (method,), arm in groups(members, ("method",)):
                    for ax, metric in zip(axes, ("TTFT_p95_ms", "ITL_p99_ms"), strict=True):
                        ax.scatter([r["actual_offload_fraction"] * 100 for r in arm],
                                   [r[metric] for r in arm], label=method)
                        ax.set(xlabel="Actual offloaded input tokens (%)", ylabel=metric.replace("_", " "))
                axes[0].legend(fontsize=7)
                fig.suptitle(f"Main100 · q{qps:g} · {mode}")
                save(fig, f"offload_latency_{qps:g}_{suffix}")

    if tables["service_proximity"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
        for ax, unit in zip(axes, ("requests_per_second", "tokens_per_second"), strict=True):
            members = [r for r in tables["service_proximity"] if r["units"] == unit]
            for (role, method, mode), rows in groups(members, ("role", "method", "output_mode")):
                ax.scatter([r["isolated_rate"] for r in rows], [r["online_rate"] for r in rows],
                           label=f"{role} · {method} · {mode}")
            if members:
                ceiling = max(max(r["isolated_rate"], r["online_rate"]) for r in members)
                ax.plot((0, ceiling), (0, ceiling), "k:", linewidth=.8)
                ax.legend(fontsize=7)
            ax.set(xlabel=f"Isolated {unit.replace('_', ' ')}",
                   ylabel=f"Central online {unit.replace('_', ' ')}")
        fig.suptitle("Diagnostic service rate comparison")
        save(fig, "isolated_capacity_vs_online")


def aggregate(campaign, output, figures=True):
    cohorts, excluded = cohort_records(campaign)
    tables = {name: [] for name in (
        "cohorts", "requests", "slo_breakdown", "host_timeline", "decode_idle_windows",
        "kernel_busy", "kernel_overlap", "kernel_timeline", "actual_d_batch", "critical_path",
    )}
    for cohort in cohorts:
        host = request_table(cohort, tables)
        batch = batch_table(cohort, tables)
        kernel_tables(cohort, tables)
        tables["cohorts"].append({
            **cohort.base, **{metric: cohort.metrics.get(metric) for metric in METRICS},
            **cohort.metrics["gate_attainment"], **host, **batch,
        })
    tables["excluded"] = excluded
    tables["stage_capacity"] = capacity_table(campaign)
    tables["observer_qualification"] = observer_qualification(campaign)
    tables["service_proximity"] = capacity_proximity(cohorts, tables["cohorts"], tables["stage_capacity"])
    tables["contrasts"] = paired_contrasts(cohorts)
    tables["contrast_ci"] = paired_intervals(tables["contrasts"])
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        if rows:
            csv_file(output / f"{name}.csv", rows)
    if figures and cohorts:
        save_figures(output, tables)
    receipt = {
        "cohorts": len(cohorts), "table_rows": {key: len(rows) for key, rows in tables.items()},
        "central_window_rule": "Scheduled-arrival ranks floor(100*.20) through floor(100*.80).",
        "proxy_interval_scope": "HTTP activity and coordinator wait. Engine backlog and physical idle require separate evidence.",
        "separate_outputs": "Natural EOS and fixed-length replay on separate axes and in separate comparison groups.",
        "qualification": "Unqualified observers and failed gates are diagnostic. Qualified CUPTI timelines only.",
        "figure_rule": "CUPTI timeline uses the middle four seconds of the qualified central window.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.runs, args.output, figures=not args.no_figures), indent=2))


if __name__ == "__main__":
    main()
