"""Bind an optimistic E release improvement to an observed P completion order."""

import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path

from ..mllm_epd_prefill_heavy_modality_discovery import timeline
from ..mllm_epd_prefill_heavy_modality_discovery.reduce_epd import payload_receipt
from .idle_feasibility import load_case


DEFAULT_CASES = (
    "long:native-recovery/fresh_native_n64",
    "long:native-recovery/fresh_native_n128",
    "long-low:native-recovery/fresh_native_lower_n64",
)
WAITING = "vllm:num_requests_waiting"


def analyse(root, case, proxy_cache, advance_ms, max_sample_age_ms):
    origin, _, (left, right), _, proxy = load_case(root, case, proxy_cache)
    config = root / origin["epoch"] / "raw" / f"{origin['tag']}_scheduler.jsonl"
    samples = sorted((row for row in timeline.read_jsonl(config) if WAITING in row.get("values", {}).get("prefill", {})),
                     key=lambda row: row["monotonic"])
    sample_times = [row["monotonic"] for row in samples]
    ordered = sorted(proxy, key=lambda row: row["P_ready"])
    submitted = [row["P_submit"] for row in ordered]
    inversions = sum(a > b for i, a in enumerate(submitted) for b in submitted[i+1:])
    metrics_count = sum(bool((row.get("P_response") or {}).get("metrics")) for row in ordered)
    previous = None
    reference, advanced = None, None
    rows = []
    for row in ordered:
        submitted, completed = row["P_submit"], row["P_ready"]
        e_begin, e_ready = row["E_submit"], row["E_complete"]
        if e_ready < e_begin:
            raise ValueError(f"{case}: invalid observed E enclosure")
        in_window = left <= e_ready < right
        advance = min(advance_ms / 1000, e_ready - e_begin) if in_window else 0.
        if previous is None:
            fixed_gate = submitted
            earlier_gate = submitted - advance
        else:
            fixed_gate = max(submitted, previous)
            earlier_gate = max(submitted - advance, previous)
        service = completed - fixed_gate
        if service < 0:
            raise ValueError(f"{case}: negative effective service interval in observed order")
        reference = max(submitted, reference) + service if reference is not None else submitted + service
        advanced = (max(submitted - advance, advanced) if advanced is not None else submitted - advance) + service
        if abs(reference - completed) > 2e-6:
            raise ValueError(f"{case}: the unmodified surrogate failed to reconstruct the native completion sequence")
        if in_window:
            i = bisect_right(sample_times, e_ready) - 1
            sample_age = (e_ready - sample_times[i]) * 1000 if i >= 0 else None
            waiting = samples[i]["values"]["prefill"][WAITING] if sample_age is not None and sample_age <= max_sample_age_ms else None
            pending = sum(other is not row and other["P_submit"] <= e_ready < other["P_ready"] for other in ordered)
            rows.append({"case": case, "E_ready_from_ROI_start_ms": (e_ready - left) * 1000,
                         "P_submit_from_ROI_start_ms": (submitted - left) * 1000,
                         "P_ready_from_ROI_start_ms": (completed - left) * 1000,
                         "native_P_HTTP_requests_outstanding_at_E_ready": pending,
                         "last_P_waiting_sample_at_E_ready": waiting,
                         "last_waiting_sample_age_ms": sample_age,
                         "P_submit_to_predecessor_completion_ms": (fixed_gate - submitted) * 1000,
                         "observed_order_effective_P_service_ms": service * 1000,
                         "assumed_E_advance_ms": advance * 1000,
                         "isolated_E_advance_effect_at_P_ready_ms": (fixed_gate - earlier_gate) * 1000,
                         "propagated_E_advance_effect_at_P_ready_ms": (completed - advanced) * 1000})
        previous = completed

    def stats(key):
        values = [row[key] for row in rows if row[key] is not None]
        return {"count": len(values), "positive": sum(value > .001 for value in values),
                "mean": sum(values) / len(values) if values else None, **timeline.distribution(values)}

    return {"case": origin, "scheduler": payload_receipt(config), "ROI_seconds": right - left,
            "eligible_E_ready": len(rows), "P_response_metrics_present": metrics_count,
            "P_submit_order_inversions_in_P_completion_order": inversions,
            "P_waiting": stats("last_P_waiting_sample_at_E_ready"),
            "P_HTTP_outstanding": stats("native_P_HTTP_requests_outstanding_at_E_ready"),
            "P_waiting_sample_age_ms": stats("last_waiting_sample_age_ms"),
            "isolated_P_ready_improvement_ms": stats("isolated_E_advance_effect_at_P_ready_ms"),
            "propagated_P_ready_improvement_ms": stats("propagated_E_advance_effect_at_P_ready_ms")}, rows


def fmt(value, decimals=2):
    return "—" if value is None else f"{value:,.{decimals}f}"


def report(value, data_path):
    advance = value["contract"]["optimistic_E_advance_ms"]
    advances = ", ".join(f"{row['propagated_P_ready_improvement_ms']['positive']}/{row['eligible_E_ready']} "
                         f"in `{row['case']['tag']}`" for row in value["case_results"])
    samples = value["contract"]["max_P_waiting_sample_age_ms"]
    metrics = sum(row["P_response_metrics_present"] for row in value["case_results"])
    inputs = sum(row["case"]["count"] for row in value["case_results"])
    result = ["# Encoder-only critical-path bound", "",
              f"An optimistic {advance:.2f} ms encoder release advance gives earlier P handoffs for {advances} "
              "in the propagated fixed-order replay.", "",
              "The physical motivation is the measured balanced whole-image probe with some independent images executed on D. "
              "The replay applies its entire latency reduction to every central E-ready request while charging zero extra movement, "
              "scheduling, or D contention. The primary cases preserve their original images, arrival times and no-assist native route.", "",
              "## Observed host timeline · TRACE_ANALYSIS", "",
              "These clean traces contain the actual encoder acknowledgment, P HTTP submission, and P-ready acknowledgment. "
              f"Per-request P engine dispatch requires additional attribution. {metrics}/{inputs} P HTTP responses carry metrics. "
              f"The last sampled P waiting count has a maximum allowed age of {samples:g} ms. P waiting counts and outstanding HTTP "
              "enclosures describe different queue boundaries.", "",
              "| Cohort | Central E-ready events | Earlier P HTTP outstanding at E-ready, positive / median | P waiting sample positive / coverage | P waiting median | P completion/submission inversions |",
              "| --- | --- | --- | --- | --- | --- |"]
    for entry in value["case_results"]:
        name = entry["case"]["tag"]
        pending, waiting = entry["P_HTTP_outstanding"], entry["P_waiting"]
        result.append(f"| `{name}` | {entry['eligible_E_ready']} | {pending['positive']}/{pending['count']} / "
                      f"{fmt(pending['p50'])} | {waiting['positive']}/{waiting['count']} | {fmt(waiting['p50'])} | "
                      f"{entry['P_submit_order_inversions_in_P_completion_order']} |")
    result.extend(["", "## Fixed observed-order replay · ANALYTICAL_BOUND", "",
                   "Order all requests by measured P-ready time. For each request use effective service "
                   "`observed P-ready − max(P HTTP submit, previous observed P-ready)`. "
                   "This reconstructs the original handoff series exactly when no release changes. It assigns each gap to a serial "
                   "predecessor and can absorb batching and other stage costs into that gap. The effective service is a model quantity.",
                   "",
                   "Advance each central P release by the full probe gain, capped at its recorded E enclosure. "
                   "The isolated counterfactual keeps every preceding native completion fixed. The propagated counterfactual applies all the "
                   "central release advances, and each earlier modeled handoff can move the next service gate. Prior work outside the "
                   "intervention keeps its native release. Output times refer to modeled P-ready. D-side scheduling effects remain open.", "",
                   "| Cohort | Isolated improved | Isolated mean / max ms | Propagated improved | Propagated mean / max ms |",
                   "| --- | --- | --- | --- | --- |"])
    for entry in value["case_results"]:
        isolated = entry["isolated_P_ready_improvement_ms"]
        propagated = entry["propagated_P_ready_improvement_ms"]
        result.append(f"| `{entry['case']['tag']}` | {isolated['positive']}/{isolated['count']} | "
                      f"{fmt(isolated['mean'])}/{fmt(isolated['max'])} | {propagated['positive']}/{propagated['count']} | "
                      f"{fmt(propagated['mean'])}/{fmt(propagated['max'])} |")
    result.extend(["", "The main cohorts contain overtakes in the observed P-ready order. Therefore this replay preserves observed "
                   "completion order rather than assuming submission FIFO. The lower-load case happens to preserve submission "
                   "order. A change that rearranges P work or changes P batching could affect individual requests. "
                   "Establishing that effect requires P dispatch/completion attribution and an actual matched intervention.", "",
                   f"[Raw-derived summary]({data_path}/encoder_critical_path.json) · [Per-request reduction]({data_path}/encoder_critical_path_rows.csv)"])
    return "\n".join(result) + "\n"


def plot(value, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = value["case_results"]
    titles = [case["case"]["tag"].removeprefix("fresh_native_") for case in cases]
    labels = range(len(cases))
    figure, axes = plt.subplots(1, 2, figsize=(9.3,3.9), layout="constrained")
    axes[0].barh(labels, [case["P_HTTP_outstanding"]["p50"] for case in cases], color="#2f6b92")
    axes[0].set(xlabel="Median outstanding requests", title="Observed older P work at E completion")
    gains = [case["propagated_P_ready_improvement_ms"] for case in cases]
    bars = axes[1].barh(labels, [100*gain["positive"]/gain["count"] for gain in gains], color="#4a7452")
    axes[1].bar_label(bars, labels=[f'{gain["positive"]}/{gain["count"]}' for gain in gains], padding=4)
    advance = value["contract"]["optimistic_E_advance_ms"]
    axes[1].set(xlim=(0,100), xlabel="Earlier modelled P handoffs (%)",
                title=f"Fixed-order {advance:g} ms earlier E release")
    for axis in axes:
        axis.set_yticks(labels, labels=titles)
        axis.invert_yaxis()
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Override the native cases, using FAMILY:EPOCH/TAG")
    parser.add_argument("--advance-ms", type=float, default=88.38)
    parser.add_argument("--max-sample-age-ms", type=float, default=750)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--link-root", required=True)
    args = parser.parse_args()
    if min(args.advance_ms, args.max_sample_age_ms) < 0:
        parser.error("Advance and sampling bounds must be nonnegative")
    observations, rows, proxies = [], [], {}
    for specimen in args.case or DEFAULT_CASES:
        observation, detail = analyse(args.raw_root, specimen, proxies, args.advance_ms, args.max_sample_age_ms)
        observations.append(observation)
        rows.extend(detail)
    if not rows:
        raise ValueError("No encoder-ready decisions in the selected observation windows")
    value = {"case_results": observations,
             "contract": {"evidence": ("TRACE_ANALYSIS", "ANALYTICAL_BOUND"),
                          "optimistic_E_advance_ms": args.advance_ms,
                          "max_P_waiting_sample_age_ms": args.max_sample_age_ms,
                          "physical_scope": "Observed native P HTTP submit / P ready; P engine first dispatch absent.",
                          "timing_model": "Fixed original P completion order and serial effective service. No reorder, batching change, "
                                          "new communication, interference or D-side gain."}}
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "encoder_critical_path.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with (args.out / "encoder_critical_path_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.report.write_text(report(value, args.link_root))
    if args.figure:
        plot(value, args.figure)
    print(f"Encoder bound over {len(rows)} E-ready requests in {len(observations)} observed native cases")


if __name__ == "__main__":
    main()
