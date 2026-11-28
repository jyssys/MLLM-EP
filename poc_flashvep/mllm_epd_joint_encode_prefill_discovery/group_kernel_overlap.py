"""Measure clock-aligned E/LM kernel overlap in a CUDA-ready serial request."""

import argparse
import hashlib
import json
from pathlib import Path

from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import cpc_gate
from ..mllm_epd_prefill_heavy_modality_discovery import timeline


TAG = "group_kernel_overlap"


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def observer_inputs(row):
    itl = timeline.distribution(row["ITL_ms"])
    return {
        "duration_seconds": (row["TTFT_ms"] + sum(row["ITL_ms"])) / 1000,
        "TTFT_p50_ms": row["TTFT_ms"],
        "TTFT_p95_ms": row["TTFT_ms"],
        "TPOT_p95_ms": row["TPOT_ms"],
        "ITL_p99_ms": itl["p99"],
    }


def match(trial, candidates, before):
    fields = ("fixture", "group_mode", "arm", "cut", "prefix_images", "prompt_tokens", "prompt_token_ids_SHA256")
    direction = -1 if before else 1
    matches = [row for row in candidates if all(row[key] == trial[key] for key in fields)
               and direction * (row["start_ns"] - trial["start_ns"]) > 0]
    return min(matches, key=lambda row: abs(row["start_ns"] - trial["start_ns"]), default=None)


def observer_check(before, trial, after, receipts, captures, full_window):
    if before is None or after is None:
        return {"passed": False, "status": "UNPAIRED"}
    gate = cpc_gate(*(observer_inputs(row) for row in (before, trial, after)), receipts)
    exact = before["output_token_ids"] == trial["output_token_ids"] == after["output_token_ids"]
    fixed16 = all(len(row["output_token_ids"]) == 16 and len(row["ITL_ms"]) == 15
                  for row in (before, trial, after))
    intact = all(row["clock_ok"] and row["collector_ok"] for row in captures.values())
    passed = gate["passed"] and exact and fixed16 and intact and full_window
    return {**gate, "passed": passed, "status": "PASS" if passed else "FAIL",
            "evidence": "TRACE_ANALYSIS" if passed else "DIAGNOSTIC",
            "exact_output": exact, "fixed16_complete": fixed16, "intact_clocks": intact,
            "full_analysis_interval_covered": full_window,
            "clean_before_tag": before["tag"], "clean_after_tag": after["tag"],
            "scope": "Two bracketing clean individual requests and one traced request per arm. "
                     "TTFT p50/p95 are the same single observation. This is a serial observer check."}


def trial_overlap(trial, last_group, trace_dir, gates):
    before, after, receipts = gates
    recipient = "prefill" if trial["arm"] == "early_p" else "decode"
    submit = trial["recipient_submit_ns"] * 1e-9
    ready_lower = last_group["cuda_ready_lower_ns"] * 1e-9
    ready_upper = last_group["cuda_ready_upper_ns"] * 1e-9
    basic = {"tag": trial["tag"], "fixture": trial["fixture"], "group_mode": trial["group_mode"],
             "arm": trial["arm"], "recipient": recipient, "evidence": "DIAGNOSTIC",
             "nominal_window_seconds": [submit, ready_lower],
             "last_E_event_ready_bounds_seconds": [ready_lower, ready_upper]}
    if ready_lower <= submit:
        return {**basic, "status": "NO_POSITIVE_REQUEST_WINDOW"}
    role_paths = {role: trace_dir / role for role in timeline.ROLES}
    anchor_paths = {role: folder / "actual_ROI.json" for role, folder in role_paths.items()}
    absent = [role for role, anchor in anchor_paths.items() if not anchor.is_file()]
    if absent:
        return {**basic, "status": "MISSING_COLLECTOR_ANCHORS", "roles": absent}
    coverage = {}
    for role, anchor in anchor_paths.items():
        meta = timeline.read_json(anchor)
        coverage[role] = timeline.clock_pair(meta["left"], meta["right"],
                                             ("host_lower_ns", "host_upper_ns"))[0]
    left = max(submit, *(bounds[0] for bounds in coverage.values()))
    right = min(ready_lower, *(bounds[1] for bounds in coverage.values()))
    if left >= right:
        return {**basic, "status": "NO_COMMON_COLLECTOR_COVERAGE", "coverage_seconds": coverage}
    captures = {role: timeline.kernel_capture(folder, left, right) for role, folder in role_paths.items()}
    gate = observer_check(before, trial, after, receipts, captures, left == submit and right == ready_lower)
    integrity = {role: {k: v for k, v in capture.items() if k not in {"kernel", "copy"}}
                 for role, capture in captures.items()}
    basic.update(collector_integrity=integrity, observer_gate=gate, actual_window_seconds=[left, right])
    if any(not x["collector_ok"] or not x["clock_ok"] for x in captures.values()):
        return {**basic, "status": "COLLECTOR_OR_CLOCK_FAILURE"}
    spans = {role: capture["kernel"] for role, capture in captures.items()}
    overlap = timeline.intersect(spans["encoder"], spans[recipient])
    overlap_ms = 1000 * timeline.elapsed(overlap)
    return {
        **basic, "status": "REDUCED", "evidence": "TRACE_ANALYSIS" if gate["passed"] else "DIAGNOSTIC",
        "window_ms": 1000 * (right - left),
        "union_busy_ms": {role: 1000 * timeline.elapsed(intervals) for role, intervals in spans.items()},
        "encoder_recipient_overlap_ms": overlap_ms,
        "encoder_recipient_overlap_fraction": overlap_ms / (1000 * (right - left)),
        "kernel_busy_fraction_by_time_bin": {
            **{role: timeline.bins(intervals, left, right, 240) for role, intervals in spans.items()},
            "joint": timeline.bins(overlap, left, right, 240),
        },
        "scope": "Clock-aligned union of concurrent-kernel intervals. This window has one test request, "
                 "with per-process kernel activity rather than per-kernel request attribution. "
                 "Kernel intersections do not include DMA, imply a deadline-safe budget, or qualify online benefit.",
    }


def reduce(run):
    raw = run / "raw"
    profiled_path = raw / "group_profiled.jsonl"
    if not profiled_path.is_file():
        return {"status": "NOT_RUN", "arms": []}
    clean_before_path = raw / "group_trials.jsonl"
    clean_after_path = raw / "group_profile_aa.jsonl"
    receipts_path = raw / f"{TAG}_profile_receipts.json"
    groups_path = raw / "group_encoder.jsonl"
    profiled = timeline.read_jsonl(profiled_path)
    clean_before = timeline.read_jsonl(clean_before_path) if clean_before_path.is_file() else []
    clean_after = timeline.read_jsonl(clean_after_path) if clean_after_path.is_file() else []
    receipts = timeline.read_json(receipts_path) if receipts_path.is_file() else {}
    trace_dir = raw / "profiles" / TAG
    inputs = [profiled_path, clean_before_path, clean_after_path, groups_path, receipts_path]
    inputs.extend(path for role in timeline.ROLES for path in (trace_dir / role).glob("*") if path.is_file())
    artifacts = {str(path.relative_to(run)): file_hash(path) for path in inputs if path.is_file()}
    by_image = {image: g for g in timeline.read_jsonl(groups_path) for image in g["ids"]}
    answers = []
    for trial in profiled:
        groups = [by_image[image] for image in trial["image_ids"]]
        if any(trial["group_mode"] != g["mode"] for g in groups):
            raise ValueError("Mismatched E group mode in profiled request")
        last_group = max(groups, key=lambda row: row["cuda_ready_lower_ns"])
        controls = (match(trial, clean_before, True), match(trial, clean_after, False), receipts)
        answers.append(trial_overlap(trial, last_group, trace_dir, controls))
    fully_qualified = (len(answers) == 2 and {row["arm"] for row in answers} == {"early_p", "early_d"}
                       and all(row["evidence"] == "TRACE_ANALYSIS" for row in answers))
    return {"status": "TRACE_ANALYSIS" if fully_qualified else "DIAGNOSTIC",
            "scope": "Two serial profiled arms. The joint observer gate requires both arms to pass.",
            "arms": answers, "input_sha256": artifacts}


def figure(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [arm for arm in result["arms"] if arm["status"] == "REDUCED"]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.5 * len(rows)), squeeze=False, layout="constrained")
    for ax, row in zip(axes[:, 0], rows, strict=True):
        activity = row["kernel_busy_fraction_by_time_bin"]
        units = ["encoder", row["recipient"], "joint"]
        ax.imshow([activity[unit] for unit in units], extent=(0, row["window_ms"], 2.5, -.5),
                  aspect="auto", vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.set(yticks=[0, 1, 2], yticklabels=["E kernels", "Recipient kernels", "Overlap"],
               xlabel="Milliseconds within the common kernel capture window",
               title=f"{row['arm']} · {row['encoder_recipient_overlap_ms']:.2f} ms of kernel overlap · {row['evidence']}")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    value = reduce(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure(value, args.figure)


if __name__ == "__main__":
    main()
