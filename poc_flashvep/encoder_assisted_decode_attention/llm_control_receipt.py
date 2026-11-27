"""Verify matched LLM controls and summarize observed helper residency and stage overlap."""
import argparse
import hashlib
import json
from pathlib import Path

from .analyze_online import common_duration, percentile, union


def duration(intervals):
    return sum(end - begin for begin, end in intervals)


def raw(path):
    return json.loads(path.read_text())


def matched_fixed_shape(runs):
    fixed = [(path, run) for path, run in runs if run["workload"] != "real-text"]
    first_path, first_run = fixed[0]
    first = first_run["requests"]
    if len(first) != 100:
        raise ValueError("Fixed shape control must start with the full Main100 roster")
    fixed_output_completions = {}
    for path, run in fixed:
        if run["cap"] != 64 or run["offered_qps"] != first_run["offered_qps"]:
            raise ValueError(f"Fixed shape control changes the output or load contract at {path}")
        completed = 0
        for reference, candidate in zip(first, run["requests"], strict=True):
            reference_offset = reference["planned"] - first_run["start_monotonic"]
            candidate_offset = candidate["planned"] - run["start_monotonic"]
            if (reference["id"] != candidate["id"] or abs(reference_offset - candidate_offset) > 2e-6
                    or reference["input_rows"] != candidate["input_rows"]):
                raise ValueError(f"Fixed shape trace differs from {first_path} at {path}")
            completed += candidate["outcome"] == "cap" and len(candidate["output_ids"]) == 64
        fixed_output_completions[path.parent.name] = completed
    return {"arms": [path.parent.name for path, _ in fixed], "matched_requests": len(first),
            "max_arrival_disagreement_allowed_seconds": 2e-6, "requested_fixed_output_rows": 64,
            "fixed_output_completions": fixed_output_completions,
            "matching": ["request order", "planned arrival", "input token count", "requested output token count"]}


def helper_receipt(run):
    helper = run["helper"]
    if helper not in ("encoder", "prefill"):
        return None
    arena = run["stage_kv_pools"][-1]
    rank = int(arena["device"].split(":")[-1])
    requests = run["requests"]
    if helper == "encoder":
        busy = union((r["e_gpu_begin"], r["e_end"]) for r in requests if "e_gpu_begin" in r)
        scope = "Encoder GPU work host enclosure" if busy else "No vision work in matched text-only VL"
    else:
        busy = union((r["p_start"], r["p_end"]) for r in requests if r.get("source_p") == rank)
        scope = "Producer P request service host enclosure"
    active = union((step["begin"], step["end"]) for step in run["decode_steps"] if step["assisted"])
    overlap = common_duration(busy, active)
    gaps = [1e3 * (after[0] - before[1]) for before, after in zip(busy, busy[1:])]
    return {
        "role": helper, "device": arena["device"], "weight_bytes": run["allocated_weight_bytes"].get(str(rank), 0),
        "pool_bytes": arena["allocated_bytes"],
        "peak_active_page_bytes": arena["allocated_bytes"] * arena["peak_active_pages"] // arena["allocated_pages"],
        "main_role_interval_basis": scope, "main_role_interior_idle_gaps": len(gaps),
        "main_role_interior_idle_gap_p50_ms": percentile(gaps, 50),
        "main_role_interior_idle_gap_p95_ms": percentile(gaps, 95),
        "assisted_decode_overlap_with_main_role_fraction": overlap / duration(active) if active else None,
        "only_observed_enclosures": True,
    }


def descriptor(path, run):
    with path.open("rb") as data:
        source_hash = hashlib.file_digest(data, "sha256").hexdigest()
    requests = run["requests"]
    return {
        "cohort_sha256": source_hash, "run": path.parent.name, "helper_details": helper_receipt(run),
        "d_pool_by_device": run["stage_kv_pools"][:-1] if run["helper"] != "none" else run["stage_kv_pools"],
        "initial_d_logical_bytes": sum(r.get("p_to_d_logical_bytes", 0) for r in requests),
        "initial_helper_peer_logical_bytes": sum(r.get("p_to_helper_logical_bytes", 0) for r in requests),
        "initial_helper_retained_logical_bytes": sum(r.get("helper_retained_bytes", 0) for r in requests),
        "recurring_logical_link_bytes": sum(s.get("D_helper_logical_bytes", 0) for s in run["decode_steps"]),
        "peak_process_allocated_bytes": run["peak_process_allocated_bytes"],
        "actual_assisted_requests": sum(bool(r.get("helper_prompt_rows")) for r in requests),
    }


def draw(summary_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = {
        "txt_matched_prefill": "Qwen text\nFixed 64\nP-assisted",
        "mm_text_encoder": "Qwen-VL text\nFixed 64\nE-assisted",
        "mm_image_replay_encoder": "Qwen-VL images\nFixed 64\nE-assisted",
        "txt_real_prefill": "Qwen LongBench\nNatural EOS\nP-assisted",
    }
    paired = raw(summary_path)["paired"]
    pairs = []
    for name in arms:
        match, = (row for row in paired if row["arm"] == name or row["arm"].startswith(name + "_"))
        pairs.append(match)
    x = list(range(len(arms)))
    left_centers = [i - .19 for i in x]
    right_centers = [i + .19 for i in x]
    fig, (latency, goodput) = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    latency.bar(left_centers, [1 + row["latency_change_fraction"]["tpot_ms"] for row in pairs],
                .36, color="#375a7f", label="TPOT p95")
    latency.bar(right_centers, [1 + row["latency_change_fraction"]["itl_ms"] for row in pairs],
                .36, color="#d17a56", label="ITL p99")
    latency.axhline(1, color="0.25", linewidth=.8, linestyle="--")
    latency.set_ylabel("Assisted / local latency (lower is better)")
    latency.legend(frameon=False)
    goodput.bar(left_centers, [row["goodput"][0] for row in pairs], .36, color="#375a7f", label="Local")
    goodput.bar(right_centers, [row["goodput"][1] for row in pairs], .36, color="#d17a56", label="Assisted")
    goodput.set_ylabel("Joint SLO goodput (requests / s)")
    goodput.legend(frameon=False)
    for axis in (latency, goodput):
        axis.set_xticks(x, list(arms.values()))
        axis.tick_params(axis="x", labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired within-workload controls on three H200s at 1 request / s")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--natural-image-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    if bool(args.summary) != bool(args.figure):
        parser.error("Supply both the paired summary and figure path to render the figure")
    text_and_fixed = [(path, raw(path)) for path in args.runs]
    image = [(path, raw(path)) for path in args.natural_image_runs]
    receipt = {"fixed_shapes": matched_fixed_shape(text_and_fixed),
               "fixed_and_text_runs": [descriptor(path, run) for path, run in text_and_fixed],
               "natural_image_runs": [descriptor(path, run) for path, run in image],
               "bytes_scope": "Logical KV payload and physical arena allocation. Cross-device traffic, CUDA "
                              "synchronization and memcpy durations have separate measurement boundaries."}
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    if args.figure:
        draw(args.summary, args.figure)


if __name__ == "__main__":
    main()
