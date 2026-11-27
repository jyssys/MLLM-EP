"""Build an audit and plots from independent native Qwen text-PD runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt


PRIMARY_COUNTERS = (
    "vllm:prefix_cache_hits_total", "vllm:external_prefix_cache_hits_total", "vllm:num_preemptions_total",
    "vllm:kv_offload_total_bytes_total:CPU_to_GPU", "vllm:kv_offload_total_bytes_total:GPU_to_CPU",
    "vllm:kv_offload_total_time_total:CPU_to_GPU", "vllm:kv_offload_total_time_total:GPU_to_CPU",
)
RATIO_METRICS = ("TTFT_p95_ms", "ITL_p99_ms", "TPOT_p95_ms", "E2E_p95_ms", "joint_goodput_req_s")


def collect(label, root):
    run = {"label": label, "name": root.name}
    for filename in ("complete", "failure"):
        terminal = root / f"state/{filename}.json"
        if terminal.is_file():
            run["terminal"] = rt.read(terminal)
    metrics = root / "analysis/control_metrics.json"
    if not metrics.is_file():
        run["evidence"] = "FAILED_EXECUTION" if "terminal" in run else "NOT_RUN"
        return run

    work = rt.read(metrics)
    client = work["client"]
    run["evidence"] = ("MEASURED_ONLINE" if client["primary_execution_valid"] and
                       run.get("terminal", {}).get("status") == "COMPLETE" else "FAILED_EXECUTION")
    run["requests"] = client["planned_count"]
    run["completed"] = client["completed_count"]
    run["metrics"] = {key: client[key] for key in ("TTFT_p50_ms", "TTFT_p95_ms", "TTFT_p99_ms", "TPOT_p95_ms",
                                                    "ITL_p99_ms", "completed_requests_per_second",
                                                    "total_output_tokens", "duration_seconds")}
    run["metrics"].update(E2E_p95_ms=work["E2E_p95_ms"], joint_goodput_req_s=work["joint_slo_goodput_req_s"],
                           joint_slo=client["categories"].get("JOINT_PASS", 0))
    run["phases"] = work["phases"]
    run["per_role_counters"] = {role: {key: values[key] for key in PRIMARY_COUNTERS if key in values}
                                for role, values in work["prometheus_deltas"].items()}
    memory_receipt = root / "state/workers_after_drain.json"
    if memory_receipt.is_file():
        run["worker_memory"] = {label: {key: worker[key] for key in
                                ("allocated_bytes", "reserved_bytes", "maximum_allocated_bytes", "device_free_bytes",
                                 "model_parameter_logical_bytes", "unique_KV_storage_bytes", "numerics")}
                                for label, worker in rt.read(memory_receipt).items()}
    run["peer_storage"] = {file.parents[1].name: rt.read(file) for file in sorted(root.glob("prefill*/state/peer_storage.json"))}
    run["source_sha256"] = rt.read(root / "state/environment.json")["source_sha256"]
    run["raw_hashes"] = {str(path.relative_to(root)): rt.sha(path) for path in (
        root / "data/roster.json", root / "configs/measured_arrival.json", root / "raw/measured_requests.jsonl",
        root / "raw/proxy.jsonl", root / "raw/metrics.json", metrics)}
    outputs = work["exact_output_sha256"]
    repeated = [(key.replace("repeat/", "cold/", 1), value) for key, value in outputs.items()
                if key.startswith("repeat/")]
    run["exact_output_repeat_of_cold"] = {"observed_repeats": len(repeated),
                                          "exact": sum(outputs.get(key) == value for key, value in repeated)}
    run["output_sha256"] = outputs
    return run


def draw(runs, output):
    from matplotlib import pyplot as plt

    measured = [run for run in runs if run["evidence"] == "MEASURED_ONLINE"]
    if not measured:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.6), layout="constrained")
    labels = [run["label"] for run in measured]
    colors = ["#175574", "#209287", "#6a9497", "#c08433", "#6467a5"]
    for ax, key, label in zip(axes.flat, ("TTFT_p95_ms", "TPOT_p95_ms", "ITL_p99_ms", "joint_goodput_req_s"),
                              ("TTFT p95 (ms)", "TPOT p95 (ms)", "ITL p99 (ms)", "Joint SLO-goodput (req/s)"), strict=True):
        ax.barh(labels, [run["metrics"][key] for run in measured], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Qwen2.5-7B PD on three H200s · Matched LongBench trace")
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def timings(label, root):
    source = root / "raw/measured_requests.jsonl"
    if not source.is_file():
        return
    planned = rt.read(root / "configs/measured_arrival.json")["trace"]
    proxy = {row["request_id"]: row for row in map(json.loads, (root / "raw/proxy.jsonl").read_text().splitlines())}
    for row in map(json.loads, source.read_text().splitlines()):
        mark = proxy.get(row["runtime_request_id"], {})
        p_http = (mark["P_ready"] - mark["P_submit"]) * 1000 if "P_ready" in mark else None
        yield {"arm": label, "request_id": row["request_id"], "status": row["status"], "index": row["index"],
               "phase": row["stratum"], "planned_offset_s": planned[row["index"]]["offset_seconds"],
               "input_tokens": row["input_tokens_manifest"], "output_tokens": row.get("output_tokens"),
               "P_HTTP_ms": p_http, "P_port": mark.get("prefill_port"), "D_port": mark.get("decode_port"),
               "dispatch_lateness_ms": row["dispatch_lateness"] * 1000,
               "output_offsets_ms": [(value - row["scheduled_arrival"]) * 1000 for value in row["token_times"]],
               "E2E_ms": row.get("E2E_ms"), "error": row.get("error")}


def compare_epd(runs, arm, path, reference, candidate):
    match = [run for run in runs if run["label"] == arm and run["evidence"] == "MEASURED_ONLINE"]
    if not match:
        raise ValueError("The requested text peer arm has no online result")
    text, = match
    native = {run["run"]: run["measurements"][0] for run in rt.read(path)["runs"] if run["status"] == "COMPLETE"}
    own, peer = native[reference], native[candidate]
    for key in ("manifest_sha256", "arrival_sha256"):
        if own[key] != peer[key]:
            raise ValueError(f"Nonpaired EPD runs differ in {key}")
    ordered_digests = {}
    for result in (own, peer):
        source = path.parent / result["run"] / "raw" / f"{result['cohort']}_requests.jsonl"
        if rt.sha(source) != result["raw_sha256"]:
            raise ValueError(f"EPD result source hash differs for {result['run']}")
        records = sorted(map(json.loads, source.read_text().splitlines()), key=lambda row: row["index"])
        if len(records) != result["scheduled"] or [row["index"] for row in records] != list(range(len(records))):
            raise ValueError("Missing or duplicated EPD arrival index")
        ordered_digests[result["run"]] = hashlib.sha256("\n".join(row["request_id"] for row in records).encode()).hexdigest()
    if len(set(ordered_digests.values())) != 1:
        raise ValueError("Nonpaired EPD arrival-ordered request IDs")
    local = dict(own["latency"], joint_goodput_req_s=own["joint_goodput_req_s"])
    remote = dict(peer["latency"], joint_goodput_req_s=peer["joint_goodput_req_s"])
    return {"scope": "Relative effect within each model and cohort. Different prompts, lengths and memory pressures.",
            "text_peer_arm": arm, "mm_epd_reference": reference, "mm_epd_peer": candidate, "epd_sha256": rt.sha(path),
            "epd_arrival_ordered_request_ids_sha256": ordered_digests,
            "text_PD_peer_over_local": text["relative_metrics_vs_reference"],
            "MM_EPD_peer_over_local": {key: remote[key] / local[key] for key in RATIO_METRICS if local[key]}}


def draw_comparison(summary, output):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    metrics = ("TTFT_p95_ms", "TPOT_p95_ms", "ITL_p99_ms", "joint_goodput_req_s")
    for index, (key, label, color) in enumerate((
            ("text_PD_peer_over_local", "Text PD · P peer vs. P-local", "#22867b"),
            ("MM_EPD_peer_over_local", "Image EPD · E peer vs. P-local", "#284e86"))):
        ax.bar([i + (index - .5) * .34 for i in range(len(metrics))],
               [100 * (summary[key][m] - 1) for m in metrics], .32, color=color, label=label)
    ax.axhline(0, color="#666666", linewidth=.7)
    ax.set_xticks(range(len(metrics)), ["TTFT p95", "TPOT p95", "ITL p99", "Joint SLO-goodput"])
    ax.set_ylabel("Change vs. equal-cache-size local arm (%)")
    ax.set_title("Peer KV storage within each controlled cohort")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=run_directory")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--timings", type=Path, help="Streaming timing evidence without prompt or output text")
    parser.add_argument("--epd-comparison", nargs=4, metavar=("SUMMARY", "REFERENCE", "CANDIDATE", "TEXT_PEER_ARM"))
    parser.add_argument("--cross-figure", type=Path)
    args = parser.parse_args()
    arms = [(label, Path(path)) for label, path in (arg.split("=", 1) for arg in args.arm)]
    runs = [collect(label, path) for label, path in arms]
    references = [run for run in runs if run["label"] == args.reference and run["evidence"] == "MEASURED_ONLINE"]
    if references:
        reference, = references
        for run in runs:
            if run["evidence"] == "MEASURED_ONLINE":
                for name in ("data/roster.json", "configs/measured_arrival.json"):
                    if run["raw_hashes"][name] != reference["raw_hashes"][name]:
                        raise ValueError(f"Nonpaired {name} for {run['label']}")
                outputs = reference["output_sha256"]
                run["exact_output_vs_reference"] = sum(outputs.get(key) == value for key, value in run["output_sha256"].items())
                run["relative_metrics_vs_reference"] = {k: run["metrics"][k] / reference["metrics"][k]
                                                         for k in RATIO_METRICS if reference["metrics"][k]}
    for run in runs:
        run.pop("output_sha256", None)
    result = {"scope": "Finite-cohort fixed-output pilots. Per-arm start counts and recorded failures apply.",
              "reference": args.reference, "arms": runs}
    if args.epd_comparison:
        path, epd_reference, epd_peer, text_peer = args.epd_comparison
        result["relative_PD_and_EPD"] = compare_epd(runs, text_peer, Path(path), epd_reference, epd_peer)
        if args.cross_figure:
            draw_comparison(result["relative_PD_and_EPD"], args.cross_figure)
    rt.write(args.out, result)
    if args.csv:
        records = [{"arm": run["label"], "evidence": run["evidence"], **run.get("metrics", {})} for run in runs]
        with args.csv.open("x", newline="") as stream:
            fields = list(dict.fromkeys(key for row in records for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    if args.figure:
        draw(runs, args.figure)
    if args.timings:
        with args.timings.open("x") as stream:
            for label, path in arms:
                for row in timings(label, path):
                    stream.write(json.dumps(row, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
