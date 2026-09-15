"""Observer-only decision/EP atlas; never substitute event sums for clean E2E."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phase(progress):
    return "early" if progress < 0.25 else "middle" if progress < 0.75 else "late"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result = ROOT / "results" / args.label
    shape_by = defaultdict(list)
    timing_by = defaultdict(list)
    for rank in range(4):
        for row in read_jsonl(result / "shape_trace" / f"shape_rank{rank}.jsonl"):
            if row["request_id"].startswith("measured_") and any(row.get("remaining_before", [])):
                shape_by[(row["request_id"], row["invocation"])].append(row)
        for row in read_jsonl(result / "ep_trace" / f"ep_rank{rank}.jsonl"):
            if row["request_id"].startswith("measured_") and any(row.get("remaining_before", [])):
                timing_by[(row["request_id"], row["invocation"], row["layer"])].append(row)
    shape_rows = []
    for call, key in enumerate(sorted(shape_by, key=lambda item: (item[0], item[1]))):
        records = shape_by[key]
        if len(records) != 4:
            raise RuntimeError(f"expected 4 physical ranks for {key}, got {len(records)}")
        global_m = records[0]["physical_m_global"]
        if any(row["physical_m_global"] != global_m for row in records):
            raise RuntimeError("ranks disagree on physical M")
        owner_load = [sum(row["rank_counts_local_source"][owner] for row in records) for owner in range(4)]
        remote = sum(
            row["rank_counts_local_source"][owner]
            for row in records for owner in range(4) if owner != row["rank"]
        )
        histogram = [sum(row["expert_histogram_local_source"][expert] for row in records) for expert in range(256)]
        if sum(histogram) != 8 * global_m or sum(owner_load) != 8 * global_m:
            raise RuntimeError("exact top-k assignment conservation failed")
        active = [count for count in histogram if count]
        live = records[0]["remaining_before"]
        if len(live) * 32 != global_m:
            raise RuntimeError("decision-live vector does not match physical 32-row blocks")
        live_ratio = sum(live) / global_m
        progress = 1 - live_ratio
        layer_timing = timing_by.get((key[0], key[1], 16), [])
        if len(layer_timing) != 4:
            raise RuntimeError(f"missing layer16 event timing for {key}")
        row = {
            "call": call, "request_id": key[0], "invocation": key[1],
            "phase": phase(progress), "normalized_progress": progress,
            "ready_requests": len(live), "decision_live_rows": sum(live),
            "physical_m": global_m, "topk_pairs": 8 * global_m,
            "remote_pairs": remote, "remote_fraction": remote / (8 * global_m),
            "rank_load": json.dumps(owner_load),
            "rank_load_cv": statistics.pstdev(owner_load) / statistics.mean(owner_load),
            "rank_fanout_mean": statistics.mean(record["rank_fanout_mean"] for record in records),
            "active_experts": len(active),
            "median_m_e": statistics.median(active),
            "tiny_le4_fraction": sum(count <= 4 for count in active) / len(active),
            "router_ms_observer": max(t["router_ms"] for t in layer_timing),
            "dispatch_ms_observer": max(t["dispatch_ms"] for t in layer_timing),
            "expert_ms_observer": max(t["expert_ms"] for t in layer_timing),
            "combine_ms_observer": max(t["combine_ms"] for t in layer_timing),
            "shared_ms_observer": max(t["shared_ms"] for t in layer_timing),
            "gather_ms_observer": max(t["gather_ms"] for t in layer_timing),
        }
        shape_rows.append(row)
    shape_fields = list(shape_rows[0]) if shape_rows else []
    write_csv(result / "TRACE_SHAPE.csv", shape_rows, shape_fields)

    decisions_by = defaultdict(list)
    for row in read_jsonl(result / "decisions.jsonl"):
        if row["request_id"].startswith("measured_"):
            decisions_by[(row["request_id"], row["call"])].append(row)
    decision_rows = []
    for key, records in sorted(decisions_by.items(), key=lambda item: (item[0][0], item[0][1])):
        masked = sum(record["masked_before"] for record in records)
        accepted = sum(record["accepted"] for record in records)
        confidences = [value for record in records for value in record["masked_confidence"]]
        progress = 1 - masked / (32 * len(records))
        decision_rows.append({
            "call": key[1], "request_id": key[0], "phase": phase(progress),
            "normalized_progress": progress, "ready_requests": len(records),
            "masked_before": masked, "accepted": accepted,
            "accepted_per_ready_request": accepted / len(records),
            "confidence_median": statistics.median(confidences) if confidences else "",
            "confidence_p90": sorted(confidences)[int(0.9 * (len(confidences) - 1))] if confidences else "",
            "threshold": records[0]["base_threshold"], "policy": records[0]["policy"],
        })
    write_csv(result / "DECISION_TRACE.csv", decision_rows, list(decision_rows[0]) if decision_rows else [])

    phase_summary = []
    for label in ("early", "middle", "late"):
        cohort = [row for row in shape_rows if row["phase"] == label]
        if not cohort:
            continue
        phase_summary.append({
            "phase": label, "waves": len(cohort),
            "median_live_ratio": statistics.median(1 - row["normalized_progress"] for row in cohort),
            "median_physical_m": statistics.median(row["physical_m"] for row in cohort),
            "median_active_experts": statistics.median(row["active_experts"] for row in cohort),
            "median_tiny_le4": statistics.median(row["tiny_le4_fraction"] for row in cohort),
            "median_remote_fraction": statistics.median(row["remote_fraction"] for row in cohort),
            "median_rank_load_cv": statistics.median(row["rank_load_cv"] for row in cohort),
            "median_rank_fanout": statistics.median(row["rank_fanout_mean"] for row in cohort),
            "median_dispatch_ms_observer": statistics.median(row["dispatch_ms_observer"] for row in cohort),
            "median_expert_ms_observer": statistics.median(row["expert_ms_observer"] for row in cohort),
            "median_combine_ms_observer": statistics.median(row["combine_ms_observer"] for row in cohort),
        })
    report = {
        "label": args.label, "shape_waves": len(shape_rows), "decision_calls": len(decision_rows),
        "phase_summary": phase_summary,
        "evidence_boundary": "GPU event observer-heavy, not clean request E2E or deployable EP-cost policy",
    }
    (result / "TRACE_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].scatter([row["normalized_progress"] for row in shape_rows],
                       [row["physical_m"] for row in shape_rows], s=16)
    axes[0, 0].set(xlabel="Ready-pool refinement progress", ylabel="Physical M")
    axes[0, 1].scatter([row["normalized_progress"] for row in shape_rows],
                       [row["active_experts"] for row in shape_rows], s=16)
    axes[0, 1].set(xlabel="Ready-pool refinement progress", ylabel="Active experts")
    axes[1, 0].scatter([row["normalized_progress"] for row in shape_rows],
                       [row["expert_ms_observer"] for row in shape_rows], s=16)
    axes[1, 0].set(xlabel="Ready-pool refinement progress", ylabel="Layer16 expert event ms (observer)")
    axes[1, 1].scatter([row["normalized_progress"] for row in decision_rows],
                       [row["accepted_per_ready_request"] for row in decision_rows], s=16)
    axes[1, 1].set(xlabel="Decision progress", ylabel="Accepted / ready request")
    fig.suptitle(f"{args.label}: observer-only state/EP trace")
    fig.tight_layout()
    figure = ROOT / "figures" / f"observer_state_ep_{args.label}.png"
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
