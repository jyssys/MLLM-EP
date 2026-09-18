#!/usr/bin/env python3
"""Analyze exact branch drift and Temporal-Edge reuse policies on 16 requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel
from scripts.analyze_low_utility_straggler import (
    build_current_unique, communication_latency, subtract_removed_hist,
)
from virtual_ep.comm_model import CommunicationScenario


NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_BYTES = 2048 * 2
EPS = (4, 8)
SELECTED_LAYERS = (1, 5, 10, 14, 19)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        **{f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 90, 95, 99)},
        "max": float(values.max(initial=0)),
    }


def load_edges(paths):
    keys = None
    chunks = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            current = sorted(key for key in source.files if key.startswith("edge_"))
            if keys is None:
                keys = current
                chunks = {key: [] for key in keys}
            if current != keys:
                raise ValueError(f"edge schema mismatch in {path}")
            for key in keys:
                chunks[key].append(source[key])
    return {key.removeprefix("edge_"): np.concatenate(values) for key, values in chunks.items()}


def policy_grid(edges):
    discovery = edges["request_id"] < 8
    heldout = ~discovery
    safe = edges["raw_output_relative_l2"] <= .05
    candidates = []
    for tau_h in (.02, .05, .10, .20, .40, .80):
        for tau_w in (.02, .05, .10, .20, .50, 1.0):
            for min_age in (2, 3, 4):
                predicted = (
                    (edges["input_hidden_relative_l2"] <= tau_h)
                    & (edges["weight_relative_drift"] <= tau_w)
                    & (edges["route_age"] >= min_age)
                )
                selected = predicted & discovery
                if not selected.any():
                    continue
                precision = float(safe[selected].mean())
                recall = float(np.count_nonzero(selected & safe) / max(1, np.count_nonzero(discovery & safe)))
                candidates.append({
                    "tau_h": tau_h, "tau_w": tau_w, "min_age": min_age,
                    "precision": precision, "recall": recall,
                    "coverage": float(selected.sum() / discovery.sum()),
                    "selected": int(selected.sum()),
                })
    eligible = [row for row in candidates if row["precision"] >= .90]
    chosen = max(eligible or candidates, key=lambda row: (row["coverage"], row["precision"]))
    predicted = (
        (edges["input_hidden_relative_l2"] <= chosen["tau_h"])
        & (edges["weight_relative_drift"] <= chosen["tau_w"])
        & (edges["route_age"] >= chosen["min_age"])
    )
    conservative = predicted & (edges["route_age"] % 2 == 0)
    result = {"selection": chosen, "safe_definition": "raw expert-output relative-L2 <= 5%"}
    for name, split in (("discovery", discovery), ("heldout", heldout)):
        result[name] = {}
        for label, mask in (("E3", predicted), ("E4", conservative)):
            selected = mask & split
            result[name][label] = {
                "records": int(split.sum()), "selected": int(selected.sum()),
                "coverage_of_stay": float(selected.sum() / split.sum()),
                "precision_for_safe": float(safe[selected].mean()) if selected.any() else None,
                "recall_of_safe": float(np.count_nonzero(selected & safe) / max(1, np.count_nonzero(split & safe))),
            }
    return result, predicted, conservative


def stability_summary(edges, gate):
    summary = {
        "records": int(len(edges["request_id"])),
        "requests": sorted(np.unique(edges["request_id"]).astype(int).tolist()),
        "layers": sorted(np.unique(edges["layer_id"]).astype(int).tolist()),
        "raw_output_cosine": distribution(edges["raw_output_cosine"]),
        "raw_output_relative_l2": distribution(edges["raw_output_relative_l2"]),
        "raw_output_norm_ratio": distribution(edges["raw_output_norm_ratio"]),
        "weighted_R0_relative_l2": distribution(edges["weighted_r0_relative_l2"]),
        "weighted_R1_relative_l2": distribution(edges["weighted_r1_relative_l2"]),
        "router_weight_absolute_drift": distribution(edges["weight_abs_drift"]),
        "router_weight_relative_drift": distribution(edges["weight_relative_drift"]),
        "input_hidden_relative_l2": distribution(edges["input_hidden_relative_l2"]),
        "post_moe_relative_l2": distribution(edges["post_moe_relative_l2"]),
        "oracle_thresholds": {}, "by_layer": {}, "by_route_age": {},
        "practical_gate": gate,
    }
    for threshold in (.01, .02, .05, .10):
        summary["oracle_thresholds"][str(threshold)] = {
            "raw_output_reusable_fraction": float(np.mean(edges["raw_output_relative_l2"] <= threshold)),
            "weighted_R0_reusable_fraction": float(np.mean(edges["weighted_r0_relative_l2"] <= threshold)),
            "weighted_R1_reusable_fraction": float(np.mean(edges["weighted_r1_relative_l2"] <= threshold)),
        }
    for layer in SELECTED_LAYERS:
        selected = edges["layer_id"] == layer
        summary["by_layer"][str(layer)] = {
            "records": int(selected.sum()),
            "raw_relative_l2": distribution(edges["raw_output_relative_l2"][selected]),
            "input_relative_l2": distribution(edges["input_hidden_relative_l2"][selected]),
        }
    ages = np.minimum(edges["route_age"], 8)
    for age in range(2, 9):
        selected = ages == age
        summary["by_route_age"]["8plus" if age == 8 else str(age)] = {
            "records": int(selected.sum()),
            "raw_relative_l2": distribution(edges["raw_output_relative_l2"][selected]) if selected.any() else None,
        }
    return summary


def balanced_sources(count, ep):
    base, remainder = divmod(int(count), ep)
    sizes = np.full(ep, base, dtype=np.int64)
    sizes[:remainder] += 1
    return np.repeat(np.arange(ep, dtype=np.int8), sizes)


def assemble_system_trace(paths, edges, practical, conservative):
    full_hist, routes, masks, physical = [], [], [], []
    unique = {ep: [] for ep in EPS}
    source = {ep: [] for ep in EPS}
    request_vector, block_vector, iteration_vector, layer_vector = [], [], [], []
    removals = {name: [] for name in ("E1_identity", "E2_l2_1", "E2_l2_2", "E2_l2_5", "E2_l2_10", "E3_practical", "E4_conservative")}
    edge_lookup = {}
    for ordinal in range(len(edges["request_id"])):
        key = tuple(int(edges[field][ordinal]) for field in (
            "request_id", "block_id", "iteration_id", "layer_id",
            "token_position", "current_slot",
        ))
        edge_lookup[key] = ordinal

    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            metadata = json.loads(str(trace["metadata_json"]))
            request_id = int(metadata["request_id"])
            layer_ids = tuple(metadata.get("selected_routed_layers", SELECTED_LAYERS))
            # NPZ indexing decompresses an entire member.  Materialize each
            # member once per request rather than once per layer/refinement.
            block_ids = trace["block_id"]
            iteration_ids = trace["iteration_id"]
            physical_rows_vector = trace["physical_rows"]
            routes_array = trace["routes"]
            masked_array = trace["masked"]
            hist_array = trace["hist_s0"]
            unique_array = {ep: trace[f"u_s0_ep{ep}"] for ep in EPS}
            # routes contains all 19 routed layers, whose IDs are 1..19.
            all_layer_ids = tuple(range(1, routes_array.shape[1] + 1))
            for step in range(len(block_ids)):
                for layer_slot, layer_id in enumerate(all_layer_ids):
                    full_hist.append(hist_array[step, layer_slot].astype(np.int32))
                    routes.append(routes_array[step, layer_slot].astype(np.int16))
                    masks.append(masked_array[step].astype(np.bool_))
                    physical_rows = int(physical_rows_vector[step])
                    physical.append(physical_rows)
                    request_vector.append(request_id)
                    block_vector.append(int(block_ids[step]))
                    iteration_vector.append(int(iteration_ids[step]))
                    layer_vector.append(layer_id)
                    for ep in EPS:
                        unique[ep].append(unique_array[ep][step, layer_slot].astype(np.int32))
                        source[ep].append(balanced_sources(physical_rows, ep)[-32:])
                    policy_masks = {name: np.zeros((32, TOP_K), dtype=np.bool_) for name in removals}
                    if layer_id in layer_ids:
                        for position in range(32):
                            for slot in range(TOP_K):
                                key = (request_id, int(block_ids[step]),
                                       int(iteration_ids[step]), layer_id,
                                       position, slot)
                                edge_index = edge_lookup.get(key)
                                if edge_index is None:
                                    continue
                                policy_masks["E1_identity"][position, slot] = True
                                drift = float(edges["raw_output_relative_l2"][edge_index])
                                for threshold, name in ((.01, "E2_l2_1"), (.02, "E2_l2_2"),
                                                        (.05, "E2_l2_5"), (.10, "E2_l2_10")):
                                    policy_masks[name][position, slot] = drift <= threshold
                                policy_masks["E3_practical"][position, slot] = practical[edge_index]
                                policy_masks["E4_conservative"][position, slot] = conservative[edge_index]
                    for name in removals:
                        removals[name].append(policy_masks[name])
    return {
        "full_hist": np.stack(full_hist), "routes": np.stack(routes),
        "masks": np.stack(masks), "physical_rows": np.asarray(physical),
        "unique": {ep: np.stack(value) for ep, value in unique.items()},
        "source": {ep: np.stack(value) for ep, value in source.items()},
        "request_id": np.asarray(request_vector), "block_id": np.asarray(block_vector),
        "iteration_id": np.asarray(iteration_vector), "layer_id": np.asarray(layer_vector),
        "removals": {name: np.stack(value) for name, value in removals.items()},
    }


def stage(hist, unique, ep, model, communication):
    per_rank = NUM_EXPERTS // ep
    rank_times = np.column_stack([
        model.predict_batch(hist[:, rank * per_rank:(rank + 1) * per_rank])
        for rank in range(ep)
    ])
    dispatch, outgoing, _incoming = communication_latency(communication.dispatch, unique)
    combine, _out, _in = communication_latency(
        communication.combine, np.transpose(unique, (0, 2, 1))
    )
    expert = rank_times.max(axis=1)
    mean = rank_times.mean(axis=1)
    second = np.sort(rank_times, axis=1)[:, -2]
    wait = np.divide(
        ep * expert - rank_times.sum(axis=1), ep * expert,
        out=np.zeros_like(expert), where=expert > 0,
    )
    remote = unique.copy(); diagonal = np.arange(ep); remote[:, diagonal, diagonal] = 0
    return {
        "dispatch_ms": float(dispatch.sum()), "expert_ms": float(expert.sum()),
        "combine_ms": float(combine.sum()), "stage_ms": float((dispatch + expert + combine).sum()),
        "remote_logical_bytes": int(remote.sum()) * HIDDEN_BYTES,
        "max_mean": float(np.mean(expert / np.maximum(mean, 1e-12))),
        "max_second": float(np.mean(expert / np.maximum(second, 1e-12))),
        "cv": float(np.mean(rank_times.std(axis=1) / np.maximum(mean, 1e-12))),
        "wait_fraction": float(wait.mean()),
    }


def system_policies(data, models, communication):
    output = {}
    all_current = np.ones(data["masks"].shape, dtype=np.bool_)
    zero = np.zeros(data["routes"].shape, dtype=np.bool_)
    for ep in EPS:
        baseline = stage(data["full_hist"], data["unique"][ep].copy(), ep,
                         models[ep], communication)
        current_base = build_current_unique(
            data["routes"], all_current, data["source"][ep], ep, zero
        )
        fixed = data["unique"][ep] - current_base
        policies = {"E0_vanilla": baseline}
        for name, removed in data["removals"].items():
            histogram = subtract_removed_hist(
                data["full_hist"], data["routes"], all_current, removed
            )
            unique = fixed + build_current_unique(
                data["routes"], all_current, data["source"][ep], ep, removed
            )
            result = stage(histogram, unique, ep, models[ep], communication)
            result.update({
                "fresh_pair_reduction_of_adjacent_live_mask_percent": 100 * removed.sum() / max(1, data["masks"].sum() * TOP_K * 19),
                "whole_physical_route_reduction_percent": 100 * removed.sum() / data["full_hist"].sum(),
                "stage_gain_percent": 100 * (1 - result["stage_ms"] / baseline["stage_ms"]),
                "expert_gain_percent": 100 * (1 - result["expert_ms"] / baseline["expert_ms"]),
                "remote_byte_reduction_percent": 100 * (1 - result["remote_logical_bytes"] / baseline["remote_logical_bytes"]),
                "removed_routes": int(removed.sum()),
                "peak_cache_bytes": int(removed.reshape(len(removed), -1).sum(axis=1).max(initial=0) * HIDDEN_BYTES),
            })
            policies[name] = result
        output[f"ep{ep}"] = {
            "label": f"SIMULATED-EP{ep}-EP2-CALIBRATED",
            "policies": policies,
        }
    return output


def figures(edges, stability, systems, structural, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(edges["raw_output_relative_l2"], bins=100, range=(0, 2), density=True, alpha=.65, label="raw/R1")
    plt.hist(edges["weighted_r0_relative_l2"], bins=100, range=(0, 2), density=True, histtype="step", label="R0 weighted")
    plt.xlabel("adjacent expert-branch relative L2"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "05_branch_output_stability_hist.png", dpi=180); plt.close()

    ages = np.minimum(edges["route_age"], 8)
    x = np.arange(2, 9)
    y = [np.median(edges["raw_output_relative_l2"][ages == age]) for age in x]
    plt.figure(figsize=(6, 4)); plt.plot(x, y, marker="o"); plt.xticks(x, ["2","3","4","5","6","7","8+"])
    plt.xlabel("route age"); plt.ylabel("median raw output relative L2"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(output / "06_output_drift_vs_route_age.png", dpi=180); plt.close()

    sample = np.linspace(0, len(edges["request_id"]) - 1, min(100_000, len(edges["request_id"]))).astype(int)
    plt.figure(figsize=(6, 4)); plt.hexbin(
        edges["input_hidden_relative_l2"][sample], edges["raw_output_relative_l2"][sample],
        gridsize=70, bins="log", mincnt=1,
    ); plt.xlabel("input hidden relative L2"); plt.ylabel("expert output relative L2"); plt.colorbar(label="log count")
    plt.tight_layout(); plt.savefig(output / "07_input_vs_output_drift.png", dpi=180); plt.close()

    thresholds = list(stability["oracle_thresholds"])
    plt.figure(figsize=(6, 4)); plt.plot(
        [100*float(x) for x in thresholds],
        [100*stability["oracle_thresholds"][x]["raw_output_reusable_fraction"] for x in thresholds], marker="o"
    ); plt.xlabel("allowed output relative L2 (%)"); plt.ylabel("STAY routes reusable (%)"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(output / "08_reuse_coverage_vs_error.png", dpi=180); plt.close()

    names = ["E1_identity", "E2_l2_5", "E3_practical", "E4_conservative"]
    plt.figure(figsize=(7, 4))
    for ep in EPS:
        plt.plot(names, [systems[f"ep{ep}"]["policies"][name]["stage_gain_percent"] for name in names], marker="o", label=f"EP{ep}")
    plt.ylabel("whole routed-stage gain (%)"); plt.xticks(rotation=20); plt.legend(); plt.tight_layout()
    plt.savefig(output / "09_reuse_policy_stage_gain.png", dpi=180); plt.close()

    source = [structural["ep8"]["source_side"]["stage_gain_percent"]]
    owner = [structural["ep8"]["owner_side"]["stage_gain_percent"]]
    plt.figure(figsize=(5, 4)); plt.bar(["source cache", "owner cache"], source + owner)
    plt.ylabel("EP8 E1 whole-stage gain (%)"); plt.tight_layout(); plt.savefig(output / "10_cache_placement_gain.png", dpi=180); plt.close()

    policy = systems["ep8"]["policies"]
    plt.figure(figsize=(6, 4))
    plt.scatter([policy[name]["peak_cache_bytes"] / 2**20 for name in names],
                [policy[name]["stage_gain_percent"] for name in names])
    for name in names:
        plt.annotate(name, (policy[name]["peak_cache_bytes"] / 2**20, policy[name]["stage_gain_percent"]))
    plt.xlabel("peak cached branch output (MiB/invocation)"); plt.ylabel("EP8 stage gain (%)"); plt.tight_layout()
    plt.savefig(output / "11_cache_memory_vs_gain.png", dpi=180); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=Path("artifacts/dllm_native_ep_discovery/branch_trace16/traces"))
    parser.add_argument("--structural", type=Path, default=Path("artifacts/dllm_native_ep_discovery/structural_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dllm_native_ep_discovery/branch_stability_summary.json"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    paths = sorted(args.trace_dir.glob("request_*.npz"))
    if len(paths) != 16:
        raise RuntimeError(f"expected 16 branch traces, found {len(paths)}")
    edges = load_edges(paths)
    gate, practical, conservative = policy_grid(edges)
    stability = stability_summary(edges, gate)
    data = assemble_system_trace(paths, edges, practical, conservative)
    models = {
        4: BatchComputeModel(Path("artifacts/route_pruning_quality/20260918/compute_model/ep4_grouped_mm_route_pruning.csv")),
        8: BatchComputeModel(Path("artifacts/route_pruning_quality/20260918/compute_model/ep8_grouped_mm_route_pruning.csv")),
    }
    communication = CommunicationScenario.load(
        Path("artifacts/virtual_ep/20260916_215010/communication/model.json"),
        "ep2_calibrated_base",
    )
    systems = system_policies(data, models, communication)
    structural = json.loads(args.structural.read_text())["track_a"]["systems"]
    diagnostic = {}
    for generation_path in sorted(args.trace_dir.parent.glob("generations_worker*.jsonl")):
        for line in generation_path.read_text().splitlines():
            row = json.loads(line)
            diagnostic[int(row["sample_id"])] = row
    reference = {}
    for line in Path("artifacts/reproduction/gsm8k_samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        if int(row["sample_id"]) in diagnostic:
            reference[int(row["sample_id"])] = row
    collector_parity = {
        "requests": len(diagnostic),
        "nfe_match": sum(diagnostic[key]["nfe"] == reference[key]["nfe"] for key in diagnostic),
        "correctness_match": sum(diagnostic[key]["correct"] == reference[key]["correct"] for key in diagnostic),
        "remaining_mask_total": sum(diagnostic[key]["remaining_masks"] for key in diagnostic),
        "note": "diagnostic collector returned the untouched official MoE result",
    }
    summary = {
        "evidence": {
            "requests": list(range(16)), "discovery": list(range(8)),
            "heldout": list(range(8, 16)), "selected_layers": SELECTED_LAYERS,
            "trace_kind": "actual duplicate-compute branch vectors; model output unchanged",
        },
        "stability": stability, "systems": systems,
        "collector_parity": collector_parity,
        "quality_run": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    args.reports.mkdir(exist_ok=True)
    (args.reports / "temporal_edge_ep_output_stability.md").write_text(f"""# Temporal-Edge EP: actual expert-branch output stability

This targeted trace re-executed current-block branches at routed layers {SELECTED_LAYERS} on 16 fixed GSM8K requests (0--7 discovery, 8--15 held out). The official MoE result was returned unchanged; these timings are not performance measurements.

There are {stability['records']:,} exact adjacent live-MASK STAY branches. Raw output relative-L2 P50/P90/P99 is {stability['raw_output_relative_l2']['p50']:.4f}/{stability['raw_output_relative_l2']['p90']:.4f}/{stability['raw_output_relative_l2']['p99']:.4f}; cosine P50 is {stability['raw_output_cosine']['p50']:.4f}. Only {100*stability['oracle_thresholds']['0.05']['raw_output_reusable_fraction']:.3f}% are within 5% relative-L2, so identity persistence does not imply numerical reuse safety. R1 (current-weight reweighting) removes router-weight drift but cannot remove raw branch drift; R0 is worse (P50 L2 {stability['weighted_R0_relative_l2']['p50']:.4f}).

The observational collector matched the frozen reference on NFE for {collector_parity['nfe_match']}/{collector_parity['requests']} requests and correctness for {collector_parity['correctness_match']}/{collector_parity['requests']}, with {collector_parity['remaining_mask_total']} remaining MASK tokens.

The discovery-selected current-observable E3 thresholds are `{gate['selection']}`. Held-out E3 coverage/precision/recall is `{gate['heldout']['E3']}`; E4 is `{gate['heldout']['E4']}`. Full distributions, layer/age breakdowns and all four oracle thresholds are in `{args.output}`.
""")
    (args.reports / "temporal_edge_ep_oracle.md").write_text(
        (args.reports / "temporal_edge_ep_oracle.md").read_text()
        + f"""

## Actual-output policies on targeted 16-request trace

E1/E2/E3/E4 in this table are applied only to the five layers with measured branch vectors; the GSM8K-128 all-19-layer E1 structural upper bound remains in the first table.

| policy | EP4 stage gain | EP8 stage gain | EP8 whole-route reduction |
|---|---:|---:|---:|
""" + "\n".join(
            f"| {name} | {systems['ep4']['policies'][name]['stage_gain_percent']:.4f}% | {systems['ep8']['policies'][name]['stage_gain_percent']:.4f}% | {systems['ep8']['policies'][name]['whole_physical_route_reduction_percent']:.4f}% |"
            for name in ("E1_identity", "E2_l2_1", "E2_l2_2", "E2_l2_5", "E2_l2_10", "E3_practical", "E4_conservative")
        ) + "\n\nE2 uses future branch outputs and is an oracle. E3/E4 use current-observable input and router drift selected on requests 0--7 and evaluated once on 8--15. All gains are simulated routed-MoE stage gains, not E2E latency.\n"
    )
    (args.reports / "temporal_edge_ep_quality.md").write_text("""# Temporal-Edge EP quality gate

No semantic reuse rollout was run. The impossible E1 whole-stage upper bound is already far below 5%, and actual-output-safe E2/E3/E4 coverage is smaller. Per the staged gate, spending quality risk on this mechanism is not justified. The 16-request diagnostic collector itself returned the unmodified official MoE output; it is observational, not a reuse-quality result.
""")
    figures(edges, stability, systems, structural,
            args.reports / "figures/dllm_native_ep_discovery")
    print(json.dumps({
        "records": stability["records"],
        "safe_5pct": stability["oracle_thresholds"]["0.05"]["raw_output_reusable_fraction"],
        "heldout_e3": gate["heldout"]["E3"],
        "ep8_e3_stage_gain": systems["ep8"]["policies"]["E3_practical"]["stage_gain_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
