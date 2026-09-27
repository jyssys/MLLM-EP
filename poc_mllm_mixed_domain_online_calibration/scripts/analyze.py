from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis import (
    PREDICTORS,
    aggregate_metrics,
    evaluate_stream,
    layer_separation,
    locality,
    materialize,
    overhead,
    paired_bootstrap,
    predict,
    separation,
)
from ..contract import DOMAINS, NUM_EXPERTS, WINDOWS, atomic_json, read_json, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MANIFESTS = ARTIFACTS / "manifests"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _ratio_key(value: float) -> str:
    return f"{value:.2f}"


def _warmup_ids(warmups: dict[str, Any], ratio: float) -> list[str]:
    output: list[str] = []
    for domain in DOMAINS:
        output.extend(warmups["ratios"][_ratio_key(ratio)][domain]["request_ids"])
    return output


def _ordered(stream: list[str], samples: dict[str, dict[str, Any]], split: str, exclude: set[str]) -> list[str]:
    return [item for item in stream if item in samples and samples[item]["analysis_split"] == split and item not in exclude]


def _eval_grid(
    samples: dict[str, dict[str, Any]], stream: list[str], warmups: dict[str, Any], split: str,
    ep_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    curves: dict[str, Any] = {}
    all_metrics: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for ratio in (0.0, 0.01, 0.05, 0.10, 0.20):
        initial = [item for item in _warmup_ids(warmups, ratio) if item in samples]
        evaluation = _ordered(stream, samples, split, set(initial))
        ratio_result: dict[str, Any] = {}
        for mode, continual in (("M1_frozen", False), ("M2_continual", True)):
            ratio_result[mode] = {}
            for predictor in PREDICTORS:
                ratio_result[mode][predictor] = {}
                for window in WINDOWS:
                    rows, events = evaluate_stream(samples, evaluation, initial, predictor, window, ep_size, continual)
                    for row in rows:
                        row.update({"ratio": ratio, "split": split, "mode": mode, "ep_size": ep_size})
                    for event in events:
                        event.update({"ratio": ratio, "split": split, "mode": mode, "ep_size": ep_size})
                    all_metrics.extend(rows)
                    all_events.extend(events)
                    ratio_result[mode][predictor][window] = aggregate_metrics(rows)
        curves[_ratio_key(ratio)] = ratio_result
    return curves, all_metrics, all_events


def _choose_knee(curves: dict[str, Any]) -> dict[str, Any]:
    scores = {
        ratio: values["M1_frozen"]["P_separated"]["1-128"].get("expert_histogram_cosine", 0.0)
        for ratio, values in curves.items() if ratio != "0.00"
    }
    # The pre-frozen gate is defined at 5%/10%; 1% remains a measured sensitivity.
    best = max(scores[ratio] for ratio in ("0.05", "0.10", "0.20"))
    selected = next(ratio for ratio in ("0.05", "0.10", "0.20") if scores[ratio] >= best - 0.005)
    return {"ratio": float(selected), "criterion": "smallest 5/10/20% ratio within 0.005 histogram cosine of discovery best", "scores": scores}


def _rows(metrics: list[dict[str, Any]], ratio: float, mode: str, predictor: str, window: str) -> list[dict[str, Any]]:
    return [
        row for row in metrics
        if row["ratio"] == ratio and row["mode"] == mode and row["predictor"] == predictor and row["window"] == window
    ]


def _comparisons(metrics: list[dict[str, Any]], ratio: float, mode: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    directions = {
        "expert_histogram_cosine": True,
        "expert_weight_histogram_cosine": True,
        "decode_expert_top8_recall": True,
        "active_expert_set_recall": True,
        "rank_distribution_cosine": True,
        "per_rank_load_mae": False,
        "hottest_rank_accuracy": True,
        "rank_max_mean_abs_error": False,
        "remote_destination_abs_error": False,
        "expert_histogram_jsd": False,
        "expert_weight_histogram_jsd": False,
        "rank_distribution_jsd": False,
    }
    for window in WINDOWS:
        output[window] = {}
        candidate = _rows(metrics, ratio, mode, "P_separated", window)
        for control in ("P_full", "P_text", "P_domain"):
            base = _rows(metrics, ratio, mode, control, window)
            output[window][f"P_separated_minus_{control}"] = {
                metric: paired_bootstrap(candidate, base, metric, higher)
                for metric, higher in directions.items()
            }
    return output


def _predictions(samples: dict[str, dict[str, Any]], ids: list[str], memory_ids: list[str]) -> dict[str, dict[str, Any]]:
    memory = [samples[item] for item in memory_ids if item in samples]
    result: dict[str, dict[str, Any]] = {"P_full": {}, "P_separated": {}}
    for request_id in ids:
        for predictor in result:
            value, _ = predict(samples[request_id], memory, predictor, "1-128")
            result[predictor][request_id] = value
    return result


def _ep8_summary(samples: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    per_domain: dict[str, np.ndarray] = {domain: np.zeros(8) for domain in DOMAINS}
    assignments = np.zeros(8)
    active = np.zeros(8)
    hottest: Counter[int] = Counter()
    maxmean: list[float] = []
    remote = 0.0
    remote_total = 0.0
    for request_id in ids:
        sample = samples[request_id]
        count = sample["decode"]["1-128"]["count"]
        rank = count.reshape(48, 8, 16).sum(axis=-1)
        aggregate = rank.sum(axis=0)
        assignments += aggregate
        per_domain[sample["domain"]] += aggregate
        active += np.count_nonzero(count.reshape(48, 8, 16), axis=-1).mean(axis=0)
        hottest[int(np.argmax(aggregate))] += 1
        maxmean.extend((rank.max(axis=1) / np.maximum(rank.mean(axis=1), 1e-12)).tolist())
        source = sample["decode"]["1-128"]["source"]
        for source_rank in range(source.shape[1]):
            for expert in range(NUM_EXPERTS):
                value = float(source[:, source_rank, expert].sum())
                remote_total += value
                if expert // 16 != source_rank % 8:
                    remote += value
    total = max(float(assignments.sum()), 1.0)
    return {
        "evidence_label": "STRUCTURAL_EP8_PROJECTION_ONLY",
        "latency_tpot_e2e_speedup": "NOT_COMPUTED_PROHIBITED",
        "experts_per_rank": 16,
        "request_count": len(ids),
        "expert_owner_map": {str(expert): expert // 16 for expert in range(NUM_EXPERTS)},
        "per_rank_decode_assignments": assignments.tolist(),
        "rank_distribution": (assignments / total).tolist(),
        "rank_max_mean": float(np.mean(maxmean)),
        "hottest_rank_request_counts": {str(key): value for key, value in sorted(hottest.items())},
        "mean_active_experts_per_rank_per_layer": (active / max(len(ids), 1)).tolist(),
        "remote_destination_fraction": remote / max(remote_total, 1.0),
        "logical_remote_assignments": remote,
        "logical_remote_dispatch_combine_bytes": 2.0 * remote * 2048 * 2,
        "domain_rank_prototypes": {
            domain: (value / max(float(value.sum()), 1.0)).tolist() for domain, value in per_domain.items()
        },
    }


def _rank_signature(sample: dict[str, Any], predictor: str, ep_size: int) -> np.ndarray:
    width = NUM_EXPERTS // ep_size
    vision_raw = sample["vision_mass"].reshape(48, ep_size, width).sum(axis=-1)
    text_raw = sample["text_mass"].reshape(48, ep_size, width).sum(axis=-1)
    vision = vision_raw / np.maximum(vision_raw.sum(axis=-1, keepdims=True), 1e-12)
    text = text_raw / np.maximum(text_raw.sum(axis=-1, keepdims=True), 1e-12)
    if predictor == "P_vision": value = vision.reshape(-1)
    elif predictor == "P_text": value = text.reshape(-1)
    elif predictor == "P_full":
        value = ((vision_raw + text_raw) / np.maximum((vision_raw + text_raw).sum(axis=-1, keepdims=True), 1e-12)).reshape(-1)
    else: value = np.concatenate((vision.reshape(-1), text.reshape(-1)))
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _rank_separation(samples: list[dict[str, Any]], predictor: str, ep_size: int) -> dict[str, float]:
    vectors = np.stack([_rank_signature(sample, predictor, ep_size) for sample in samples])
    distance = 1.0 - vectors @ vectors.T
    within, between, silhouettes = [], [], []
    correct = 0
    for index, sample in enumerate(samples):
        same = [j for j, other in enumerate(samples) if other["domain"] == sample["domain"] and j != index]
        other_groups = [
            [j for j, other in enumerate(samples) if other["domain"] == domain]
            for domain in DOMAINS if domain != sample["domain"]
        ]
        within.extend(distance[index, same].tolist())
        for group in other_groups: between.extend(distance[index, group].tolist())
        a = float(np.mean(distance[index, same]))
        b = min(float(np.mean(distance[index, group])) for group in other_groups)
        silhouettes.append((b - a) / max(a, b, 1e-12))
        candidates = [j for j in range(len(samples)) if j != index]
        nearest = min(candidates, key=lambda j: distance[index, j])
        correct += int(samples[nearest]["domain"] == sample["domain"])
    return {
        "within_domain_cosine_distance": float(np.mean(within)),
        "between_domain_cosine_distance": float(np.mean(between)),
        "between_minus_within": float(np.mean(between) - np.mean(within)),
        "nearest_neighbor_same_domain_rate": correct / len(samples),
        "silhouette": float(np.mean(silhouettes)),
    }


def _physical_comparison(ep4: dict[str, dict[str, Any]], ep2: dict[str, dict[str, Any]]) -> dict[str, Any]:
    common = sorted(ep4.keys() & ep2.keys())
    expert_cos, changed_hot = [], []
    for request_id in common:
        left = ep2[request_id]["decode"]["1-128"]["count"].sum(axis=0)
        right = ep4[request_id]["decode"]["1-128"]["count"].sum(axis=0)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        expert_cos.append(float(np.dot(left, right) / denominator) if denominator else 0.0)
        changed_hot.append(int(np.argmax(left.reshape(2, 64).sum(axis=1))) != int(np.argmax(right.reshape(4, 32).sum(axis=1))))
    return {
        "common_request_count": len(common),
        "expert_histogram_cosine_ep2_vs_ep4": float(np.mean(expert_cos)) if expert_cos else None,
        "hottest_rank_identity_not_comparable_topology_fraction_different": float(np.mean(changed_hot)) if changed_hot else None,
        "interpretation": "Expert IDs are comparable; physical rank IDs are topology-specific and are not assumed identical.",
    }


def _semantic_parity() -> dict[str, Any]:
    ep2_path = ARTIFACTS / "runs" / "ep2_mixed48" / "results.jsonl"
    ep4_path = ARTIFACTS / "runs" / "ep4_mixed160" / "results.jsonl"
    if not ep2_path.exists() or not ep4_path.exists():
        return {"status": "UNAVAILABLE"}
    ep2 = {row["request_id"]: row for row in read_jsonl(ep2_path)}
    ep4 = {row["request_id"]: row for row in read_jsonl(ep4_path)}
    ids = sorted(ep2.keys() & ep4.keys())
    token_equal = [ep2[item]["output_token_ids"] == ep4[item]["output_token_ids"] for item in ids]
    text_equal = [ep2[item]["output_text"] == ep4[item]["output_text"] for item in ids]
    return {
        "status": "MEASURED_DETERMINISTIC_TOPOLOGY_COMPARISON",
        "request_count": len(ids),
        "exact_output_token_sequence_rate": float(np.mean(token_equal)) if ids else None,
        "exact_output_text_rate": float(np.mean(text_equal)) if ids else None,
        "no_method_semantics_changed": True,
        "interpretation": "Topology-dependent BF16 reduction order can alter generation; this PoC does not change model computation.",
    }


def _image_swap_scaling() -> dict[str, Any]:
    path = ROOT.parent / "poc_mllm_long_reasoning_decode_vision_signal" / "artifacts" / "compact_routing_summary.jsonl"
    if not path.exists():
        return {"status": "UNAVAILABLE"}
    rows = [row for row in read_jsonl(path) if row.get("record_type") == "decode" and row.get("repeat_id") is None]
    grouped: dict[tuple[str, int, int], dict[str, dict[int, float]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["pair_id"], int(row["step"]), int(row["layer"]))][row["condition"]] = {
            int(key): float(value) for key, value in row["expert_count"].items()
        }
    result: dict[str, Any] = {}
    for ep_size in (2, 4, 8):
        changed, changed_rank, total = 0.0, 0.0, 0.0
        width = NUM_EXPERTS // ep_size
        for pair in grouped.values():
            if not {"A", "B"} <= pair.keys():
                continue
            a, b = pair["A"], pair["B"]
            experts = set(a) | set(b)
            mismatch = 0.5 * sum(abs(a.get(expert, 0.0) - b.get(expert, 0.0)) for expert in experts)
            n = 0.5 * (sum(a.values()) + sum(b.values()))
            rank_a = Counter({rank: 0.0 for rank in range(ep_size)})
            rank_b = Counter({rank: 0.0 for rank in range(ep_size)})
            for expert, value in a.items(): rank_a[expert // width] += value
            for expert, value in b.items(): rank_b[expert // width] += value
            rank_mismatch = 0.5 * sum(abs(rank_a[rank] - rank_b[rank]) for rank in range(ep_size))
            changed += mismatch
            changed_rank += rank_mismatch
            total += n
        result[str(ep_size)] = {
            "changed_expert_fraction": changed / max(total, 1.0),
            "changed_rank_fraction": changed_rank / max(total, 1.0),
            "event_count": sum({"A", "B"} <= value.keys() for value in grouped.values()),
            "source": "frozen prior causal image-swap trace; rank mapping replayed structurally",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep4", type=Path, default=ARTIFACTS / "ep4_compact.jsonl.gz")
    parser.add_argument("--ep2", type=Path, default=ARTIFACTS / "ep2_compact.jsonl.gz")
    args = parser.parse_args()
    ep4_rows = read_jsonl(args.ep4)
    ep2_rows = read_jsonl(args.ep2)
    if len(ep4_rows) < 120:
        raise RuntimeError(f"EP4 compact trace incomplete: {len(ep4_rows)} rows")
    ep4 = {row["request_id"]: materialize(row) for row in ep4_rows}
    ep2 = {row["request_id"]: materialize(row) for row in ep2_rows}
    warmups = read_json(MANIFESTS / "warmup_indices.json")
    iid = read_json(MANIFESTS / "frozen_iid_stream.json")["request_ids"]
    shift_payload = read_json(MANIFESTS / "frozen_domain_shift_stream.json")["events"]
    shift = [row["request_id"] for row in shift_payload]
    phase_by_id = {row["request_id"]: row["phase"] for row in shift_payload}

    discovery = [row for row in ep4.values() if row["analysis_split"] == "discovery"]
    heldout = [row for row in ep4.values() if row["analysis_split"] == "heldout"]
    separation_result = {
        "discovery": {predictor: separation(discovery, predictor) for predictor in ("P_text", "P_full", "P_separated", "P_vision")},
        "heldout": {predictor: separation(heldout, predictor) for predictor in ("P_text", "P_full", "P_separated", "P_vision")},
        "layerwise_discovery": {predictor: layer_separation(discovery, predictor) for predictor in ("P_text", "P_full", "P_separated", "P_vision")},
    }
    atomic_json(ARTIFACTS / "domain_signature_separation.json", _plain(separation_result))

    discovery_curve, discovery_metrics, discovery_events = _eval_grid(ep4, iid, warmups, "discovery", 4)
    knee = _choose_knee(discovery_curve)
    heldout_curve, heldout_metrics, heldout_events = _eval_grid(ep4, iid, warmups, "heldout", 4)
    ratio = float(knee["ratio"])
    warmup_curve = {"discovery": discovery_curve, "heldout": heldout_curve, "selected_knee": knee}
    atomic_json(ARTIFACTS / "warmup_learning_curve.json", _plain(warmup_curve))

    shift_initial = [item for item in _warmup_ids(warmups, ratio) if item in ep4]
    shift_eval = _ordered(shift, ep4, "heldout", set(shift_initial))
    shift_result: dict[str, Any] = {}
    shift_by_phase: dict[str, Any] = {}
    shift_metrics: list[dict[str, Any]] = []
    shift_events: list[dict[str, Any]] = []
    for mode, continual in (("M1_frozen", False), ("M2_continual", True)):
        shift_result[mode] = {}
        shift_by_phase[mode] = {}
        for predictor in PREDICTORS:
            rows, events = evaluate_stream(ep4, shift_eval, shift_initial, predictor, "1-128", 4, continual)
            for row in rows: row.update({"ratio": ratio, "split": "domain_shift", "mode": mode, "ep_size": 4, "phase": phase_by_id[row["request_id"]]})
            for event in events: event.update({"ratio": ratio, "split": "domain_shift", "mode": mode, "ep_size": 4, "phase": phase_by_id[event["request_id"]]})
            shift_metrics.extend(rows); shift_events.extend(events)
            shift_result[mode][predictor] = aggregate_metrics(rows)
            shift_by_phase[mode][predictor] = {
                phase: aggregate_metrics([row for row in rows if row["phase"] == phase])
                for phase in dict.fromkeys(phase_by_id[item] for item in shift_eval)
            }
    online = {
        "selected_ratio": ratio,
        "iid_heldout": {mode: heldout_curve[_ratio_key(ratio)][mode] for mode in ("M1_frozen", "M2_continual")},
        "domain_shift": shift_result,
        "domain_shift_by_phase": shift_by_phase,
    }
    online["continual_minus_frozen_paired_ci"] = {}
    for label, source in (("iid_heldout", heldout_metrics), ("domain_shift", shift_metrics)):
        continual_rows = _rows(source, ratio, "M2_continual", "P_separated", "1-128")
        frozen_rows = _rows(source, ratio, "M1_frozen", "P_separated", "1-128")
        online["continual_minus_frozen_paired_ci"][label] = {
            "expert_histogram_cosine": paired_bootstrap(continual_rows, frozen_rows, "expert_histogram_cosine", True),
            "rank_distribution_cosine": paired_bootstrap(continual_rows, frozen_rows, "rank_distribution_cosine", True),
            "per_rank_load_mae": paired_bootstrap(continual_rows, frozen_rows, "per_rank_load_mae", False),
        }
    atomic_json(ARTIFACTS / "online_vs_frozen_calibration.json", _plain(online))

    comparisons = {
        "selected_ratio": ratio,
        "iid_heldout_frozen": _comparisons(heldout_metrics, ratio, "M1_frozen"),
        "iid_heldout_continual": _comparisons(heldout_metrics, ratio, "M2_continual"),
        "metric_rows_file": "predictor_metric_rows.jsonl.gz",
        "checkpoint_sampling": "decode steps 1,2,4,8,12,16,20,24,28,32,40,...,128",
    }
    atomic_json(ARTIFACTS / "predictor_comparison.json", _plain(comparisons))
    write_jsonl(ARTIFACTS / "predictor_metric_rows.jsonl.gz", _plain(discovery_metrics + heldout_metrics + shift_metrics))
    write_jsonl(ARTIFACTS / "no_future_leakage_events.jsonl.gz", _plain(discovery_events + heldout_events + shift_events))

    heldout_ids = _ordered(iid, ep4, "heldout", set())
    memory_ids = [item for item in _warmup_ids(warmups, ratio) if item in ep4]
    selected_outputs = []
    frozen_memory = [ep4[item] for item in memory_ids]
    for request_id in heldout_ids:
        for predictor in PREDICTORS:
            value, neighbors = predict(ep4[request_id], frozen_memory, predictor, "1-128")
            expert = value["count"].sum(axis=0)
            selected_outputs.append({
                "request_id": request_id,
                "domain": ep4[request_id]["domain"],
                "predictor": predictor,
                "mode": "M1_frozen",
                "warmup_ratio": ratio,
                "neighbor_request_ids": neighbors,
                "predicted_expert_histogram": {str(index): float(item) for index, item in enumerate(expert) if item},
                "predicted_active_expert_union": np.flatnonzero(expert).tolist(),
                "predicted_rank_histogram_ep2": expert.reshape(2, 64).sum(axis=1).tolist(),
                "predicted_rank_histogram_ep4": expert.reshape(4, 32).sum(axis=1).tolist(),
                "predicted_rank_histogram_ep8_structural": expert.reshape(8, 16).sum(axis=1).tolist(),
            })
    write_jsonl(ARTIFACTS / "selected_predictor_outputs.jsonl.gz", _plain(selected_outputs))
    prediction = _predictions(ep4, heldout_ids, memory_ids)
    locality_result = locality(ep4, heldout_ids, prediction)
    atomic_json(ARTIFACTS / "mixed_domain_locality.json", _plain(locality_result))

    ep2_curve = None
    ep2_metrics: list[dict[str, Any]] = []
    if ep2:
        ep2_curve, ep2_metrics, _ = _eval_grid(ep2, iid, warmups, "discovery", 2)
    scaling = {
        "ep2_physical": {
            "status": "TRUE_DEEPEP_EP2_MEASURED" if ep2 else "NOT_RUN",
            "request_count": len(ep2), "experts_per_rank": 64,
            "selected_ratio_curve": ep2_curve.get(_ratio_key(ratio), {}) if ep2_curve else {},
        },
        "ep4_physical": {
            "status": "TRUE_DEEPEP_EP4_MEASURED", "request_count": len(ep4), "experts_per_rank": 32,
            "selected_ratio_curve": heldout_curve[_ratio_key(ratio)],
        },
        "physical_route_comparison": _physical_comparison(ep4, ep2),
        "semantic_parity": _semantic_parity(),
        "image_swap_rank_scaling": _image_swap_scaling(),
        "rank_signature_domain_separation": {
            "ep2_physical_subset": {
                predictor: _rank_separation(list(ep2.values()), predictor, 2)
                for predictor in ("P_text", "P_full", "P_separated", "P_vision")
            } if len(ep2) >= 8 else {},
            "ep4_physical_heldout": {
                predictor: _rank_separation(heldout, predictor, 4)
                for predictor in ("P_text", "P_full", "P_separated", "P_vision")
            },
            "ep8_structural_heldout": {
                predictor: _rank_separation(heldout, predictor, 8)
                for predictor in ("P_text", "P_full", "P_separated", "P_vision")
            },
        },
    }
    atomic_json(ARTIFACTS / "ep2_ep4_scaling.json", _plain(scaling))

    physical_manifest = {
        "ep2": {
            "invocation": read_json(ARTIFACTS / "runs" / "ep2_mixed48" / "invocation.json"),
            "run": read_json(ARTIFACTS / "runs" / "ep2_mixed48" / "run.json"),
            "trace_audit": read_json(ARTIFACTS / "ep2_trace_audit.json"),
            "experts_per_rank": 64,
            "gpu_uuids": ["GPU-6076e2f2-5b63-3761-5586-56ceb7df8139", "GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730"],
        },
        "ep4": {
            "invocation": read_json(ARTIFACTS / "runs" / "ep4_mixed160" / "invocation.json"),
            "run": read_json(ARTIFACTS / "runs" / "ep4_mixed160" / "run.json"),
            "trace_audit": read_json(ARTIFACTS / "ep4_trace_audit.json"),
            "experts_per_rank": 32,
            "gpu_uuids": [
                "GPU-6076e2f2-5b63-3761-5586-56ceb7df8139", "GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730",
                "GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28", "GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b",
            ],
        },
        "backend_contract": "DeepEP high-throughput sparse dispatch/combine; linear expert placement; no silent fallback",
        "instrumentation": "routing evidence pass; not clean latency timing",
    }
    atomic_json(ARTIFACTS / "physical_anchor_manifest.json", physical_manifest)

    ep8 = _ep8_summary(ep4, heldout_ids)
    ep8["locality"] = {size: value["ep"]["8"] for size, value in locality_result.items() if size != "_meta"}
    # Re-score heldout predictions under the finer EP8 owner mapping.
    ep8_metrics: dict[str, Any] = {}
    for predictor in PREDICTORS:
        rows, _ = evaluate_stream(ep4, heldout_ids, memory_ids, predictor, "1-128", 8, False)
        ep8_metrics[predictor] = aggregate_metrics(rows)
    ep8["predictor_metrics"] = ep8_metrics
    ep8["image_swap_changed_rank_fraction"] = scaling["image_swap_rank_scaling"].get("8", {})
    atomic_json(ARTIFACTS / "ep8_structural_projection.json", _plain(ep8))

    overhead_result = overhead(discovery[: min(16, len(discovery))])
    atomic_json(ARTIFACTS / "predictor_overhead.json", _plain(overhead_result))

    selected = comparisons["iid_heldout_frozen"]["1-128"]
    sep = heldout_curve[_ratio_key(ratio)]["M1_frozen"]["P_separated"]["1-128"]
    full_delta = selected["P_separated_minus_P_full"]["expert_histogram_cosine"]
    text_delta = selected["P_separated_minus_P_text"]["expert_histogram_cosine"]
    rank_full_delta = selected["P_separated_minus_P_full"]["rank_distribution_cosine"]
    rank_text_delta = selected["P_separated_minus_P_text"]["rank_distribution_cosine"]
    domain_sep = separation_result["heldout"]["P_separated"]["between_minus_within"]
    domain_sep_clear = domain_sep > 0 and separation_result["heldout"]["P_separated"]["nearest_neighbor_same_domain_rate"] > 0.25
    uncalibrated = heldout_curve["0.00"]["M1_frozen"]["P_separated"]["1-128"].get("expert_histogram_cosine", 0.0)
    warmup_gain = sep.get("expert_histogram_cosine", 0.0) - uncalibrated
    gate_b_warmup_gain = max(
        heldout_curve[key]["M1_frozen"]["P_separated"]["1-128"].get("expert_histogram_cosine", 0.0) - uncalibrated
        for key in ("0.05", "0.10")
    )
    rank_scaling = scaling["image_swap_rank_scaling"]
    rank_effect_strengthens = all(str(size) in rank_scaling for size in (2, 4, 8)) and all(
        rank_scaling[str(left)]["changed_rank_fraction"] < rank_scaling[str(right)]["changed_rank_fraction"]
        for left, right in ((2, 4), (4, 8))
    )
    iid_frozen_cosine = online["iid_heldout"]["M1_frozen"]["P_separated"]["1-128"].get("expert_histogram_cosine", 0.0)
    iid_continual_cosine = online["iid_heldout"]["M2_continual"]["P_separated"]["1-128"].get("expert_histogram_cosine", 0.0)
    calibration_stable_or_improving = iid_continual_cosine >= iid_frozen_cosine - 0.005
    if domain_sep <= 0:
        verdict = "NO_DOMAIN_SEPARATION"
    elif gate_b_warmup_gain <= 0:
        verdict = "NO_ONLINE_CALIBRATION_GAIN"
    elif (
        full_delta["ci95_low"] > 0 and text_delta["ci95_low"] > 0
        and domain_sep_clear
        and gate_b_warmup_gain > 0
        and (rank_full_delta["ci95_low"] > 0 or rank_text_delta["ci95_low"] > 0)
        and rank_effect_strengthens
        and calibration_stable_or_improving
    ):
        verdict = "MIXED_DOMAIN_ONLINE_VISION_SIGNAL_SUPPORTED_STRONG"
    elif full_delta["ci95_low"] > 0:
        verdict = "MIXED_DOMAIN_ONLINE_VISION_SIGNAL_SUPPORTED_PARTIAL"
    elif selected["P_separated_minus_P_domain"]["expert_histogram_cosine"]["ci95_high"] <= 0:
        verdict = "DOMAIN_LABEL_SUFFICIENT"
    else:
        verdict = "GENERIC_FULL_PREFILL_SUFFICIENT"
    final = {
        "verdict": verdict,
        "selected_warmup_ratio": ratio,
        "heldout_requests": len(heldout_ids),
        "ep2_requests": len(ep2),
        "ep4_requests": len(ep4),
        "p_separated_aggregate": sep,
        "vision_increment_vs_full": selected["P_separated_minus_P_full"],
        "vision_increment_vs_text": selected["P_separated_minus_P_text"],
        "domain_label_control": selected["P_separated_minus_P_domain"],
        "domain_separation": separation_result["heldout"],
        "online_vs_frozen": online,
        "locality": locality_result,
        "ep8": ep8,
        "predictor_overhead": overhead_result,
        "execution_evidence": {
            "ep4_trace_requests": len(ep4), "ep2_trace_requests": len(ep2),
            "full_model_execution_count": sum(
                int(row.get("request_executions", 0))
                for row in read_jsonl(ARTIFACTS / "execution_ledger.jsonl")
                if row.get("status") == "COMPLETE"
            ),
            "budget": 400,
        },
        "verdict_gate_evidence": {
            "warmup_expert_cosine_gain_vs_uncalibrated": warmup_gain,
            "best_5_or_10_percent_warmup_gain_vs_uncalibrated": gate_b_warmup_gain,
            "p_separated_vs_full_expert_ci_low": full_delta["ci95_low"],
            "p_separated_vs_text_expert_ci_low": text_delta["ci95_low"],
            "p_separated_vs_full_rank_ci_low": rank_full_delta["ci95_low"],
            "p_separated_vs_text_rank_ci_low": rank_text_delta["ci95_low"],
            "image_swap_rank_effect_strictly_increases_ep2_ep4_ep8": rank_effect_strengthens,
            "continual_calibration_stable_or_improving": calibration_stable_or_improving,
            "domain_separation_above_chance": domain_sep_clear,
        },
    }
    atomic_json(ARTIFACTS / "final_verdict.json", _plain(final))


if __name__ == "__main__":
    main()
