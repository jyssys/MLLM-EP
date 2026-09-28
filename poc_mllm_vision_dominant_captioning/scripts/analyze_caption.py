from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from poc_mllm_vision_dominant_captioning.analysis import (
    PREDICTORS,
    aggregate_metrics,
    cosine,
    jsd,
    layer_separation,
    locality,
    materialize,
    paired_bootstrap,
    predict,
    prediction_metrics,
    rank_histogram,
    separation,
)
from poc_mllm_vision_dominant_captioning.contract import (
    DOMAINS,
    NUM_EXPERTS,
    WINDOWS,
    atomic_json,
    read_json,
    read_jsonl,
    owner_rank,
    UNIVERSAL_PROMPT,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
METRIC_DIRECTIONS = {
    "expert_histogram_cosine": True,
    "expert_histogram_jsd": False,
    "decode_expert_top8_recall": True,
    "active_expert_set_recall": True,
    "rank_distribution_cosine": True,
    "rank_distribution_jsd": False,
    "per_rank_load_mae": False,
    "hottest_rank_accuracy": True,
}


def _decode_separation(samples: list[dict[str, Any]], ep_size: int) -> dict[str, float]:
    vectors, ranks, labels = [], [], []
    for sample in samples:
        count = sample["decode"]["1-256"]["count"].reshape(-1)
        rank = rank_histogram(sample["decode"]["1-256"]["count"], ep_size).reshape(-1)
        vectors.append(count / max(float(np.linalg.norm(count)), 1e-12))
        ranks.append(rank / max(float(np.linalg.norm(rank)), 1e-12))
        labels.append(sample["domain"])
    def score(array: np.ndarray) -> dict[str, float]:
        matrix = array @ array.T
        within, between, correct = [], [], 0
        for i, domain in enumerate(labels):
            candidates = [j for j in range(len(labels)) if j != i]
            nearest = max(candidates, key=lambda j: matrix[i, j])
            correct += int(labels[nearest] == domain)
            for j in range(i + 1, len(labels)):
                (within if labels[j] == domain else between).append(1.0 - matrix[i, j])
        return {
            "within_domain_cosine_distance": float(np.mean(within)),
            "between_domain_cosine_distance": float(np.mean(between)),
            "between_minus_within": float(np.mean(between) - np.mean(within)),
            "nearest_neighbor_same_domain_rate": correct / len(labels),
        }
    return {"expert": score(np.stack(vectors)), "rank": score(np.stack(ranks))}


def _domain_rank_profiles(samples: list[dict[str, Any]], ep_size: int) -> dict[str, Any]:
    profiles = {}
    for domain in DOMAINS:
        vectors = []
        for sample in samples:
            if sample["domain"] != domain:
                continue
            value = rank_histogram(sample["decode"]["1-256"]["count"], ep_size)
            value = value / max(float(value.sum()), 1e-12)
            vectors.append(value)
        profiles[domain] = np.mean(vectors, axis=0).tolist()
    centroids = {
        domain: np.asarray(value).reshape(-1) / max(float(np.linalg.norm(value)), 1e-12)
        for domain, value in profiles.items()
    }
    matrix = {
        left: {
            right: float(1.0 - np.dot(centroids[left], centroids[right]))
            for right in DOMAINS
        }
        for left in DOMAINS
    }
    return {
        "ep_size": ep_size,
        "experts_per_rank": NUM_EXPERTS // ep_size,
        "domain_layer_rank_profiles": profiles,
        "centroid_cosine_distance": matrix,
    }


def _predictors(samples: list[dict[str, Any]], ep_size: int) -> dict[str, Any]:
    discovery = [row for row in samples if row["analysis_split"] == "discovery"]
    heldout = [row for row in samples if row["analysis_split"] == "heldout"]
    metric_rows = []
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {
        key: {} for key in ("P_text", "P_vision", "P_full", "P_separated")
    }
    for query in heldout:
        for predictor in ("P_text", "P_vision", "P_full", "P_separated"):
            for window in WINDOWS:
                prediction, neighbors = predict(query, discovery, predictor, window)
                metrics = prediction_metrics(query["decode"][window], prediction, ep_size)
                if metrics:
                    metric_rows.append({
                        "request_id": query["request_id"],
                        "domain": query["domain"],
                        "predictor": predictor,
                        "window": window,
                        "neighbor_request_ids": neighbors,
                        **metrics,
                    })
                if window == "1-256":
                    predictions[predictor][query["request_id"]] = prediction
    aggregate = {}
    for predictor in predictions:
        aggregate[predictor] = {}
        for window in WINDOWS:
            rows = [row for row in metric_rows if row["predictor"] == predictor and row["window"] == window]
            aggregate[predictor][window] = aggregate_metrics(rows)
    ci = {}
    for candidate, control, label in (
        ("P_vision", "P_text", "vision_vs_text"),
        ("P_separated", "P_text", "separated_vs_text"),
        ("P_separated", "P_full", "separated_vs_full"),
    ):
        ci[label] = {}
        for window in WINDOWS:
            ci[label][window] = {}
            left = [row for row in metric_rows if row["predictor"] == candidate and row["window"] == window]
            right = [row for row in metric_rows if row["predictor"] == control and row["window"] == window]
            for metric, higher in METRIC_DIRECTIONS.items():
                ci[label][window][metric] = paired_bootstrap(left, right, metric, higher)
    return {
        "discovery_requests": len(discovery),
        "heldout_requests": len(heldout),
        "aggregate": aggregate,
        "paired_bootstrap": ci,
        "metric_rows": metric_rows,
        "predictions": predictions,
    }


def _multiset_change(left: np.ndarray, right: np.ndarray) -> float:
    total = max(float(left.sum()), float(right.sum()), 1.0)
    return float(np.abs(left - right).sum()) / (2.0 * total)


def _causal(compact: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {row["request_id"]: materialize(row) for row in compact}
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for request in rows:
        if "::" not in request:
            continue
        if "::A_repeat" in request:
            pair, suffix = request.split("::A_repeat", 1)
            condition = f"repeat{suffix.split('::', 1)[0]}"
        else:
            pair, suffix = request.rsplit("::", 2)[0], request.rsplit("::", 2)[1]
            condition = suffix
        groups[pair][condition] = request
    records = []
    for pair, mapping in groups.items():
        comparisons = []
        if all(key in mapping for key in ("A", "B_same")):
            comparisons.append(("same_domain", mapping["A"], mapping["B_same"]))
        if all(key in mapping for key in ("A", "B_cross")):
            comparisons.append(("cross_domain", mapping["A"], mapping["B_cross"]))
        if all(key in mapping for key in ("repeat0", "repeat1")):
            comparisons.append(("repeat_noise", mapping["repeat0"], mapping["repeat1"]))
        for kind, left_id, right_id in comparisons:
            for window in WINDOWS:
                left = rows[left_id]["decode"][window]["count"]
                right = rows[right_id]["decode"][window]["count"]
                if not left.sum() or not right.sum():
                    continue
                left_routes = {
                    (int(row["step"]), int(row["layer"])): row
                    for row in rows[left_id].get("sampled_routes", [])
                    if (window == "1-256" or (
                        window == "1-32" and 1 <= int(row["step"]) <= 32
                    ) or (
                        window == "33-64" and 33 <= int(row["step"]) <= 64
                    ) or (
                        window == "65-96" and 65 <= int(row["step"]) <= 96
                    ) or (
                        window == "97-128" and 97 <= int(row["step"]) <= 128
                    ) or (
                        window == "129-192" and 129 <= int(row["step"]) <= 192
                    ) or (
                        window == "193-256" and 193 <= int(row["step"]) <= 256
                    ))
                }
                right_routes = {
                    (int(row["step"]), int(row["layer"])): row
                    for row in rows[right_id].get("sampled_routes", [])
                    if (int(row["step"]), int(row["layer"])) in left_routes
                }
                common = sorted(set(left_routes) & set(right_routes))
                for ep_size in (2, 4, 8):
                    left_rank, right_rank = rank_histogram(left, ep_size), rank_histogram(right, ep_size)
                    expert_changes, rank_changes, jaccards = [], [], []
                    for key in common:
                        a = [int(value) for value in left_routes[key]["route_ids"]]
                        b = [int(value) for value in right_routes[key]["route_ids"]]
                        width = min(len(a), len(b))
                        if not width:
                            continue
                        expert_changes.append(sum(a[index] != b[index] for index in range(width)) / width)
                        rank_changes.append(sum(owner_rank(a[index], ep_size) != owner_rank(b[index], ep_size) for index in range(width)) / width)
                        union = set(a) | set(b)
                        jaccards.append(len(set(a) & set(b)) / len(union) if union else 1.0)
                    records.append({
                        "pair_id": pair,
                        "comparison": kind,
                        "window": window,
                        "ep_size": ep_size,
                        "sampled_route_events": len(expert_changes),
                        # The prior long-reasoning PoC defined "changed fraction"
                        # as histogram/multiset displacement, not ordered top-k slot
                        # mismatch.  Preserve that definition for the cross-regime
                        # comparison and expose ordered slot churn separately.
                        "ordered_expert_slot_change_fraction": float(np.mean(expert_changes)) if expert_changes else None,
                        "expert_changed_fraction": _multiset_change(left, right),
                        "topk_jaccard": float(np.mean(jaccards)) if jaccards else None,
                        "histogram_multiset_change": _multiset_change(left, right),
                        "expert_jsd": jsd(left, right),
                        "expert_cosine": cosine(left, right),
                        "ordered_rank_slot_change_fraction": float(np.mean(rank_changes)) if rank_changes else None,
                        "changed_rank_fraction": _multiset_change(left_rank, right_rank),
                        "rank_jsd": jsd(left_rank, right_rank),
                    })
    aggregate = {}
    for kind in ("same_domain", "cross_domain", "repeat_noise"):
        aggregate[kind] = {}
        for window in WINDOWS:
            aggregate[kind][window] = {}
            for ep_size in (2, 4, 8):
                subset = [row for row in records if row["comparison"] == kind and row["window"] == window and row["ep_size"] == ep_size]
                aggregate[kind][window][str(ep_size)] = {
                    "pairs": len(subset),
                    **({
                        key: float(np.mean([row[key] for row in subset]))
                        for key in ("ordered_expert_slot_change_fraction", "expert_changed_fraction", "topk_jaccard", "histogram_multiset_change", "expert_jsd", "expert_cosine", "ordered_rank_slot_change_fraction", "changed_rank_fraction", "rank_jsd")
                        if all(item.get(key) is not None for item in subset)
                    } if subset else {}),
                }
    rng = np.random.default_rng(3407)
    bootstrap = {}
    for kind in ("same_domain", "cross_domain"):
        bootstrap[kind] = {}
        for window in WINDOWS:
            bootstrap[kind][window] = {}
            for ep_size in (2, 4, 8):
                bootstrap[kind][window][str(ep_size)] = {}
                left = [row for row in records if row["comparison"] == kind and row["window"] == window and row["ep_size"] == ep_size]
                noise = [row for row in records if row["comparison"] == "repeat_noise" and row["window"] == window and row["ep_size"] == ep_size]
                for metric in ("expert_changed_fraction", "changed_rank_fraction", "expert_jsd", "rank_jsd"):
                    if not left or not noise:
                        continue
                    a = np.asarray([row[metric] for row in left])
                    b = np.asarray([row[metric] for row in noise])
                    draws = (
                        a[rng.integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
                        - b[rng.integers(0, len(b), size=(2000, len(b)))].mean(axis=1)
                    )
                    bootstrap[kind][window][str(ep_size)][metric] = {
                        "mean_difference_vs_repeat_noise": float(a.mean() - b.mean()),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "effect_pairs": len(a),
                        "repeat_pairs": len(b),
                    }
    return {"records": records, "aggregate": aggregate, "bootstrap_vs_repeat_noise": bootstrap}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep4-compact", type=Path, required=True)
    parser.add_argument("--natural-results", type=Path, required=True)
    parser.add_argument("--ep2-compact", type=Path)
    parser.add_argument("--causal-compact", type=Path)
    parser.add_argument("--ep2-causal-compact", type=Path)
    parser.add_argument(
        "--iid-stream",
        type=Path,
        default=ROOT / "artifacts/manifests/frozen_iid_stream.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    ep4_rows = read_jsonl(args.ep4_compact)
    samples = [materialize(row) for row in ep4_rows]
    results = read_jsonl(args.natural_results)
    quality_rows = []
    for row in results:
        words = str(row.get("output_text", "")).split()
        ngrams = [tuple(words[index:index + 4]) for index in range(max(0, len(words) - 3))]
        repeated = 0.0 if not ngrams else 1.0 - len(set(ngrams)) / len(ngrams)
        lowered = str(row.get("output_text", "")).lower()
        quality_rows.append({
            "request_id": row["request_id"],
            "domain": row["domain"],
            "word_count": len(words),
            "repeated_4gram_fraction": repeated,
            "malformed": len(words) < 8,
            "refusal": any(text in lowered for text in ("i cannot", "i can't", "unable to describe")),
            "prompt_leakage": UNIVERSAL_PROMPT.lower() in lowered,
            "finish_reason": row["finish_reason"],
        })
    output_lengths = {}
    observed_cohort = {}
    for domain in DOMAINS:
        values = [row["output_token_count"] for row in results if row["domain"] == domain]
        vision_values = [row["vision_token_count"] for row in results if row["domain"] == domain]
        output_lengths[domain] = {
            "n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "ended_before_cap": sum(value < 256 for value in values),
        }
        observed_cohort[domain] = {
            "n": len(vision_values),
            "visual_tokens_min": min(vision_values),
            "visual_tokens_median": float(np.median(vision_values)),
            "visual_tokens_p90": float(np.quantile(vision_values, 0.9)),
            "visual_tokens_max": max(vision_values),
        }
    output_lengths["overall"] = {
        "n": len(results),
        "mean": float(np.mean([row["output_token_count"] for row in results])),
        "median": float(np.median([row["output_token_count"] for row in results])),
        "p90": float(np.quantile([row["output_token_count"] for row in results], 0.9)),
        "normal_eos": sum(row["finish_reason"] == "stop" for row in results),
        "truncated": sum(row["finish_reason"] == "length" for row in results),
        "at_least_64": sum(row["output_token_count"] >= 64 for row in results),
        "at_least_128": sum(row["output_token_count"] >= 128 for row in results),
    }
    sep = {
        predictor: separation(samples, predictor)
        for predictor in ("P_text", "P_vision", "P_full", "P_separated")
    }
    sep["decode_truth"] = _decode_separation(samples, 4)
    sep["layerwise"] = {
        predictor: layer_separation(samples, predictor)
        for predictor in ("P_text", "P_vision", "P_full", "P_separated")
    }
    predictors = _predictors(samples, 4)
    locality_predictions = predictors.pop("predictions")
    # FIFO must follow the pre-routing frozen IID serving order.  Lexical
    # request-id sorting groups domains and is not a valid mixed-domain FIFO.
    frozen_order = read_json(args.iid_stream)["request_ids"]
    heldout_ids = [request for request in frozen_order if request in locality_predictions["P_full"]]
    if set(heldout_ids) != set(locality_predictions["P_full"]):
        raise AssertionError("frozen IID stream does not cover every held-out request exactly once")
    sample_map = {row["request_id"]: row for row in samples}
    loc = locality(sample_map, heldout_ids, locality_predictions, wave_sizes=(1, 2, 4, 8, 16))
    atomic_json(args.output_dir / "caption_output_lengths.json", output_lengths)
    atomic_json(args.output_dir / "cohort_observed_metadata.json", observed_cohort)
    atomic_json(args.output_dir / "caption_quality_sanity.json", {
        "requests": len(quality_rows),
        "malformed": sum(row["malformed"] for row in quality_rows),
        "refusals": sum(row["refusal"] for row in quality_rows),
        "prompt_leakage": sum(row["prompt_leakage"] for row in quality_rows),
        "mean_repeated_4gram_fraction": float(np.mean([row["repeated_4gram_fraction"] for row in quality_rows])),
        "p90_repeated_4gram_fraction": float(np.quantile([row["repeated_4gram_fraction"] for row in quality_rows], 0.9)),
        "rows": quality_rows,
        "CIDEr_BLEU_SPICE": "NOT_RUN_UNIVERSAL_PROMPT_NOT_DATASET_OFFICIAL_CAPTION_PROTOCOL",
    })
    atomic_json(args.output_dir / "domain_separation.json", sep)
    atomic_json(args.output_dir / "ep_rank_profiles.json", {
        str(ep_size): _domain_rank_profiles(samples, ep_size)
        for ep_size in (2, 4, 8)
    })
    write_jsonl(args.output_dir / "predictor_metric_rows.jsonl.gz", predictors.pop("metric_rows"))
    atomic_json(args.output_dir / "predictor_comparison.json", predictors)
    atomic_json(args.output_dir / "batch_locality.json", loc)
    if args.causal_compact and args.causal_compact.exists():
        atomic_json(args.output_dir / "causal_effect.json", _causal(read_jsonl(args.causal_compact)))
    if args.ep2_compact and args.ep2_compact.exists():
        ep2 = [materialize(row) for row in read_jsonl(args.ep2_compact)]
        atomic_json(args.output_dir / "ep2_domain_separation.json", {
            predictor: separation(ep2, predictor)
            for predictor in ("P_text", "P_vision", "P_full", "P_separated")
        })
        ep2_predictors = _predictors(ep2, 2)
        ep2_predictors.pop("metric_rows")
        ep2_predictors.pop("predictions")
        atomic_json(args.output_dir / "ep2_predictor_comparison.json", ep2_predictors)
    if args.ep2_causal_compact and args.ep2_causal_compact.exists():
        atomic_json(args.output_dir / "ep2_causal_effect.json", _causal(read_jsonl(args.ep2_causal_compact)))


if __name__ == "__main__":
    main()
