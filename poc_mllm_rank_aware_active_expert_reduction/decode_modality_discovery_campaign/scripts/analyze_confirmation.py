from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import atomic_json, read_json, read_jsonl_gz
from .analyze_discovery import add_persistence, bootstrap, circuit_rows, paired_effect, wave_metrics


def structural_effect(rows: list[dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    hypothesis = frozen["primary_hypothesis"]
    if hypothesis == "H1_REDUNDANCY":
        return paired_effect(rows, "output_effective_rank", -1.0)
    if hypothesis == "H2_PERSISTENCE":
        return paired_effect(add_persistence(rows), "step_topk_jaccard", 1.0)
    if hypothesis == "H3_CIRCUIT":
        return paired_effect(circuit_rows(rows), "motif_persistence_jaccard", 1.0)
    if hypothesis == "H4_RANK_OCCUPANCY":
        return paired_effect(wave_metrics(rows), "singleton_experts", 1.0)
    raise ValueError(hypothesis)


def domain_stability(effect: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    domain = {case["request_id"]: case["domain"] for case in manifest["cases"] if case["split"] == "confirmation"}
    grouped: dict[str, list[float]] = defaultdict(list)
    for request, value in effect["per_request_delta"].items():
        grouped[domain[request]].append(float(value))
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    return {"domain_means": means, "positive_domains": sum(value > 0 for value in means.values()),
            "domains": len(means), "stable_at_least_four_of_five": sum(value > 0 for value in means.values()) >= 4}


def functional_comparison(labels: list[dict[str, Any]], candidate: str, keep: int) -> dict[str, Any]:
    by = {(row["request_id"], row["policy"], row["keep_k"]): row for row in labels}
    requests = sorted({row["request_id"] for row in labels if row["policy"] == candidate and row["keep_k"] == keep})
    vs_tail, vs_random = {}, {}
    for request in requests:
        value = by[(request, candidate, keep)]["horizon_mean_damage"]
        vs_tail[request] = by[(request, "router_tail", keep)]["horizon_mean_damage"] - value
        vs_random[request] = by[(request, "random", keep)]["horizon_mean_damage"] - value
    return {"positive_means_candidate_lower_damage": True, "vs_router_tail": bootstrap(vs_tail),
            "vs_random": bootstrap(vs_random)}


def closure(labels: list[dict[str, Any]], policy: str, keep: int) -> float:
    values = [float(row["predicted_wave_closure_events"]) for row in labels if row["policy"] == policy and row["keep_k"] == keep]
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--probe-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, frozen = read_json(args.manifest), read_json(args.frozen)
    rows, labels = list(read_jsonl_gz(args.sites)), list(read_jsonl_gz(args.probe_labels))
    effect = structural_effect(rows, frozen)
    candidate, rank_policy = frozen["policy"], frozen["rank_policy"]
    functional = {str(keep): functional_comparison(labels, candidate, keep) for keep in (6, 4)}
    rank_functional = None if rank_policy == candidate else {
        str(keep): functional_comparison(labels, rank_policy, keep) for keep in (6, 4)
    }
    stability = domain_stability(effect, manifest)
    primary = functional["6"]
    g1 = effect["ci95"][0] > 0
    g2 = primary["vs_router_tail"]["ci95"][0] > 0 and primary["vs_random"]["ci95"][0] > 0
    g3 = stability["stable_at_least_four_of_five"]
    if g1 and g2 and g3:
        verdict = "MLLM_DECODE_SKIP_SIGNAL_CONFIRMED"
    elif g1 and not g2:
        verdict = "MODALITY_EFFECT_NOT_USEFUL_FOR_SKIPPING"
    elif not g1 and g2:
        verdict = "GENERIC_SKIP_STRUCTURE_ONLY"
    else:
        verdict = "NO_REPRODUCIBLE_DECODE_MODALITY_SIGNAL"
    rank_closure = {
        "candidate_k6": closure(labels, candidate, 6),
        "rank_candidate_k6": closure(labels, rank_policy, 6),
        "router_tail_k6": closure(labels, "router_tail", 6),
        "random_k6": closure(labels, "random", 6),
    }
    payload = {
        "schema_version": 1, "primary_hypothesis": frozen["primary_hypothesis"],
        "primary_policy": candidate, "rank_policy": rank_policy,
        "structural_effect": effect, "functional": functional,
        "rank_policy_functional": rank_functional, "rank_closure": rank_closure,
        "domain_stability": stability, "gates": {"G1_modality": g1, "G2_skip_utility": g2, "G3_robustness": g3},
        "true_ep4_runtime_promoted": bool(g1 and g2 and g3), "verdict": verdict,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
