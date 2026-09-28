from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..contract import SEED, atomic_json, read_jsonl_gz


def cosine(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def jaccard(left: list[int] | set[int], right: list[int] | set[int]) -> float:
    a, b = set(left), set(right)
    return float(len(a & b) / max(len(a | b), 1))


def bootstrap(values: dict[str, float], repeats: int = 10000) -> dict[str, Any]:
    requests = sorted(values)
    observed = float(np.mean([values[key] for key in requests]))
    rng, samples = np.random.default_rng(SEED), []
    for _ in range(repeats):
        chosen = rng.choice(requests, len(requests), replace=True)
        samples.append(float(np.mean([values[str(key)] for key in chosen])))
    return {
        "mean": observed,
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "requests": len(requests),
        "bootstrap_unit": "request",
        "repeats": repeats,
    }


def request_condition_mean(rows: list[dict[str, Any]], key: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["request_id"], row["condition"])].append(float(row[key]))
    return {index: float(np.mean(values)) for index, values in grouped.items()}


def paired_effect(rows: list[dict[str, Any]], key: str, direction: float = 1.0) -> dict[str, Any]:
    means = request_condition_mean(rows, key)
    requests = sorted({request for request, condition in means if condition == "original"})
    neutral = {
        request: direction * (means[(request, "original")] - means[(request, "semantic_neutral")])
        for request in requests
    }
    swap_absolute = {
        request: abs(means[(request, "original")] - means[(request, "same_domain_swap")])
        for request in requests
    }
    result = bootstrap(neutral)
    result["positive_means_more_candidate_structure_in_original"] = True
    result["same_domain_swap_absolute_mean"] = float(np.mean(list(swap_absolute.values())))
    result["per_request_delta"] = neutral
    return result


def policy_gain(rows: list[dict[str, Any]], candidate: str, keep: int) -> dict[str, Any]:
    original = [row for row in rows if row["condition"] == "original"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original:
        grouped[row["request_id"]].append(row)
    tail, random = {}, {}
    for request, values in grouped.items():
        candidate_damage = np.mean([row[f"{candidate}_k{keep}_local_relative_l2"] for row in values])
        tail[request] = float(np.mean([row[f"router_tail_k{keep}_local_relative_l2"] for row in values]) - candidate_damage)
        random[request] = float(np.mean([row[f"random_k{keep}_local_relative_l2"] for row in values]) - candidate_damage)
    return {"vs_router_tail": bootstrap(tail), "vs_random": bootstrap(random)}


def add_persistence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["case_id"], row["condition"], row["layer"], row["decode_step"]): row for row in rows}
    steps = sorted({row["decode_step"] for row in rows})
    result = []
    for row in rows:
        prior = [value for value in steps if value < row["decode_step"]]
        if not prior:
            continue
        previous = lookup[(row["case_id"], row["condition"], row["layer"], prior[-1])]
        result.append({
            **row,
            "step_topk_jaccard": jaccard(row["expert_ids"], previous["expert_ids"]),
            "step_router_cosine": cosine(row["router_probs"], previous["router_probs"]),
            "step_rank_jaccard": jaccard(row["owner_ranks_ep4"], previous["owner_ranks_ep4"]),
        })
    return result


def circuit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["case_id"], row["condition"], row["layer"], row["decode_step"]): row for row in rows}
    layers = sorted({row["layer"] for row in rows})
    steps = sorted({row["decode_step"] for row in rows})
    motifs: dict[tuple[str, str, int], set[tuple[int, int, int]]] = {}
    for row in rows:
        key = (row["case_id"], row["condition"], row["decode_step"])
        if key in motifs:
            continue
        edges: set[tuple[int, int, int]] = set()
        for left, right in zip(layers[:-1], layers[1:]):
            a, b = lookup[(row["case_id"], row["condition"], left, row["decode_step"])], lookup[(row["case_id"], row["condition"], right, row["decode_step"])]
            edges.update((left, int(x), int(y)) for x in a["expert_ids"] for y in b["expert_ids"])
        motifs[key] = edges
    result = []
    cases = sorted({row["case_id"] for row in rows})
    conditions = sorted({row["condition"] for row in rows})
    for case in cases:
        request = next(row["request_id"] for row in rows if row["case_id"] == case)
        for condition in conditions:
            for index in range(1, len(steps)):
                left, right = motifs[(case, condition, steps[index - 1])], motifs[(case, condition, steps[index])]
                result.append({"case_id": case, "request_id": request, "condition": condition,
                               "decode_step": steps[index], "motif_persistence_jaccard": jaccard(left, right)})
    return result


def wave_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["wave_b8"], row["layer"], row["decode_step"])].append(row)
    result = []
    for (condition, wave, layer, step), values in grouped.items():
        counts = Counter(int(expert) for row in values for expert in row["expert_ids"])
        active_per_rank = [sum(expert // 32 == rank for expert in counts) for rank in range(4)]
        total = max(sum(active_per_rank), 1)
        result.append({
            "condition": condition, "wave_b8": wave, "layer": layer, "decode_step": step,
            "request_id": f"wave{wave}",
            "active_expert_union": len(counts),
            "singleton_experts": sum(value == 1 for value in counts.values()),
            "max_rank_active_experts": max(active_per_rank),
            "rank_max_mean": max(active_per_rank) / max(np.mean(active_per_rank), 1e-12),
            "working_set_hhi": float(sum((value / total) ** 2 for value in active_per_rank)),
        })
    return result


def closure_gain(rows: list[dict[str, Any]], policy: str, keep: int) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "original":
            grouped[(row["wave_b8"], row["layer"], row["decode_step"])].append(row)
    per_wave: dict[str, float] = {}
    for key, values in grouped.items():
        before = [Counter() for _ in range(4)]
        after = [Counter() for _ in range(4)]
        for row in values:
            removed = {int(slot) - 1 for slot in row[f"{policy}_k{keep}_removed_slots"]}
            for slot, expert in enumerate(row["expert_ids"]):
                before[int(expert) // 32][int(expert)] += 1
                if slot not in removed:
                    after[int(expert) // 32][int(expert)] += 1
        reduction = max(len(value) for value in before) - max(len(value) for value in after)
        per_wave["-".join(map(str, key))] = float(reduction)
    return {"mean_max_rank_active_expert_reduction": float(np.mean(list(per_wave.values()))), "units": len(per_wave)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-label", default="DISCOVERY_ONLY_NOT_CONFIRMATION")
    args = parser.parse_args()
    rows = list(read_jsonl_gz(args.sites))
    persistence = add_persistence(rows)
    circuits = circuit_rows(rows)
    waves = wave_metrics(rows)

    h1 = {
        "mean_output_cosine": paired_effect(rows, "mean_pairwise_output_cosine", 1.0),
        "effective_rank_lower_is_redundant": paired_effect(rows, "output_effective_rank", -1.0),
        "largest_cluster": paired_effect(rows, "largest_cluster", 1.0),
        "reconstruction_residual_lower_is_redundant": paired_effect(rows, "mean_loo_reconstruction_residual", -1.0),
        "local_skip_k6": policy_gain(rows, "redundancy", 6),
        "local_skip_k4": policy_gain(rows, "redundancy", 4),
    }
    h2 = {
        "topk_persistence": paired_effect(persistence, "step_topk_jaccard", 1.0),
        "router_persistence": paired_effect(persistence, "step_router_cosine", 1.0),
        "rank_persistence": paired_effect(persistence, "step_rank_jaccard", 1.0),
        "local_skip_k6": policy_gain(rows, "temporal_persistence", 6),
        "local_skip_k4": policy_gain(rows, "temporal_persistence", 4),
    }
    h3 = {
        "motif_persistence": paired_effect(circuits, "motif_persistence_jaccard", 1.0),
        "local_skip_k6": policy_gain(rows, "cross_layer_circuit", 6),
        "local_skip_k4": policy_gain(rows, "cross_layer_circuit", 4),
    }
    h4 = {
        "singleton_experts": paired_effect(waves, "singleton_experts", 1.0),
        "rank_imbalance": paired_effect(waves, "rank_max_mean", 1.0),
        "working_set_hhi": paired_effect(waves, "working_set_hhi", 1.0),
        "rank_closure_k6": closure_gain(rows, "rank_closure", 6),
        "tail_closure_k6": closure_gain(rows, "router_tail", 6),
        "random_closure_k6": closure_gain(rows, "random", 6),
        "local_skip_k6": policy_gain(rows, "rank_closure", 6),
    }
    h5 = {
        "router_entropy": paired_effect(rows, "router_entropy", 1.0),
        "effective_k": paired_effect(rows, "topk_effective_k", 1.0),
        "topk_mass": paired_effect(rows, "full_topk_mass", 1.0),
        "router_tail_local_k6": {
            "mean": float(np.mean([row["router_tail_k6_local_relative_l2"] for row in rows if row["condition"] == "original"]))
        },
    }
    candidates = {
        "H1_REDUNDANCY": min(
            h1["effective_rank_lower_is_redundant"]["ci95"][0],
            h1["local_skip_k6"]["vs_router_tail"]["ci95"][0],
        ),
        "H2_PERSISTENCE": min(
            h2["topk_persistence"]["ci95"][0],
            h2["local_skip_k6"]["vs_router_tail"]["ci95"][0],
        ),
        "H3_CIRCUIT": min(
            h3["motif_persistence"]["ci95"][0],
            h3["local_skip_k6"]["vs_router_tail"]["ci95"][0],
        ),
        "H4_RANK_OCCUPANCY": min(
            h4["singleton_experts"]["ci95"][0],
            h4["local_skip_k6"]["vs_router_tail"]["ci95"][0],
        ),
    }
    strongest = max(candidates, key=candidates.get)
    payload = {
        "schema_version": 1,
        "evidence": args.evidence_label,
        "requests": len({row["request_id"] for row in rows}),
        "sites": len(rows),
        "h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5,
        "candidate_conservative_scores": candidates,
        "strongest_discovery_candidate": strongest,
        "candidate_passes_positive_lower_bounds": candidates[strongest] > 0,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"strongest": strongest, "scores": candidates}, indent=2))


if __name__ == "__main__":
    main()
