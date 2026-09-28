from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from ..contract import MODEL_PATH, TOP_K, read_json, write_jsonl_gz

SITE_RE = re.compile(r"site\.(.+)\.dp\d+\.layer(\d+)\.step(\d+)\.call\d+\.npz$")


def cosine_matrix(rows: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    normalized = rows / norms
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def effective_rank(rows: np.ndarray) -> tuple[float, float, float]:
    values = np.linalg.svd(rows.astype(np.float64), compute_uv=False) ** 2
    probs = values / max(float(values.sum()), 1e-30)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-30))))
    return float(np.exp(entropy)), float(probs[0]), float(probs[:2].sum())


def largest_cluster(cosines: np.ndarray, threshold: float = 0.95) -> tuple[int, int]:
    parent = list(range(len(cosines)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left in range(len(cosines)):
        for right in range(left + 1, len(cosines)):
            if cosines[left, right] >= threshold:
                lroot, rroot = find(left), find(right)
                parent[rroot] = lroot
    sizes: dict[int, int] = defaultdict(int)
    for value in range(len(cosines)):
        sizes[find(value)] += 1
    return len(sizes), max(sizes.values())


def reconstruction_residuals(rows: np.ndarray) -> list[float]:
    result = []
    values = rows.astype(np.float64)
    for index in range(len(values)):
        target = values[index]
        others = np.delete(values, index, axis=0).T
        coeff, *_ = np.linalg.lstsq(others, target, rcond=1e-5)
        reconstruction = others @ coeff
        result.append(float(np.linalg.norm(target - reconstruction) / max(np.linalg.norm(target), 1e-12)))
    return result


def local_damage(outputs: np.ndarray, weights: np.ndarray, removed: set[int]) -> dict[str, float]:
    baseline = np.sum(weights[:, None] * outputs, axis=0)
    keep = np.asarray([index not in removed for index in range(len(weights))])
    mass = float(weights[keep].sum())
    changed = np.sum((weights[keep] / max(mass, 1e-12))[:, None] * outputs[keep], axis=0)
    denominator = max(float(np.linalg.norm(baseline)), 1e-12)
    cosine = float(np.dot(baseline, changed) / max(np.linalg.norm(baseline) * np.linalg.norm(changed), 1e-12))
    return {"relative_l2": float(np.linalg.norm(baseline - changed) / denominator), "cosine": cosine}


def redundancy_removals(cosines: np.ndarray, weights: np.ndarray, count: int) -> set[int]:
    retained = set(range(len(weights)))
    removed: set[int] = set()
    while len(removed) < count:
        pairs = [
            (float(cosines[left, right]), left, right)
            for left in retained for right in retained if left < right
        ]
        _, left, right = max(pairs)
        victim = left if weights[left] <= weights[right] else right
        retained.remove(victim)
        removed.add(victim)
    return removed


def random_removals(site_id: str, count: int) -> set[int]:
    seed = int(hashlib.sha256(site_id.encode()).hexdigest()[:16], 16)
    return set(np.random.default_rng(seed).choice(TOP_K, count, replace=False).tolist())


def scored_removals(scores: list[float], weights: np.ndarray, count: int) -> set[int]:
    order = sorted(range(TOP_K), key=lambda slot: (-scores[slot], float(weights[slot]), -slot))
    return set(order[:count])


def parse_sites(trace: Path, manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    selected_cases = [case for case in manifest["cases"] if case["split"] == split]
    cases = {case["case_id"]: case for case in selected_cases}
    waves = {case["case_id"]: index // 8 for index, case in enumerate(selected_cases)}
    rows = []
    for path in sorted((trace / "raw").glob("site.*.npz")):
        match = SITE_RE.match(path.name)
        if not match:
            continue
        capture_id, layer, step = match.group(1), int(match.group(2)), int(match.group(3))
        case_id, condition = capture_id.rsplit(".", 1)
        if case_id not in cases:
            raise RuntimeError(case_id)
        with np.load(path) as payload:
            rows.append({
                "site_id": f"{capture_id}.l{layer}.s{step}",
                "capture_id": capture_id,
                "case_id": case_id,
                "request_id": cases[case_id]["request_id"],
                "domain": cases[case_id]["domain"],
                "split": split,
                "condition": condition,
                "layer": layer,
                "decode_step": step,
                "wave_b8": waves[case_id],
                "hidden": payload["hidden"][0].astype(np.float32),
                "hidden_fingerprint": str(payload["hidden_fingerprint"]),
                "router_logits": payload["router_logits"][0].astype(np.float32),
                "expert_ids": payload["topk_ids"][0].astype(int),
                "weights": payload["topk_weights"][0].astype(np.float32),
            })
    return rows


def add_structural_context(sites: list[dict[str, Any]]) -> None:
    lookup = {
        (row["case_id"], row["condition"], row["layer"], row["decode_step"]): row
        for row in sites
    }
    layers = sorted({row["layer"] for row in sites})
    steps = sorted({row["decode_step"] for row in sites})
    wave_counts: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    for row in sites:
        key = (row["condition"], row["wave_b8"], row["layer"], row["decode_step"])
        wave_counts[key].update(int(value) for value in row["expert_ids"])
    for row in sites:
        previous_steps = [value for value in steps if value < row["decode_step"]]
        previous_layers = [value for value in layers if value < row["layer"]]
        previous = lookup.get(
            (row["case_id"], row["condition"], row["layer"], previous_steps[-1])
        ) if previous_steps else None
        lower = lookup.get(
            (row["case_id"], row["condition"], previous_layers[-1], row["decode_step"])
        ) if previous_layers else None
        temporal = set() if previous is None else {int(value) for value in previous["expert_ids"]}
        cross_layer = set() if lower is None else {int(value) for value in lower["expert_ids"]}
        motif_frequency: Counter[int] = Counter()
        if previous_layers:
            lower_layer = previous_layers[-1]
            for step in steps:
                if step > row["decode_step"]:
                    continue
                current = lookup.get((row["case_id"], row["condition"], row["layer"], step))
                prior = lookup.get((row["case_id"], row["condition"], lower_layer, step))
                if current is None or prior is None:
                    continue
                prior_ids = {int(value) for value in prior["expert_ids"]}
                for expert in current["expert_ids"]:
                    motif_frequency[int(expert)] += len(prior_ids)
        counts = wave_counts[(row["condition"], row["wave_b8"], row["layer"], row["decode_step"])]
        active_per_rank = Counter(expert // 32 for expert in counts)
        bottleneck = max(active_per_rank, key=lambda rank: (active_per_rank[rank], -rank))
        row["temporal_scores"] = [float(int(expert) in temporal) for expert in row["expert_ids"]]
        row["cross_layer_scores"] = [
            float(int(expert) in cross_layer) + float(motif_frequency[int(expert)]) / 64.0
            for expert in row["expert_ids"]
        ]
        row["rank_closure_scores"] = [
            2.0 * float(counts[int(expert)] == 1) + float(int(expert) // 32 == bottleneck)
            for expert in row["expert_ids"]
        ]
        row["wave_expert_frequency"] = [int(counts[int(expert)]) for expert in row["expert_ids"]]
        row["wave_bottleneck_rank"] = int(bottleneck)


def load_weight(model: Path, index: dict[str, str], layer: int, expert: int, device: str):
    gate_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    down_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    with safe_open(model / index[gate_key], framework="pt", device="cpu") as handle:
        gate_up = handle.get_slice(gate_key)[expert].to(device)
    with safe_open(model / index[down_key], framework="pt", device="cpu") as handle:
        down = handle.get_slice(down_key)[expert].to(device)
    return gate_up, down


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--site-output", type=Path, required=True)
    parser.add_argument("--path-output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sites = parse_sites(args.trace, read_json(args.manifest), args.split)
    add_structural_context(sites)
    index = read_json(args.model_path / "model.safetensors.index.json")["weight_map"]
    site_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for layer in sorted({row["layer"] for row in sites}):
        layer_sites = [row for row in sites if row["layer"] == layer]
        outputs = np.empty((len(layer_sites), TOP_K, 2048), dtype=np.float16)
        locations: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for site_index, row in enumerate(layer_sites):
            for slot, expert in enumerate(row["expert_ids"]):
                locations[int(expert)].append((site_index, slot))
        for expert, pairs in sorted(locations.items()):
            gate_up, down = load_weight(args.model_path, index, layer, expert, args.device)
            hidden = torch.from_numpy(np.stack([layer_sites[i]["hidden"] for i, _ in pairs])).to(
                args.device, dtype=torch.bfloat16
            )
            with torch.inference_mode():
                projected = hidden @ gate_up
                gate, up = projected[:, :768], projected[:, 768:]
                value = (torch.nn.functional.silu(gate) * up) @ down
            values = value.to(torch.float16).cpu().numpy()
            for output, (site_index, slot) in zip(values, pairs):
                outputs[site_index, slot] = output
            del gate_up, down, hidden, projected, value

        for site_index, row in enumerate(layer_sites):
            raw = outputs[site_index].astype(np.float32)
            weighted = row["weights"][:, None] * raw
            raw_cosine = cosine_matrix(raw)
            weighted_cosine = cosine_matrix(weighted)
            residuals = reconstruction_residuals(raw)
            raw_erank, raw_top1, raw_top2 = effective_rank(raw)
            weighted_erank, weighted_top1, weighted_top2 = effective_rank(weighted)
            clusters, largest = largest_cluster(raw_cosine)
            logits = row["router_logits"].astype(np.float64)
            logits -= float(np.max(logits))
            probs = np.exp(logits); probs /= probs.sum()
            entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-30))))
            weight_probs = row["weights"] / row["weights"].sum()
            site = {
                **{key: value for key, value in row.items() if key not in ("hidden", "router_logits", "weights", "expert_ids")},
                "expert_ids": row["expert_ids"].tolist(),
                "weights": row["weights"].tolist(),
                "router_probs": probs.astype(np.float32).tolist(),
                "owner_ranks_ep4": (row["expert_ids"] // 32).tolist(),
                "wave_b8": row["wave_b8"],
                "wave_expert_frequency": row["wave_expert_frequency"],
                "wave_bottleneck_rank": row["wave_bottleneck_rank"],
                "temporal_scores": row["temporal_scores"],
                "cross_layer_scores": row["cross_layer_scores"],
                "rank_closure_scores": row["rank_closure_scores"],
                "router_entropy": entropy,
                "top1_top2_margin": float(row["weights"][0] - row["weights"][1]),
                "topk_effective_k": float(1.0 / np.sum(weight_probs ** 2)),
                "full_topk_mass": float(np.sort(probs)[-TOP_K:].sum()),
                "mean_pairwise_output_cosine": float(raw_cosine[np.triu_indices(TOP_K, 1)].mean()),
                "max_pairwise_output_cosine": float(raw_cosine[np.triu_indices(TOP_K, 1)].max()),
                "mean_pairwise_weighted_cosine": float(weighted_cosine[np.triu_indices(TOP_K, 1)].mean()),
                "output_effective_rank": raw_erank,
                "weighted_effective_rank": weighted_erank,
                "output_top1_energy": raw_top1,
                "output_top2_energy": raw_top2,
                "weighted_top1_energy": weighted_top1,
                "weighted_top2_energy": weighted_top2,
                "cluster_threshold": 0.95,
                "near_duplicate_clusters": clusters,
                "largest_cluster": largest,
                "mean_loo_reconstruction_residual": float(np.mean(residuals)),
            }
            for keep in (6, 4):
                count = TOP_K - keep
                policies = {
                    "router_tail": set(range(keep, TOP_K)),
                    "random": random_removals(row["site_id"], count),
                    "redundancy": redundancy_removals(raw_cosine, row["weights"], count),
                    "temporal_persistence": scored_removals(row["temporal_scores"], row["weights"], count),
                    "cross_layer_circuit": scored_removals(row["cross_layer_scores"], row["weights"], count),
                    "rank_closure": scored_removals(row["rank_closure_scores"], row["weights"], count),
                    "temporal_rank": scored_removals([
                        value + 0.1 * rank for value, rank in zip(row["temporal_scores"], row["rank_closure_scores"])
                    ], row["weights"], count),
                    "circuit_rank": scored_removals([
                        value + 0.1 * rank for value, rank in zip(row["cross_layer_scores"], row["rank_closure_scores"])
                    ], row["weights"], count),
                }
                for name, removed in policies.items():
                    metric = local_damage(raw, row["weights"], removed)
                    site[f"{name}_k{keep}_removed_slots"] = sorted(value + 1 for value in removed)
                    site[f"{name}_k{keep}_local_relative_l2"] = metric["relative_l2"]
                    site[f"{name}_k{keep}_local_cosine"] = metric["cosine"]
            site_rows.append(site)
            for slot, expert in enumerate(row["expert_ids"]):
                others = np.delete(raw_cosine[slot], slot)
                other_indices = [value for value in range(TOP_K) if value != slot]
                partner_position = int(np.argmax(others))
                partner = other_indices[partner_position]
                path_rows.append({
                    "site_id": row["site_id"], "capture_id": row["capture_id"],
                    "case_id": row["case_id"], "request_id": row["request_id"],
                    "domain": row["domain"], "split": row["split"], "condition": row["condition"],
                    "layer": layer, "decode_step": row["decode_step"], "slot_rank": slot + 1,
                    "expert_id": int(expert), "owner_rank_ep4": int(expert) // 32,
                    "router_weight": float(row["weights"][slot]),
                    "max_output_cosine": float(others[partner_position]),
                    "most_similar_expert_id": int(row["expert_ids"][partner]),
                    "most_similar_owner_rank_ep4": int(row["expert_ids"][partner]) // 32,
                    "loo_reconstruction_residual": residuals[slot],
                    "weighted_contribution_norm": float(np.linalg.norm(weighted[slot])),
                    "temporal_persistence_score": row["temporal_scores"][slot],
                    "cross_layer_circuit_score": row["cross_layer_scores"][slot],
                    "wave_expert_frequency_b8": row["wave_expert_frequency"][slot],
                    "rank_closure_score": row["rank_closure_scores"][slot],
                })
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_jsonl_gz(args.site_output, site_rows)
    write_jsonl_gz(args.path_output, path_rows)
    print(json.dumps({"sites": len(site_rows), "paths": len(path_rows), "split": args.split}, indent=2))


if __name__ == "__main__":
    main()
