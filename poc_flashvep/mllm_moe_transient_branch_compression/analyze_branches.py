"""Analyze same-image expert-branch compression from observer-heavy captures.

This is an oracle/proxy analysis, not a latency or production-method claim. It
never deletes tokens and preserves every original router weight. A target
branch may reuse a representative branch output from the same expert; all
other branches and the token residual remain conceptually exact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LAYERS_DEFAULT = (4, 12, 24, 36, 44, 47)
GROUP_CAPS = (2, 3, 4, 6, 8)
HIDDEN_THRESHOLDS = (0.95, 0.98, 0.99, 0.995, 0.999)
BRANCH_COS = 0.99
BRANCH_REL = 0.10
COMBINED_REL = 0.01


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.einsum("ij,ij->i", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def pair_metrics(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cos = cosine_rows(a, b)
    rel = np.linalg.norm(a - b, axis=1) / np.maximum(
        (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)) / 2, 1e-12
    )
    return cos, rel


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_layer(result: Path, manifest: dict[str, Any], meta: dict[str, Any],
               layer: int) -> dict[str, np.ndarray]:
    sample = meta["sample_id"]
    source_rank = next(
        int(row["source_dp_rank"]) for row in manifest["schedule"]
        if row["sample_id"] == sample
    )
    prompt_tokens = int(meta["prompt_tokens"])
    router_parts = [
        np.load(result / "raw" / f"router.{sample}.dp{source_rank}.tp{tp}.layer{layer}.npz")
        for tp in (0, 1)
    ]
    router = {
        key: np.concatenate([part[key] for part in router_parts], axis=0)[:prompt_tokens]
        for key in ("fingerprints", "selected", "topk_ids", "topk_weights")
    }
    expert_rows: dict[str, list[tuple[np.ndarray, ...]]] = defaultdict(list)
    for ep_rank in range(4):
        arrays = np.load(result / "raw" / f"experts.{sample}.ep{ep_rank}.layer{layer}.npz")
        if "hidden_states" not in arrays:
            raise AssertionError("capture lacks hidden_states")
        for idx, fingerprint in enumerate(arrays["fingerprints"]):
            expert_rows[str(fingerprint)].append((
                arrays["expert_ids"][idx], arrays["router_weights"][idx],
                arrays["local_mask"][idx], arrays["raw_outputs"][idx],
                arrays["stock_output"][idx], arrays["hidden_states"][idx],
            ))

    positions = np.flatnonzero(router["selected"])
    hidden, ids, weights, outputs, stock, kept_positions = [], [], [], [], [], []
    for position in positions:
        fingerprint = str(router["fingerprints"][position])
        pieces = expert_rows.get(fingerprint, [])
        if not pieces:
            continue
        local_ids = router["topk_ids"][position].astype(np.int16)
        local_weights = router["topk_weights"][position].astype(np.float32)
        coverage = np.zeros(local_ids.shape[0], dtype=np.int16)
        raw = np.zeros((local_ids.shape[0], 2048), dtype=np.float32)
        stock_sum = np.zeros(2048, dtype=np.float32)
        token_hidden = None
        for p_ids, p_weights, local_mask, p_raw, p_stock, p_hidden in pieces:
            stock_sum += p_stock.astype(np.float32)
            token_hidden = p_hidden.astype(np.float32)
            for slot in np.flatnonzero(local_mask):
                if int(p_ids[slot]) != int(local_ids[slot]):
                    raise AssertionError((sample, layer, position, slot, "route"))
                if not np.isclose(p_weights[slot], local_weights[slot], atol=1e-5):
                    raise AssertionError((sample, layer, position, slot, "weight"))
                coverage[slot] += 1
                raw[slot] = p_raw[slot].astype(np.float32)
        if not np.all(coverage == 1) or token_hidden is None:
            raise AssertionError((sample, layer, position, coverage.tolist()))
        kept_positions.append(int(position))
        hidden.append(token_hidden)
        ids.append(local_ids)
        weights.append(local_weights)
        outputs.append(raw)
        stock.append(stock_sum)
    if not kept_positions:
        raise RuntimeError((sample, layer, "no sampled rows"))
    return {
        "position": np.asarray(kept_positions),
        "hidden": np.stack(hidden),
        "ids": np.stack(ids),
        "weights": normalize_weights(np.stack(weights)),
        "outputs": np.stack(outputs),
        "stock": np.stack(stock),
    }


def modality_and_coords(meta: dict[str, Any], positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start, end = map(int, meta["visual_span"])
    text_start, text_end = map(int, meta["post_visual_span"])
    modality = np.full(positions.shape, "other", dtype="<U6")
    modality[(positions >= start) & (positions < end)] = "vision"
    # Match the earlier capture protocol: special/control tokens before the visual
    # span are neither evidence for Vision nor the post-image Text control.
    modality[(positions >= text_start) & (positions < text_end)] = "text"
    grid_t, grid_h, grid_w = map(int, meta["image_grid_thw"])
    merge = int(meta["merge_size"])
    height, width = grid_t * (grid_h // merge), grid_w // merge
    index = positions - start
    row = np.where(modality == "vision", index // width, -1)
    col = np.where(modality == "vision", index % width, -1)
    return modality, np.stack([row, col], axis=1).astype(np.int32)


def branch_groups(data: dict[str, np.ndarray], modality: np.ndarray,
                  wanted: str) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    for expert in np.unique(data["ids"]):
        token, slot = np.where((data["ids"] == expert) & (modality[:, None] == wanted))
        if len(token) >= 2:
            yield int(expert), token.astype(np.int32), slot.astype(np.int32)


def nearest_pairs(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    similarity = norms @ norms.T
    np.fill_diagonal(similarity, -np.inf)
    source = np.arange(len(features), dtype=np.int32)
    target = np.argmax(similarity, axis=1).astype(np.int32)
    return source, target


def pair_rows(meta: dict[str, Any], layer: int, data: dict[str, np.ndarray],
              modality: np.ndarray, coords: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wanted in ("vision", "text"):
        for expert, token, slot in branch_groups(data, modality, wanted):
            hidden = data["hidden"][token]
            output = data["outputs"][token, slot]
            pair_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            pair_sets["hidden_nearest"] = nearest_pairs(hidden)
            pair_sets["output_nearest"] = nearest_pairs(output)
            rng = np.random.default_rng(1000003 + layer * 131 + expert)
            order = rng.permutation(len(token))
            pair_sets["arbitrary"] = (order[::2][:len(order)//2], order[1::2][:len(order)//2])
            if wanted == "vision":
                pos = data["position"][token]
                lookup = {int(value): idx for idx, value in enumerate(pos)}
                src, dst = [], []
                for idx, value in enumerate(pos):
                    if int(value + 1) in lookup:
                        src.append(idx); dst.append(lookup[int(value + 1)])
                pair_sets["contiguous_1d"] = (np.asarray(src), np.asarray(dst))
                c = coords[token]
                src, dst = [], []
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        if int(np.abs(c[i] - c[j]).sum()) == 1:
                            src.append(i); dst.append(j)
                pair_sets["neighbor_2d"] = (np.asarray(src), np.asarray(dst))
            for kind, (a, b) in pair_sets.items():
                if not len(a):
                    continue
                hcos, hrel = pair_metrics(hidden[a], hidden[b])
                ocos, orel = pair_metrics(output[a], output[b])
                for i in range(len(a)):
                    rows.append({
                        "sample_id": meta["sample_id"], "category": meta["category"],
                        "edge": meta.get("fixed_edge", meta.get("input_size", [0])[0]),
                        "layer": layer, "expert": expert, "modality": wanted,
                        "pair_kind": kind, "input_cosine": float(hcos[i]),
                        "input_relative_l2": float(hrel[i]),
                        "output_cosine": float(ocos[i]),
                        "output_relative_l2": float(orel[i]),
                        "contraction_ratio": float(orel[i] / max(hrel[i], 1e-12)),
                        "strict_branch_safe": bool(ocos[i] >= BRANCH_COS and orel[i] <= BRANCH_REL),
                    })
    return rows


def allowed_matrix(policy: str, hidden: np.ndarray, output: np.ndarray,
                   positions: np.ndarray, coords: np.ndarray, routes: np.ndarray,
                   hidden_threshold: float | None) -> tuple[np.ndarray, np.ndarray]:
    hn = hidden / np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-12)
    on = output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    hsim = hn @ hn.T
    osim = on @ on.T
    if policy == "output_oracle":
        onorm = np.linalg.norm(output, axis=1)
        rel = np.linalg.norm(output[:, None] - output[None, :], axis=2) / np.maximum(
            (onorm[:, None] + onorm[None, :]) / 2, 1e-12)
        allowed = (osim >= BRANCH_COS) & (rel <= BRANCH_REL)
        distance = 1 - osim
    elif policy == "hidden_sim":
        allowed = hsim >= float(hidden_threshold)
        distance = 1 - hsim
    elif policy == "routing_sim":
        overlap = np.empty((len(routes), len(routes)), dtype=np.float32)
        sets = [set(map(int, row)) for row in routes]
        for i in range(len(sets)):
            for j in range(len(sets)):
                overlap[i, j] = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
        allowed = overlap >= 0.6
        if hidden_threshold is not None:
            allowed &= hsim >= hidden_threshold
        distance = 1 - overlap
    elif policy == "contiguous_1d":
        allowed = np.abs(positions[:, None] - positions[None, :]) == 1
        if hidden_threshold is not None:
            allowed &= hsim >= hidden_threshold
        distance = 1 - hsim
    elif policy == "neighbor_2d":
        manhattan = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
        allowed = manhattan == 1
        if hidden_threshold is not None:
            allowed &= hsim >= hidden_threshold
        distance = 1 - hsim
    elif policy == "window_2x2":
        blocks = coords // 2
        allowed = np.all(blocks[:, None, :] == blocks[None, :, :], axis=2)
        if hidden_threshold is not None:
            allowed &= hsim >= hidden_threshold
        distance = 1 - hsim
    elif policy == "random":
        allowed = np.ones((len(hidden), len(hidden)), dtype=bool)
        rng = np.random.default_rng(1777 + len(hidden))
        distance = rng.random((len(hidden), len(hidden)))
    else:
        raise ValueError(policy)
    np.fill_diagonal(allowed, False)
    np.fill_diagonal(distance, np.inf)
    return allowed, distance


def greedy_groups(allowed: np.ndarray, distance: np.ndarray, cap: int) -> list[list[int]]:
    remaining = set(range(len(allowed)))
    groups: list[list[int]] = []
    while remaining:
        candidates = []
        for anchor in remaining:
            neighbors = [idx for idx in remaining if allowed[anchor, idx]]
            candidates.append((len(neighbors), -float(np.sum(distance[anchor, neighbors])) if neighbors else 0, anchor))
        _, _, anchor = max(candidates)
        neighbors = sorted(
            (idx for idx in remaining if idx != anchor and allowed[anchor, idx]),
            key=lambda idx: float(distance[anchor, idx]),
        )[:cap - 1]
        group = [anchor, *neighbors]
        groups.append(group)
        remaining.difference_update(group)
    return groups


def choose_representative(group: list[int], feature: np.ndarray) -> int:
    if len(group) == 1:
        return group[0]
    local = feature[group]
    norm = local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
    distance = 1 - norm @ norm.T
    return group[int(np.argmin(distance.sum(axis=1)))]


def evaluate_policy(meta: dict[str, Any], layer: int, data: dict[str, np.ndarray],
                    modality: np.ndarray, coords: np.ndarray, wanted: str,
                    policy: str, cap: int, representative: str,
                    hidden_threshold: float | None) -> dict[str, Any]:
    approx = data["outputs"].copy()
    replaced = np.zeros(data["ids"].shape, dtype=bool)
    groups_total = 0
    size_hist: dict[int, int] = defaultdict(int)
    for _, token, slot in branch_groups(data, modality, wanted):
        hidden = data["hidden"][token]
        output = data["outputs"][token, slot]
        routes = data["ids"][token]
        allowed, distance = allowed_matrix(
            policy, hidden, output, data["position"][token], coords[token], routes,
            hidden_threshold,
        )
        for group in greedy_groups(allowed, distance, cap):
            size_hist[len(group)] += 1
            groups_total += 1
            feature = output if representative == "output_medoid" else hidden
            rep = choose_representative(group, feature) if representative != "anchor" else group[0]
            for local_idx in group:
                if local_idx == rep:
                    continue
                approx[token[local_idx], slot[local_idx]] = output[rep]
                replaced[token[local_idx], slot[local_idx]] = True

    exact = np.einsum("tk,tkh->th", data["weights"], data["outputs"])
    predicted = np.einsum("tk,tkh->th", data["weights"], approx)
    rel = np.linalg.norm(predicted - exact, axis=1) / np.maximum(np.linalg.norm(exact, axis=1), 1e-12)
    cos = cosine_rows(predicted, exact)
    eligible_token = modality == wanted
    pass_mask = (rel <= COMBINED_REL) & (cos >= 0.9999)
    safe_replaced = replaced.copy()
    safe_replaced[~pass_mask] = False
    target_mask = np.broadcast_to(eligible_token[:, None], replaced.shape)
    total = int(target_mask.sum())
    pre = int((replaced & target_mask).sum())
    safe = int((safe_replaced & target_mask).sum())
    per_rank_total = [int(((data["ids"] // 32 == rank) & target_mask).sum()) for rank in range(4)]
    per_rank_safe = [int(((data["ids"] // 32 == rank) & safe_replaced).sum()) for rank in range(4)]
    local_rel = rel[eligible_token]
    local_cos = cos[eligible_token]
    return {
        "sample_id": meta["sample_id"], "category": meta["category"],
        "edge": meta.get("fixed_edge", meta.get("input_size", [0])[0]),
        "layer": layer, "modality": wanted, "policy": policy,
        "representative": representative, "group_cap": cap,
        "hidden_threshold": hidden_threshold, "assignments": total,
        "representative_branches_pre_gate": total - pre,
        "branch_reduction_pre_gate": pre / max(total, 1),
        "safe_compressible_assignments": safe,
        "representative_branches_safe": total - safe,
        "safe_branch_reduction": safe / max(total, 1),
        "combined_pass_fraction_pre_revert": float(pass_mask[eligible_token].mean()),
        "combined_rel_l2_median": float(np.median(local_rel)),
        "combined_rel_l2_p90": float(np.quantile(local_rel, 0.9)),
        "combined_cosine_median": float(np.median(local_cos)),
        "groups": groups_total,
        "groups_ge2": int(sum(value for size, value in size_hist.items() if size >= 2)),
        "groups_ge4": int(sum(value for size, value in size_hist.items() if size >= 4)),
        "per_rank_reduction": json.dumps([
            1 - (per_rank_total[i] - per_rank_safe[i]) / max(per_rank_total[i], 1)
            for i in range(4)
        ]),
        "group_size_histogram": json.dumps(dict(sorted(size_hist.items()))),
    }


def rle_rows(meta: dict[str, Any], layer: int, data: dict[str, np.ndarray],
             modality: np.ndarray, coords: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for expert, token, _ in branch_groups(data, modality, "vision"):
        pos = data["position"][token]
        ordered = np.argsort(pos)
        runs: list[int] = []
        current = 1
        for before, after in zip(pos[ordered][:-1], pos[ordered][1:], strict=True):
            if after == before + 1:
                current += 1
            else:
                runs.append(current); current = 1
        runs.append(current)
        c = coords[token]
        neighbors = [set() for _ in range(len(token))]
        for i in range(len(token)):
            for j in range(i + 1, len(token)):
                if int(np.abs(c[i] - c[j]).sum()) == 1:
                    neighbors[i].add(j); neighbors[j].add(i)
        components: list[int] = []
        remaining = set(range(len(token)))
        while remaining:
            root = remaining.pop(); stack = [root]; size = 1
            while stack:
                for other in neighbors[stack.pop()] & remaining:
                    remaining.remove(other); stack.append(other); size += 1
            components.append(size)
        rows.append({
            "sample_id": meta["sample_id"], "layer": layer, "expert": expert,
            "assignments": len(token), "rle_runs": json.dumps(runs),
            "rle_fraction_ge2": sum(v for v in runs if v >= 2) / len(token),
            "rle_fraction_ge4": sum(v for v in runs if v >= 4) / len(token),
            "cc_sizes": json.dumps(components),
            "cc_fraction_ge2": sum(v for v in components if v >= 2) / len(token),
            "cc_fraction_ge4": sum(v for v in components if v >= 4) / len(token),
        })
    return rows


def aggregate(rows: list[dict[str, Any]], value: str,
              keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, local in sorted(groups.items(), key=lambda item: str(item[0])):
        values = np.asarray([float(row[value]) for row in local])
        output.append({**dict(zip(keys, key, strict=True)), "n": len(values),
                       f"{value}_mean": float(values.mean()),
                       f"{value}_median": float(np.median(values)),
                       f"{value}_p10": float(np.quantile(values, .1)),
                       f"{value}_p90": float(np.quantile(values, .9))})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", default="")
    args = parser.parse_args()
    result = args.result_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((result / "manifest.json").read_text())
    layers = tuple(int(x) for x in args.layers.split(",") if x) or tuple(
        int(x) for x in manifest["policy"]["layers"]
    )
    pairs: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    rles: list[dict[str, Any]] = []
    reconstruction_checks = []
    for meta in manifest["samples"]:
        for layer in layers:
            data = load_layer(result, manifest, meta, layer)
            modality, coords = modality_and_coords(meta, data["position"])
            exact = np.einsum("tk,tkh->th", data["weights"], data["outputs"])
            check_cos = cosine_rows(exact, data["stock"])
            check_rel = np.linalg.norm(exact - data["stock"], axis=1) / np.maximum(
                np.linalg.norm(data["stock"], axis=1), 1e-12
            )
            reconstruction_checks.append({
                "sample_id": meta["sample_id"], "layer": layer,
                "tokens": len(data["position"]), "min_cosine": float(check_cos.min()),
                "median_cosine": float(np.median(check_cos)),
                "max_relative_l2": float(check_rel.max()),
                "median_relative_l2": float(np.median(check_rel)),
            })
            pairs.extend(pair_rows(meta, layer, data, modality, coords))
            rles.extend(rle_rows(meta, layer, data, modality, coords))
            for wanted in ("vision", "text"):
                if not np.any(modality == wanted):
                    continue
                for cap in GROUP_CAPS:
                    policies.append(evaluate_policy(
                        meta, layer, data, modality, coords, wanted,
                        "output_oracle", cap, "output_medoid", None,
                    ))
                    policies.append(evaluate_policy(
                        meta, layer, data, modality, coords, wanted,
                        "random", cap, "anchor", None,
                    ))
                    for threshold in HIDDEN_THRESHOLDS:
                        policies.append(evaluate_policy(
                            meta, layer, data, modality, coords, wanted,
                            "hidden_sim", cap, "hidden_medoid", threshold,
                        ))
                        policies.append(evaluate_policy(
                            meta, layer, data, modality, coords, wanted,
                            "routing_sim", cap, "hidden_medoid", threshold,
                        ))
                        if wanted == "vision":
                            for policy in ("contiguous_1d", "neighbor_2d", "window_2x2"):
                                policies.append(evaluate_policy(
                                    meta, layer, data, modality, coords, wanted,
                                    policy, cap, "hidden_medoid", threshold,
                                ))
            print(json.dumps({"sample": meta["sample_id"], "layer": layer,
                              "sampled_tokens": len(data["position"])}), flush=True)
    write_csv(out / "pair_atlas.csv", pairs)
    write_csv(out / "compression_policies.csv", policies)
    write_csv(out / "route_runs.csv", rles)
    write_csv(out / "capture_reconstruction.csv", reconstruction_checks)
    pair_summary = aggregate(pairs, "output_relative_l2", ("modality", "pair_kind"))
    pair_summary += aggregate(pairs, "contraction_ratio", ("modality", "pair_kind"))
    policy_summary = aggregate(
        policies, "safe_branch_reduction",
        ("modality", "policy", "representative", "group_cap", "hidden_threshold"),
    )
    write_csv(out / "pair_summary.csv", pair_summary)
    write_csv(out / "policy_summary.csv", policy_summary)
    write_json(out / "summary.json", {
        "source": str(result), "layers": layers,
        "samples": len(manifest["samples"]), "pair_rows": len(pairs),
        "policy_rows": len(policies), "rle_rows": len(rles),
        "capture_correctness": {
            "min_cosine": min(row["min_cosine"] for row in reconstruction_checks),
            "max_relative_l2": max(row["max_relative_l2"] for row in reconstruction_checks),
        },
        "thresholds": {"branch_cosine": BRANCH_COS, "branch_relative_l2": BRANCH_REL,
                       "combined_relative_l2": COMBINED_REL, "combined_cosine": .9999},
    })


if __name__ == "__main__":
    main()
