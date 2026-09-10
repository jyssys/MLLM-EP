#!/usr/bin/env python3
"""Offline quality/load oracle for visual-token heterogeneous Top-K.

The input is a prior observer-heavy capture of real Qwen3-VL hidden states,
Top-8 routes, router weights, and individual expert outputs.  The analysis is
strictly an oracle/screen: it does not claim runtime or end-to-end speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    load_layer,
    modality_and_coords,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_load(ids: np.ndarray, keep: np.ndarray) -> np.ndarray:
    ranks = ids // 32
    return np.asarray([np.count_nonzero(keep & (ranks == r)) for r in range(4)], dtype=np.int64)


def vision_slots(modality: np.ndarray, width: int = 8) -> np.ndarray:
    return np.broadcast_to((modality == "vision")[:, None], (len(modality), width)).copy()


def random_keep(modality: np.ndarray, k: int, seed: int) -> np.ndarray:
    keep = np.ones((len(modality), 8), dtype=bool)
    rng = np.random.default_rng(seed)
    for token in np.flatnonzero(modality == "vision"):
        keep[token, rng.choice(8, size=8 - k, replace=False)] = False
    return keep


def semantic_keep(weights: np.ndarray, modality: np.ndarray, k: int) -> np.ndarray:
    keep = np.ones_like(weights, dtype=bool)
    for token in np.flatnonzero(modality == "vision"):
        order = np.argsort(weights[token], kind="stable")
        keep[token, order[: 8 - k]] = False
    return keep


def tail_aware_keep(ids: np.ndarray, weights: np.ndarray, modality: np.ndarray, k: int,
                    importance_slack: float) -> np.ndarray:
    """Greedily remove cheap assignments from the current critical EP rank.

    Every visual token keeps exactly ``k`` branches.  At each round a token may
    remove only a branch within ``importance_slack`` of its cheapest remaining
    branch.  Among those low-risk candidates, assignments on the currently
    most-loaded rank are preferred.  This explicitly separates semantic risk
    from critical-rank placement without changing assignment count.
    """
    keep = np.ones_like(weights, dtype=bool)
    visual_tokens = np.flatnonzero(modality == "vision")
    # One synchronous removal round per unit of K reduction.  Updating the
    # global rank load between rounds retains the critical-rank feedback while
    # avoiding an artificial O(assignments^2) oracle implementation.
    for _ in range(8 - k):
        loads = rank_load(ids, keep)
        max_load = int(loads.max())
        choices: list[tuple[int, int]] = []
        for token in visual_tokens:
            slots = np.flatnonzero(keep[token])
            minimum = float(weights[token, slots].min())
            allowed = slots[weights[token, slots] <= minimum + importance_slack]
            if not len(allowed):
                allowed = slots[np.argsort(weights[token, slots], kind="stable")[:1]]
            candidates: list[tuple[int, float, float, int]] = []
            for slot in allowed:
                rank = int(ids[token, slot] // 32)
                is_critical = int(loads[rank] == max_load)
                # Prefer critical rank, then higher load, then lower weight.
                candidates.append((-is_critical, -float(loads[rank]),
                                   float(weights[token, slot]), int(slot)))
            if not candidates:
                raise RuntimeError("tail-aware policy ran out of candidates")
            choices.append((int(token), min(candidates)[-1]))
        for token, slot in choices:
            keep[token, slot] = False
    return keep


def budget_keep(ids: np.ndarray, weights: np.ndarray, modality: np.ndarray,
                fraction: float, policy: str, seed: int = 20260911) -> np.ndarray:
    """Drop an exact fraction of visual assignments while retaining Top-1.

    ``semantic`` chooses globally lowest router weights. ``tail`` uses a
    four-way load-balancing heap in spirit: it repeatedly drains the currently
    most-loaded rank, selecting that rank's cheapest remaining eligible
    assignment. This is an optimistic, runtime-visible routing oracle.
    """
    keep = np.ones_like(weights, dtype=bool)
    visual = np.flatnonzero(modality == "vision")
    target = int(round(len(visual) * 8 * fraction))
    eligible: list[tuple[float, int, int, int]] = []
    rng = np.random.default_rng(seed)
    for token in visual:
        protected = int(np.argmax(weights[token]))
        for slot in range(8):
            if slot == protected:
                continue
            rank = int(ids[token, slot] // 32)
            score = float(weights[token, slot])
            eligible.append((score, int(token), slot, rank))
    if target > len(eligible):
        raise ValueError((fraction, target, len(eligible)))
    if policy == "random":
        chosen = [eligible[i] for i in rng.choice(len(eligible), size=target, replace=False)]
    elif policy == "semantic":
        chosen = sorted(eligible)[:target]
    elif policy == "tail":
        per_rank = {rank: sorted((row for row in eligible if row[3] == rank))
                    for rank in range(4)}
        index = {rank: 0 for rank in range(4)}
        loads = rank_load(ids, keep).astype(np.int64)
        chosen = []
        for _ in range(target):
            available = [rank for rank in range(4) if index[rank] < len(per_rank[rank])]
            if not available:
                raise RuntimeError("budget tail oracle exhausted")
            max_load = max(loads[rank] for rank in available)
            critical = [rank for rank in available if loads[rank] == max_load]
            rank = min(critical, key=lambda r: per_rank[r][index[r]][0])
            chosen.append(per_rank[rank][index[rank]])
            index[rank] += 1
            loads[rank] -= 1
    else:
        raise ValueError(policy)
    for _, token, slot, _ in chosen:
        keep[token, slot] = False
    return keep


def policy_metrics(data: dict[str, np.ndarray], modality: np.ndarray, keep: np.ndarray,
                   policy: str, target_k: int, sample: dict[str, Any], layer: int,
                   replicate: int, importance_slack: float) -> dict[str, Any]:
    ids, weights, outputs = data["ids"], data["weights"], data["outputs"]
    baseline = np.einsum("mk,mkh->mh", weights, outputs, optimize=True)
    candidate = np.einsum("mk,mkh->mh", weights * keep, outputs, optimize=True)
    delta = candidate - baseline
    base_norm = np.linalg.norm(baseline, axis=1)
    rel_per_token = np.linalg.norm(delta, axis=1) / np.maximum(base_norm, 1e-12)
    cos_per_token = np.einsum("mh,mh->m", candidate, baseline) / np.maximum(
        np.linalg.norm(candidate, axis=1) * base_norm, 1e-12
    )
    before = rank_load(ids, np.ones_like(keep, dtype=bool))
    after = rank_load(ids, keep)
    vis = vision_slots(modality)
    dropped = (~keep) & vis
    total_vis = int(vis.sum())
    weighted_total = float((weights * vis).sum())
    vision_rows = modality == "vision"
    return {
        "sample_id": sample["sample_id"], "category": sample["category"],
        "edge": sample["fixed_edge"], "layer": layer, "policy": policy,
        "target_k": target_k, "replicate": replicate,
        "importance_slack": importance_slack,
        "tokens": len(modality), "vision_tokens": int(vision_rows.sum()),
        "assignments_dropped": int(dropped.sum()),
        "vision_assignment_reduction": float(dropped.sum() / max(total_vis, 1)),
        "dropped_router_mass": float((weights * dropped).sum() / max(weighted_total, 1e-12)),
        "rank0_before": int(before[0]), "rank1_before": int(before[1]),
        "rank2_before": int(before[2]), "rank3_before": int(before[3]),
        "rank0_after": int(after[0]), "rank1_after": int(after[1]),
        "rank2_after": int(after[2]), "rank3_after": int(after[3]),
        "max_rank_before": int(before.max()), "max_rank_after": int(after.max()),
        "max_rank_reduction": float(1 - after.max() / max(before.max(), 1)),
        "mean_rank_reduction": float(1 - after.mean() / max(before.mean(), 1e-12)),
        "rank_cv_before": float(before.std() / max(before.mean(), 1e-12)),
        "rank_cv_after": float(after.std() / max(after.mean(), 1e-12)),
        "combined_rel_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(baseline), 1e-12)),
        "vision_combined_rel_l2": float(np.linalg.norm(delta[vision_rows]) /
                                         max(np.linalg.norm(baseline[vision_rows]), 1e-12)),
        "combined_cosine": float(np.dot(candidate.ravel(), baseline.ravel()) /
                                  max(np.linalg.norm(candidate) * np.linalg.norm(baseline), 1e-12)),
        "vision_token_cos_p01": float(np.quantile(cos_per_token[vision_rows], .01)),
        "vision_token_rel_p50": float(np.median(rel_per_token[vision_rows])),
        "vision_token_rel_p99": float(np.quantile(rel_per_token[vision_rows], .99)),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(r["policy"], r["target_k"], r["importance_slack"]) for r in rows})
    metrics = ("vision_assignment_reduction", "dropped_router_mass", "max_rank_reduction",
               "mean_rank_reduction", "combined_rel_l2", "vision_combined_rel_l2",
               "combined_cosine", "vision_token_cos_p01", "vision_token_rel_p99")
    output = []
    for policy, target_k, slack in keys:
        selected = [r for r in rows if (r["policy"], r["target_k"], r["importance_slack"])
                    == (policy, target_k, slack)]
        row: dict[str, Any] = {"policy": policy, "target_k": target_k,
                               "importance_slack": slack, "n": len(selected)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in selected])
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_p10"] = float(np.quantile(values, .1))
            row[f"{metric}_p90"] = float(np.quantile(values, .9))
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 12, 24, 36, 44, 47])
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 6, 4, 2, 1])
    parser.add_argument("--random-replicates", type=int, default=5)
    parser.add_argument("--importance-slacks", type=float, nargs="+", default=[0.0, .005, .01, .02])
    parser.add_argument("--budgets", type=float, nargs="+", default=[.1, .2, .3, .4, .5])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    rows: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        for layer in args.layers:
            data = load_layer(args.capture, manifest, sample, layer)
            modality, _ = modality_and_coords(sample, data["position"])
            for k in args.ks:
                semantic = semantic_keep(data["weights"], modality, k)
                rows.append(policy_metrics(data, modality, semantic, "semantic", k,
                                           sample, layer, 0, 0.0))
                for rep in range(args.random_replicates):
                    random = random_keep(modality, k, 20260911 + rep * 1009 + layer)
                    rows.append(policy_metrics(data, modality, random, "random", k,
                                               sample, layer, rep, 0.0))
                for slack in args.importance_slacks:
                    tail = tail_aware_keep(data["ids"], data["weights"], modality, k, slack)
                    rows.append(policy_metrics(data, modality, tail, "tail_aware", k,
                                               sample, layer, 0, slack))
            for budget in args.budgets:
                for policy in ("random", "semantic", "tail"):
                    keep = budget_keep(data["ids"], data["weights"], modality,
                                       budget, policy, 20260911 + layer)
                    rows.append(policy_metrics(
                        data, modality, keep, f"budget_{policy}", -1, sample, layer,
                        int(round(budget * 100)), budget))
    write_csv(args.output / "policy_rows.csv", rows)
    summary = summarize(rows)
    write_csv(args.output / "policy_summary.csv", summary)
    (args.output / "manifest.json").write_text(json.dumps({
        "scope": "OFFLINE_REAL_QWEN3_VL_BRANCH_CAPTURE_ORACLE",
        "capture": str(args.capture.resolve()),
        "layers": args.layers, "ks": args.ks,
        "random_replicates": args.random_replicates,
        "importance_slacks": args.importance_slacks,
        "budgets": args.budgets,
        "rows": len(rows),
        "evidence_boundary": "quality/load oracle only; no runtime or E2E claim",
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
