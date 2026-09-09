"""Contribution-weighted sharing oracle and expert-skipping control.

This deliberately uses captured outputs to select the cheapest branches. It is
an upper bound, not a deployable policy. Its purpose is to determine whether a
positive result is truly branch sharing or merely low-contribution skipping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    cosine_rows, load_layer, modality_and_coords, write_csv, write_json,
)

FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35)


def share_candidates(data: dict[str, np.ndarray], modality: np.ndarray,
                     coords: np.ndarray, spatial: bool) -> list[tuple[float, int, int, int]]:
    """Return (local-error proxy, token, slot, representative-token)."""
    exact = np.einsum("tk,tkh->th", data["weights"], data["outputs"])
    denom = np.maximum(np.linalg.norm(exact, axis=1), 1e-12)
    candidates = []
    visual = np.flatnonzero(modality == "vision")
    for token in visual:
        for slot, expert in enumerate(data["ids"][token]):
            peer = visual[np.any(data["ids"][visual] == expert, axis=1)]
            peer = peer[peer != token]
            if spatial and len(peer):
                distance = np.abs(coords[peer] - coords[token]).sum(axis=1)
                peer = peer[distance == 1]
            if not len(peer):
                continue
            delta = data["outputs"][peer, np.argmax(data["ids"][peer] == expert, axis=1)] \
                - data["outputs"][token, slot]
            error = data["weights"][token, slot] * np.linalg.norm(delta, axis=1) / denom[token]
            best = int(np.argmin(error))
            candidates.append((float(error[best]), int(token), int(slot), int(peer[best])))
    return sorted(candidates)


def skip_candidates(data: dict[str, np.ndarray], modality: np.ndarray) -> list[tuple[float, int, int, int]]:
    exact = np.einsum("tk,tkh->th", data["weights"], data["outputs"])
    denom = np.maximum(np.linalg.norm(exact, axis=1), 1e-12)
    candidates = []
    for token in np.flatnonzero(modality == "vision"):
        for slot in range(data["ids"].shape[1]):
            contribution = data["weights"][token, slot] * data["outputs"][token, slot]
            candidates.append((float(np.linalg.norm(contribution) / denom[token]),
                               int(token), int(slot), -1))
    return sorted(candidates)


def evaluate(data: dict[str, np.ndarray], modality: np.ndarray,
             candidates: list[tuple[float, int, int, int]], fraction: float,
             kind: str) -> dict[str, Any]:
    target_mask = np.broadcast_to((modality == "vision")[:, None], data["ids"].shape)
    target = max(1, round(int(target_mask.sum()) * fraction))
    approx = data["outputs"].copy()
    selected: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    for _, token, slot, representative in candidates:
        key = (token, slot)
        if key in protected or key in selected:
            continue
        if representative >= 0:
            expert = data["ids"][token, slot]
            rep_slot = int(np.argmax(data["ids"][representative] == expert))
            rep_key = (representative, rep_slot)
            if rep_key in selected:
                continue
            approx[token, slot] = data["outputs"][representative, rep_slot]
            protected.add(rep_key)
        else:
            approx[token, slot] = 0
        selected.add(key)
        if len(selected) >= target:
            break
    exact = np.einsum("tk,tkh->th", data["weights"], data["outputs"])
    predicted = np.einsum("tk,tkh->th", data["weights"], approx)
    visual = modality == "vision"
    affected = np.zeros(len(modality), dtype=bool)
    for token, _ in selected:
        affected[token] = True
    relative = np.linalg.norm(predicted - exact, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1), 1e-12)
    cosine = cosine_rows(predicted, exact)
    return {
        "kind": kind, "target_fraction": fraction,
        "achieved_fraction": len(selected) / max(int(target_mask.sum()), 1),
        "combined_rel_l2_median": float(np.median(relative[visual])),
        "combined_rel_l2_p90": float(np.quantile(relative[visual], .9)),
        "combined_rel_l2_p99": float(np.quantile(relative[visual], .99)),
        "combined_cosine_median": float(np.median(cosine[visual])),
        "combined_cosine_p10": float(np.quantile(cosine[visual], .1)),
        "tokens_pass_1pct": float(((relative <= .01) & (cosine >= .9999))[visual].mean()),
        "tokens_pass_5pct": float(((relative <= .05) & (cosine >= .999))[visual].mean()),
        "affected_token_fraction": float(affected[visual].mean()),
        "affected_rel_l2_median": float(np.median(relative[affected])),
        "affected_rel_l2_p90": float(np.quantile(relative[affected], .9)),
        "affected_cosine_median": float(np.median(cosine[affected])),
        "affected_pass_1pct": float(((relative <= .01) & (cosine >= .9999))[affected].mean()),
        "affected_pass_5pct": float(((relative <= .05) & (cosine >= .999))[affected].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.result_dir / "manifest.json").read_text())
    layers = [int(value) for value in manifest["policy"]["layers"]]
    rows = []
    for meta in manifest["samples"]:
        for layer in layers:
            data = load_layer(args.result_dir, manifest, meta, layer)
            modality, coords = modality_and_coords(meta, data["position"])
            candidate_sets = {
                "output_oracle_share": share_candidates(data, modality, coords, False),
                "spatial_output_oracle_share": share_candidates(data, modality, coords, True),
                "contribution_skip": skip_candidates(data, modality),
            }
            for kind, candidates in candidate_sets.items():
                for fraction in FRACTIONS:
                    rows.append({"sample_id": meta["sample_id"], "category": meta["category"],
                                 "edge": meta.get("fixed_edge", 448), "layer": layer,
                                 **evaluate(data, modality, candidates, fraction, kind)})
            print(json.dumps({"sample": meta["sample_id"], "layer": layer}), flush=True)
    write_csv(args.output_dir / "contribution_oracle.csv", rows)
    write_json(args.output_dir / "summary.json", {"source": str(args.result_dir),
               "rows": len(rows), "fractions": FRACTIONS,
               "warning": "captured-output oracle; not a deployable policy"})


if __name__ == "__main__":
    main()
