from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import SUBSPACE_RANK, atomic_json, read_json, write_jsonl_gz


def row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(values.shape)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def right_basis(rows: np.ndarray, rank: int = SUBSPACE_RANK) -> np.ndarray:
    normalized = row_normalize(rows)
    _, _, vh = np.linalg.svd(normalized, full_matrices=False)
    return vh[: min(rank, vh.shape[0])]


def projection_energy(row: np.ndarray, basis: np.ndarray) -> float:
    normalized = row_normalize(np.asarray(row).reshape(1, -1))[0]
    return float(np.square(basis @ normalized).sum())


def files(raw: Path, prefix: str) -> list[Path]:
    return sorted(Path(value) for value in glob.glob(str(raw / f"{prefix}*.npz")))


def load_capture(raw: Path, capture_id: str, layer: int, step: int) -> dict[str, Any]:
    prefill = files(raw, f"prefill.{capture_id}.")
    decode = files(raw, f"decode.{capture_id}.")
    if not prefill or not decode:
        raise RuntimeError(f"incomplete capture {capture_id}: prefill={len(prefill)}, decode={len(decode)}")
    vision, text, visual_count, text_count = [], [], 0, 0
    for path in prefill:
        with np.load(path) as payload:
            vision.append(payload["visual_rows"].astype(np.float32))
            text.append(payload["text_rows"].astype(np.float32))
            visual_count += int(payload["visual_count"])
            text_count += int(payload["text_count"])
    # The decode target is stored only on TP rank zero, but guard against duplicate calls.
    if len(decode) != 1:
        raise RuntimeError(f"expected one decode record for {capture_id}, got {decode}")
    with np.load(decode[0]) as payload:
        hidden = payload["hidden"].astype(np.float32).reshape(-1)
        fingerprint = str(payload["hidden_fingerprint"].item())
        ids = payload["topk_ids"].astype(np.int64).reshape(-1)
        weights = payload["topk_weights"].astype(np.float64).reshape(-1)
    order = np.argsort(-weights)
    ids, weights = ids[order], weights[order]
    if len(ids) != 8 or not np.isclose(weights.sum(), 1.0, atol=2e-3):
        raise RuntimeError((capture_id, ids.tolist(), weights.tolist(), weights.sum()))
    vrows, trows = np.concatenate(vision), np.concatenate(text)
    vb, tb = right_basis(vrows), right_basis(trows)
    fb = right_basis(np.concatenate((vrows, trows)))
    ev, et, ef = (projection_energy(hidden, basis) for basis in (vb, tb, fb))
    eps = 1e-6
    return {
        "capture_id": capture_id,
        "layer": layer,
        "step": step,
        "target_hidden_hash": fingerprint,
        "topk_ids": ids.tolist(),
        "topk_weights": weights.tolist(),
        "vision_energy": ev,
        "text_energy": et,
        "full_energy": ef,
        "separated_visual_score": float(np.log((ev + eps) / (et + eps))),
        "removed_mass_k6": float(weights[6:].sum()),
        "removed_mass_k4": float(weights[4:].sum()),
        "router_removed_mass_control": float(0.5 * (weights[6:].sum() + weights[4:].sum())),
        "sampled_visual_rows": int(vrows.shape[0]),
        "sampled_text_rows": int(trows.shape[0]),
        "logical_visual_rows_observed_across_tp": visual_count,
        "logical_text_rows_observed_across_tp": text_count,
        "subspace_rank": SUBSPACE_RANK,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    raw = args.capture_run / "raw"
    signal_rows, probe_cases = [], []
    for case in manifest["cases"]:
        for condition in ("orig", "cf"):
            capture_id = f"{case['case_id']}.{condition}"
            signal = load_capture(raw, capture_id, case["target_layer"], case["probe_step"])
            signal.update({
                "case_id": case["case_id"], "condition": condition,
                "split": case["split"], "request_id": case["request_id"],
                "domain": case["domain"],
            })
            signal_rows.append(signal)
            probe_cases.append({
                **case,
                "condition": condition,
                "probe_case_id": capture_id,
                "image_path": case["original_image_path"] if condition == "orig" else case["counterfactual_image_path"],
                "target_hidden_hash": signal["target_hidden_hash"],
                "topk_ids": signal["topk_ids"],
                "topk_weights": signal["topk_weights"],
                "keep6_expert_ids": signal["topk_ids"][:6],
                "keep6_mass": float(sum(signal["topk_weights"][:6])),
                "keep4_expert_ids": signal["topk_ids"][:4],
                "keep4_mass": float(sum(signal["topk_weights"][:4])),
            })
    write_jsonl_gz(args.signals, signal_rows)
    atomic_json(args.plan, {
        "schema_version": 1,
        "frozen_before_structured_damage": True,
        "selection_uses_damage": False,
        "policy": "NATIVE_ROUTER_TOP_WEIGHT_K6_K4_NO_REPLACEMENT_RENORMALIZE",
        "semantic_diagnostic_only": True,
        "case_count": len(probe_cases),
        "cases": probe_cases,
    })
    print({"signals": len(signal_rows), "probe_cases": len(probe_cases)})


if __name__ == "__main__":
    main()
