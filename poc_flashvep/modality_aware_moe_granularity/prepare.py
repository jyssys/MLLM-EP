"""Prepare immutable real-route modality/M cases and offline route statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROUTE_ROOT = Path(
    "/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/"
    "live_prefill_execution_regime_20260821_111609"
)
REQUESTS = ("model_card", "deep_field", "retina")
LAYERS = (4, 24, 44)
MODALITIES = ("vision", "text")
M_VALUES = (32, 64, 128, 256, 512)
IMAGE_TOKEN_ID = 151655


def _entropy(hist: np.ndarray) -> float:
    p = hist[hist > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def _features(routes: np.ndarray) -> dict[str, float | int]:
    # routes: [M, 8], exact expert IDs from the validated real trace.
    flat = routes.reshape(-1)
    counts = np.bincount(flat, minlength=128).astype(np.int64)
    active = counts[counts > 0]
    rank = counts.reshape(4, 32).sum(axis=1)
    assignments = int(flat.size)
    mean_rank = float(rank.mean())
    token_ranks = np.array([len(np.unique(row // 32)) for row in routes])
    return {
        "tokens": int(len(routes)),
        "assignments": assignments,
        "active_experts": int(len(active)),
        "effective_experts": float(np.exp(_entropy(counts))),
        "expert_entropy": _entropy(counts),
        "expert_hhi": float(np.sum((counts.astype(np.float64) / assignments) ** 2)),
        "top4_assignment_fraction": float(np.sort(counts)[-4:].sum() / assignments),
        "top8_assignment_fraction": float(np.sort(counts)[-8:].sum() / assignments),
        "mean_active_expert_m": float(active.mean()),
        "p50_active_expert_m": float(np.quantile(active, .50)),
        "p95_active_expert_m": float(np.quantile(active, .95)),
        "max_expert_m": int(active.max()),
        "tiny_expert_le1_fraction": float(np.mean(active <= 1)),
        "tiny_expert_le2_fraction": float(np.mean(active <= 2)),
        "tiny_expert_le4_fraction": float(np.mean(active <= 4)),
        "rank_max_mean": float(rank.max() / max(mean_rank, 1e-12)),
        "rank_cv": float(rank.std() / max(mean_rank, 1e-12)),
        "rank_max": int(rank.max()),
        "rank_min": int(rank.min()),
        "unique_ranks_per_token": float(token_ranks.mean()),
        "p_u4": float(np.mean(token_ranks == 4)),
        "p_u_ge3": float(np.mean(token_ranks >= 3)),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out: Path, route_root: Path = ROUTE_ROOT) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "raw").mkdir()
    manifest = json.loads((route_root / "workload_manifest.json").read_text())
    by_id = {pair["vision"]["request_id"]: pair for pair in manifest["pairs"]}
    rows: list[dict] = []
    cases: list[dict] = []
    for request_id in REQUESTS:
        pair = by_id[request_id]
        for modality in MODALITIES:
            item = pair[modality]
            source = route_root / item["route_file"]
            with np.load(source) as archive:
                routes = np.asarray(archive["routed_experts"], dtype=np.int64)
                token_ids = np.asarray(archive["prompt_token_ids"], dtype=np.int64)
            if routes.shape[1:] != (48, 8):
                raise ValueError((request_id, modality, routes.shape))
            mask = token_ids == IMAGE_TOKEN_ID if modality == "vision" else token_ids != IMAGE_TOKEN_ID
            positions = np.flatnonzero(mask).astype(np.int64)
            for m in M_VALUES:
                if len(positions) < m:
                    continue
                chosen = positions[:m]
                for layer in LAYERS:
                    selected = routes[chosen, layer, :]
                    feature = _features(selected)
                    case_id = f"{request_id}_{modality}_l{layer}_m{m}"
                    case = {
                        "case_id": case_id,
                        "request_id": request_id,
                        "category": str(pair["vision"]["category"]),
                        "modality": modality,
                        "layer": layer,
                        "M": m,
                        "source_route": str(source),
                        "source_sha256": _sha(source),
                        "selected_positions": chosen.tolist(),
                        "routes": selected.tolist(),
                        "token_count": int(m),
                        "total_assignments": int(m * 8),
                    }
                    cases.append(case)
                    rows.append({**case, "routes": None, **feature})
    if not cases:
        raise RuntimeError("no cases prepared")
    (out / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    pd.DataFrame(rows).drop(columns=["routes", "selected_positions"]).to_csv(
        out / "route_statistics.csv", index=False
    )
    (out / "workload_manifest.json").write_text(json.dumps({
        "model": manifest["model"],
        "source": str(route_root),
        "source_manifest_sha256": _sha(route_root / "workload_manifest.json"),
        "requests": REQUESTS,
        "layers": LAYERS,
        "modalities": MODALITIES,
        "M_values": M_VALUES,
        "image_token_id": IMAGE_TOKEN_ID,
        "selection": "first M positions in original sequence order; no token reorder",
        "configuration": {
            "dtype": "BF16", "tp": 2, "dp": 2, "ep": 4, "pp": 1,
            "backend": "deepep_high_throughput", "triton_experts": True,
            "dbo": False, "prefix_cache": False, "enforce_eager": True,
            "expert_placement": "expert_id // 32", "physical_gpus": [1, 2, 3, 4],
        },
        "activation_provenance": "validated BF16 layer-24 Qwen3-VL capture used by replay; exact real route IDs retained",
        "case_count": len(cases),
    }, indent=2) + "\n")
    (out / "selection_policy.json").write_text(json.dumps({
        "pre_registered": True,
        "requests": REQUESTS, "layers": LAYERS, "M_values": M_VALUES,
        "common_fixed_granularity": 128,
        "warmups": 3, "iterations": 20,
        "random_hidden": True,
        "note": "operator-level characterization; route geometry is real, activations cycle validated capture rows",
    }, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--route-root", type=Path, default=ROUTE_ROOT)
    args = ap.parse_args()
    prepare(args.output, args.route_root)


if __name__ == "__main__":
    main()
