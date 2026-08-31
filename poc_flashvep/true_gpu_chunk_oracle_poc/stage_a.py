"""Construct preregistered same-M routing-shape pairs and analyze replay.

The pair selector is intentionally fixed before GPU timing: for each request,
layer and M, windows on an eight-token grid are considered, non-overlapping
pairs are required, and the pair with the largest composite routing-shape
distance is selected (lexicographic tie break). No measured latency enters
selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / "poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
LONG = ROOT / "poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000"
# Fixed bounded coverage: natural, fine-grained, chart/document, and
# multi-image examples, with the longer requests retained for M=512.
SHORT_IDS = ("coins", "cat", "coffee", "logo", "histology", "coffee_rocket", "model_card", "retina")
LAYERS = (0, 12, 24, 36, 47)
EXPERTS = 128
EP = 4


def _features(route: np.ndarray) -> dict[str, float | list[float]]:
    """Shape features over a token window; every token has top-8 IDs."""
    counts = np.bincount(route.reshape(-1), minlength=EXPERTS).astype(np.int64)
    total = int(counts.sum())
    nz = counts[counts > 0]
    prob = counts[counts > 0] / max(total, 1)
    ranks = np.bincount((route.reshape(-1) // 32), minlength=EP).astype(np.int64)
    active = int((counts > 0).sum())
    entropy = float(-(prob * np.log2(np.maximum(prob, 1e-15))).sum())
    hhi = float((prob**2).sum())
    median = float(np.median(nz)) if len(nz) else 0.0
    p10 = float(np.quantile(nz, .10)) if len(nz) else 0.0
    return {
        "assignments": total,
        "active_experts": active,
        "entropy": entropy,
        "hhi": hhi,
        "max_expert_load": int(counts.max()),
        "median_active_load": median,
        "p10_active_load": p10,
        "tiny_le_1": float((nz <= 1).mean()) if len(nz) else 0.0,
        "tiny_le_2": float((nz <= 2).mean()) if len(nz) else 0.0,
        "tiny_le_4": float((nz <= 4).mean()) if len(nz) else 0.0,
        "rank_counts": ranks.tolist(),
        "rank_imbalance": float(ranks.max() / max(ranks.mean(), 1e-12)),
        "rank_cv": float(ranks.std() / max(ranks.mean(), 1e-12)),
        "hist": (counts / max(total, 1)).tolist(),
    }


def _load() -> list[dict[str, Any]]:
    manifest = json.loads((SHORT / "workload_manifest.json").read_text())
    by_id = {p["vision"]["request_id"]: p["vision"] for p in manifest["pairs"]}
    rows: list[dict[str, Any]] = []
    for sid in SHORT_IDS:
        item = by_id[sid]
        with np.load(SHORT / item["route_file"]) as z:
            rows.append({"request_id": sid, "category": item["category"], "source": "short", "routes": z["routed_experts"].astype(np.int64), "token_ids": z["prompt_token_ids"].astype(np.int64)})
    long_manifest = json.loads((LONG / "sample_manifest.json").read_text())
    by_id = {x["sample_id"]: x for x in long_manifest["samples"]}
    for p in sorted(LONG.glob("routing.*.npz")):
        sid = p.name[len("routing."):-len(".npz")]
        with np.load(p) as z:
            rows.append({"request_id": sid, "category": by_id.get(sid, {}).get("category", "long"), "source": "long", "routes": z["routed_experts"].astype(np.int64), "token_ids": z["prompt_token_ids"].astype(np.int64)})
    return rows


def _pair_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    # Fixed, latency-blind composite: normalized histogram L1 plus shape scalars.
    hist_l1 = float(np.abs(np.asarray(a["hist"]) - np.asarray(b["hist"])).sum())
    active = abs(float(a["active_experts"]) - float(b["active_experts"])) / EXPERTS
    hhi = abs(float(a["hhi"]) - float(b["hhi"]))
    maxload = abs(float(a["max_expert_load"]) - float(b["max_expert_load"])) / max(float(a["assignments"]), 1.0)
    rank = abs(float(a["rank_cv"]) - float(b["rank_cv"]))
    return hist_l1 + active + hhi + maxload + rank


def select_pairs(ms: tuple[int, ...] = (128, 256, 512)) -> list[dict[str, Any]]:
    rows = _load()
    out: list[dict[str, Any]] = []
    for sample in rows:
        route = sample["routes"]
        for m in ms:
            layers = LAYERS if sample["source"] == "short" else (0, 24, 47)
            for layer in layers:
                candidates: list[dict[str, Any]] = []
                for start in range(0, max(0, len(route) - m + 1), 8):
                    end = start + m
                    f = _features(route[start:end, layer, :])
                    f["start"], f["end"] = start, end
                    candidates.append(f)
                # Shape extremes are sufficient for the preregistered pair and
                # keep candidate selection bounded on the long traces.  The
                # probe is selected without timing information.
                if len(candidates) > 32:
                    keys = [
                        lambda x: (float(x["entropy"]), -int(x["start"])),
                        lambda x: (-float(x["entropy"]), -int(x["start"])),
                        lambda x: (float(x["hhi"]), -int(x["start"])),
                        lambda x: (-float(x["hhi"]), -int(x["start"])),
                        lambda x: (float(x["max_expert_load"]), -int(x["start"])),
                        lambda x: (-float(x["max_expert_load"]), -int(x["start"])),
                    ]
                    probe_set: dict[int, dict[str, Any]] = {}
                    for key_fn in keys:
                        probe_set.update({int(x["start"]): x for x in sorted(candidates, key=key_fn)[:16]})
                    candidates = list(probe_set.values())
                best = None
                for i, a in enumerate(candidates):
                    for j in range(i + 1, len(candidates)):
                        b = candidates[j]
                        if int(a["end"]) > int(b["start"]):
                            continue
                        score = _pair_score(a, b)
                        key = (score, -int(a["start"]), -int(b["start"]))
                        if best is None or key > best[0]:
                            best = (key, a, b)
                if best is None:
                    continue
                _, a, b = best
                out.append({
                    "pair_id": len(out), "request_id": sample["request_id"], "category": sample["category"], "source": sample["source"], "layer": layer, "M": m,
                    "a": {k: v for k, v in a.items()}, "b": {k: v for k, v in b.items()},
                    "selection": "max composite routing-shape distance; grid=8; non-overlap; latency-blind",
                })
    # Fixed bounded GPU workload, selected without latency: retain evenly
    # spaced request/layer pairs per M from the deterministic enumeration.
    caps = {128: 16, 256: 16, 512: 8}
    bounded: list[dict[str, Any]] = []
    for m in ms:
        group = [x for x in out if int(x["M"]) == m]
        if len(group) > caps.get(m, len(group)):
            indices = np.linspace(0, len(group) - 1, caps[m], dtype=int)
            group = [group[int(i)] for i in indices]
        bounded.extend(group)
    for index, pair in enumerate(bounded):
        pair["pair_id"] = index
    return bounded


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", required=True); ap.add_argument("--make-candidates", action="store_true")
    args = ap.parse_args(); result = Path(args.result); result.mkdir(parents=True, exist_ok=True)
    pairs = select_pairs()
    (result / "candidates.json").write_text(json.dumps({"status": "ok", "pairs": pairs, "shape_features": ["active_experts", "entropy", "hhi", "max_expert_load", "p10_active_load", "median_active_load", "tiny_le_1", "tiny_le_2", "tiny_le_4", "rank_imbalance", "rank_cv"], "pair_policy": pairs[0]["selection"] if pairs else "none", "bounded_caps": {"128": 16, "256": 16, "512": 8}, "bounded_selection": "evenly spaced in deterministic request/layer enumeration per M"}, separators=(",", ":")) + "\n")
    rows = []
    for p in pairs:
        for label in ("a", "b"):
            r = {"pair_id": p["pair_id"], "request_id": p["request_id"], "category": p["category"], "source": p["source"], "layer": p["layer"], "M": p["M"], "candidate": label}
            r.update({k: v for k, v in p[label].items() if k not in ("hist", "rank_counts")})
            rows.append(r)
    pd.DataFrame(rows).to_csv(result / "candidate_features.csv", index=False)
    print(f"selected_pairs={len(pairs)} counts_by_M=" + json.dumps(pd.DataFrame(rows).groupby("M").size().to_dict() if rows else {}))


if __name__ == "__main__":
    main()
