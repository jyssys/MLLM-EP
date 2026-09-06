"""Bounded diagnostic surrogates for missing expert contributions.

This is intentionally not a method implementation.  It asks whether a
same-expert empirical mean or a spatially adjacent visual token can stand in
for omitted expert output vectors in the captured trace.  Hidden-state and
weight-space affinity are unavailable in this compact capture, so results are
labelled output-affinity diagnostics rather than claims about SpecMoE.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["modality", "scope", "surrogate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--result-dir", type=Path, required=True)
    args = ap.parse_args(); result = args.result_dir
    manifest = json.loads((result / "manifest.json").read_text())
    source_dp = {}
    for rank, names in enumerate(manifest.get("partition", [])):
        for name in names: source_dp[name] = rank
    layers = manifest.get("policy", {}).get("layers", [])
    raw = result / "raw"; records = []
    for meta in manifest["samples"]:
        sample = meta["sample_id"]; dp = source_dp.get(sample, 0); start, end = meta["visual_span"]
        for layer in layers:
            parts = []
            for tp in (0, 1):
                p = raw / f"router.{sample}.dp{dp}.tp{tp}.layer{layer}.npz"
                if p.exists(): parts.append(np.load(p))
            if not parts: continue
            router = {k: np.concatenate([part[k] for part in parts], axis=0) for k in ("fingerprints", "selected", "topk_ids", "topk_weights")}
            by_fp = defaultdict(list)
            for ep in range(4):
                p = raw / f"experts.{sample}.ep{ep}.layer{layer}.npz"
                if not p.exists(): continue
                a = np.load(p)
                for i, fp in enumerate(a["fingerprints"]): by_fp[str(fp)].append((a["expert_ids"][i], a["local_mask"][i], a["raw_outputs"][i]))
            for pos in np.flatnonzero(router["selected"]):
                pieces = by_fp.get(str(router["fingerprints"][pos]), [])
                if not pieces: continue
                ids = router["topk_ids"][pos].astype(int); weights = router["topk_weights"][pos].astype(np.float64)
                out = np.zeros((8, pieces[0][2].shape[-1]), dtype=np.float32); cov = np.zeros(8, dtype=int)
                for expert_ids, mask, raw_out in pieces:
                    for slot in np.flatnonzero(mask): out[slot] += raw_out[slot].astype(np.float32); cov[slot] += 1
                if not np.all(cov == 1): continue
                records.append({"sample": sample, "layer": int(layer), "position": int(pos),
                                "modality": "visual" if start <= pos < end else "text", "ids": ids, "weights": weights, "out": out})
    # Candidate pools are restricted to the same sample/layer/expert to avoid
    # silently conflating layer weights or request-specific activation scales.
    pools = defaultdict(list)
    for rec in records:
        for slot, expert in enumerate(rec["ids"]): pools[(rec["sample"], rec["layer"], int(expert))].append((rec["position"], rec["out"][slot], rec["modality"]))
    rng = np.random.default_rng(20260906)
    rows = []
    for rec in records:
        w = rec["weights"]; w = w / max(float(w.sum()), 1e-12); exact = np.sum(w[:, None] * rec["out"], axis=0)
        order = np.argsort(-w, kind="stable")
        for m in (4, 6, 7):
            omitted = order[m:]
            candidates = {"same_expert_mean": [], "spatial_neighbor": [], "random_same_expert": []}
            for slot in omitted:
                pool = pools[(rec["sample"], rec["layer"], int(rec["ids"][slot]))]
                if pool:
                    mean = np.mean([x[1] for x in pool], axis=0)
                    candidates["same_expert_mean"].append(mean)
                    # Exclude the current token itself; otherwise the
                    # purported neighbour would trivially be the exact answer.
                    visual = [x for x in pool if x[0] != rec["position"] and x[2] == "visual" and rec["modality"] == "visual"]
                    if visual:
                        candidates["spatial_neighbor"].append(min(visual, key=lambda x: abs(x[0] - rec["position"]))[1])
                    else: candidates["spatial_neighbor"].append(mean)
                    candidates["random_same_expert"].append(pool[int(rng.integers(len(pool)))][1])
                else:
                    zero = np.zeros(rec["out"].shape[-1], dtype=np.float32)
                    for name in candidates: candidates[name].append(zero)
            for name, values in candidates.items():
                pred = rec["out"].copy()
                for slot, value in zip(omitted, values): pred[slot] = value
                y = np.sum(w[:, None] * pred, axis=0)
                rows.append({"sample": rec["sample"], "layer": rec["layer"], "position": rec["position"], "modality": rec["modality"], "scope": f"top{m}", "surrogate": name, "cosine": cosine(y, exact), "relative_l2": float(np.linalg.norm(y - exact) / (np.linalg.norm(exact) + 1e-12))})
    write_csv(result / "surrogate_results.csv", rows)
    summary = []
    grouped = defaultdict(list)
    for row in rows: grouped[(row["modality"], row["scope"], row["surrogate"])].append(row)
    for (modality, scope, surrogate), values in sorted(grouped.items()):
        summary.append({"modality": modality, "scope": scope, "surrogate": surrogate, "tokens": len(values), "cosine_mean": float(np.mean([x["cosine"] for x in values])), "relative_l2_mean": float(np.mean([x["relative_l2"] for x in values])), "relative_l2_p90": float(np.quantile([x["relative_l2"] for x in values], .9))})
    write_csv(result / "surrogate_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
