"""Stratified same-expert pair atlas for weight, route, and spatial controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poc_flashvep.mllm_moe_transient_branch_compression.analyze_branches import (
    branch_groups, load_layer, modality_and_coords, nearest_pairs, pair_metrics,
    write_csv, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.result_dir / "manifest.json").read_text())
    rows = []
    for meta in manifest["samples"]:
        for layer in manifest["policy"]["layers"]:
            data = load_layer(args.result_dir, manifest, meta, int(layer))
            modality, coords = modality_and_coords(meta, data["position"])
            for wanted in ("vision", "text"):
                for expert, token, slot in branch_groups(data, modality, wanted):
                    hidden, output = data["hidden"][token], data["outputs"][token, slot]
                    pairs = {"hidden_nearest": nearest_pairs(hidden),
                             "output_nearest": nearest_pairs(output)}
                    order = np.arange(len(token))
                    pairs["arbitrary"] = (order[::2][:len(order)//2], order[1::2][:len(order)//2])
                    if wanted == "vision":
                        pos = data["position"][token]
                        lookup = {int(value): i for i, value in enumerate(pos)}
                        src = [i for i, value in enumerate(pos) if int(value + 1) in lookup]
                        pairs["contiguous_1d"] = (np.asarray(src), np.asarray([lookup[int(pos[i] + 1)] for i in src]))
                        src, dst = [], []
                        for i in range(len(token)):
                            distance = np.abs(coords[token[i + 1:]] - coords[token[i]]).sum(axis=1)
                            for offset in np.flatnonzero(distance == 1):
                                src.append(i); dst.append(i + 1 + int(offset))
                        pairs["neighbor_2d"] = (np.asarray(src), np.asarray(dst))
                    for kind, (source, target) in pairs.items():
                        if not len(source):
                            continue
                        hcos, hrel = pair_metrics(hidden[source], hidden[target])
                        ocos, orel = pair_metrics(output[source], output[target])
                        for i, (a, b) in enumerate(zip(source, target, strict=True)):
                            route_a, route_b = set(map(int, data["ids"][token[a]])), set(map(int, data["ids"][token[b]]))
                            spatial = int(np.abs(coords[token[a]] - coords[token[b]]).sum()) if wanted == "vision" else -1
                            rows.append({"sample_id": meta["sample_id"], "category": meta["category"],
                                "edge": meta["fixed_edge"], "layer": layer, "modality": wanted,
                                "expert": expert, "pair_kind": kind,
                                "source_position": int(data["position"][token[a]]),
                                "target_position": int(data["position"][token[b]]),
                                "source_weight": float(data["weights"][token[a], slot[a]]),
                                "target_weight": float(data["weights"][token[b], slot[b]]),
                                "route_jaccard": len(route_a & route_b) / len(route_a | route_b),
                                "spatial_manhattan": spatial, "input_cosine": float(hcos[i]),
                                "input_relative_l2": float(hrel[i]), "output_cosine": float(ocos[i]),
                                "output_relative_l2": float(orel[i]),
                                "contraction_ratio": float(orel[i] / max(float(hrel[i]), 1e-12))})
            print(json.dumps({"sample": meta["sample_id"], "layer": layer}), flush=True)
    write_csv(args.output_dir / "stratified_pairs.csv", rows)
    write_json(args.output_dir / "summary.json", {"rows": len(rows), "source": str(args.result_dir)})


if __name__ == "__main__":
    main()
