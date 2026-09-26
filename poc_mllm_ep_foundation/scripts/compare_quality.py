from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poc_mllm_ep_foundation.io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-score", type=Path, required=True)
    parser.add_argument("--ep-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    reference = json.loads(args.reference_score.read_text())
    ep = json.loads(args.ep_score.read_text())
    left = {row["request_id"]: row for row in reference["per_sample"]}
    right = {row["request_id"]: row for row in ep["per_sample"]}
    ids = sorted(set(left) & set(right))
    if not ids:
        raise RuntimeError("no paired samples")
    groups: dict[str, list[str]] = {}
    for request_id in ids:
        groups.setdefault(str(left[request_id]["image_group"]), []).append(request_id)
    names = sorted(groups)
    group_delta = np.asarray([
        np.mean([float(right[i]["correct"]) - float(left[i]["correct"])
                 for i in groups[name]]) for name in names
    ])
    rng = np.random.default_rng(args.seed)
    sampled = group_delta[rng.integers(0, len(group_delta), size=(args.bootstrap, len(group_delta)))].mean(1)
    delta = float(group_delta.mean())
    output = {
        "schema_version": "mllm_ep_paired_quality.v1",
        "task": reference["task"],
        "paired_n": len(ids), "image_clusters": len(names),
        "reference_accuracy": float(np.mean([left[i]["correct"] for i in ids])),
        "ep_accuracy": float(np.mean([right[i]["correct"] for i in ids])),
        "ep_minus_reference_pp": delta * 100,
        "cluster_bootstrap_95ci_pp": [float(np.quantile(sampled, 0.025) * 100),
                                      float(np.quantile(sampled, 0.975) * 100)],
        "noninferiority_margin_pp": -1.0,
        "noninferiority_proven": bool(np.quantile(sampled, 0.025) * 100 > -1.0),
        "exact_output_agreement": float(np.mean([
            left[i]["prediction"] == right[i]["prediction"] for i in ids
        ])),
        "baseline_correct_to_wrong": sum(left[i]["correct"] and not right[i]["correct"] for i in ids),
        "baseline_wrong_to_correct": sum(not left[i]["correct"] and right[i]["correct"] for i in ids),
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
