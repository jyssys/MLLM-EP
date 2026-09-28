#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from poc_mllm_vision_conditioned_decode_damage_skipping.contract import (
    atomic_json, read_json, read_jsonl, write_jsonl_gz,
)


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    return shifted - np.log(np.exp(shifted).sum())


def metrics(base: np.ndarray, changed: np.ndarray, target: int) -> dict[str, Any]:
    base_lp, changed_lp = log_softmax(base), log_softmax(changed)
    base_p, changed_p = np.exp(base_lp), np.exp(changed_lp)
    middle = np.maximum(0.5 * (base_p + changed_p), 1e-300)
    kl = float(np.sum(base_p * (base_lp - changed_lp)))
    jsd = 0.5 * float(np.sum(base_p * (base_lp - np.log(middle))))
    jsd += 0.5 * float(np.sum(changed_p * (changed_lp - np.log(middle))))
    top1_base, top1_changed = int(np.argmax(base)), int(np.argmax(changed))
    overlap_k = min(10, int(base.size), int(changed.size))
    overlap = len(
        set(np.argpartition(base, -overlap_k)[-overlap_k:])
        & set(np.argpartition(changed, -overlap_k)[-overlap_k:])
    ) / overlap_k
    target_drop = float(base_lp[target] - changed_lp[target])
    flip = top1_base != top1_changed
    score = max(0.0, target_drop) + 2.0 * jsd + float(flip)
    return {
        "next_token_kl": kl,
        "next_token_jsd": jsd,
        "target_logprob_drop_nats": target_drop,
        "top1_flip": flip,
        "top10_logit_overlap": overlap,
        "baseline_top1_token": top1_base,
        "modified_top1_token": top1_changed,
        "functional_damage_score": score,
        "absolute_safe": target_drop <= 0.01 and jsd <= 0.0001 and not flip,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--local-labels", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    local = {
        (row["request_id"], row["layer"], row["decode_step"], row["expert_id"]): row
        for row in read_jsonl(args.local_labels)
    }
    labels = []
    marker_audit = []
    def locate(relative: Path) -> Path:
        matches = [run / relative for run in args.run if (run / relative).is_file()]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one {relative}, found {matches}")
        return matches[0]

    for case in plan["cases"]:
        baseline_path = locate(Path("logits") / f"{case['case_id']}.baseline.npz")
        with np.load(baseline_path) as payload:
            baseline = payload["logits"]
            target = int(payload["target_token_id"])
        for path in case["paths"]:
            probe = f"{case['case_id']}.drop_slot{path['slot_rank']}"
            with np.load(locate(Path("logits") / f"{probe}.npz")) as payload:
                changed = payload["logits"]
                if int(payload["target_token_id"]) != target:
                    raise RuntimeError(f"target mismatch for {probe}")
            key = (case["request_id"], case["target_layer"], case["probe_step"], path["expert_id"])
            row = dict(local[key])
            row.update({
                "case_id": case["case_id"],
                "probe_id": probe,
                **metrics(baseline, changed, target),
                "functional_evidence": "MEASURED_TRUE_EP4_DROP_ONE_REMAINING_FORWARD",
            })
            labels.append(row)
            marker_paths = []
            for run in args.run:
                marker_paths.extend(glob.glob(str(run / "markers" / f"{probe}.pid*.json")))
            markers = [json.loads(Path(value).read_text()) for value in marker_paths]
            marker_audit.append({
                "probe_id": probe,
                "marker_count": len(markers),
                "owned_target_count": sum(bool(value["owned_target"]) for value in markers),
                "all_fingerprint_match": all(value["fingerprint"] == case["target_hidden_hash"] for value in markers),
            })
    if len(labels) != plan["case_count"] * 8:
        raise RuntimeError((len(labels), plan["case_count"] * 8))
    # A route is materialized only on the relevant source/receiver ranks, so a
    # matching target row is not expected on every one of the four EP ranks.
    # Exactness requires at least one matching materialization, exactly one
    # physical owner applying the drop, and no fingerprint mismatch.
    if any(row["marker_count"] < 1 or row["owned_target_count"] != 1 or not row["all_fingerprint_match"] for row in marker_audit):
        raise RuntimeError("intervention marker audit failed")
    write_jsonl_gz(args.labels, labels)
    values = np.asarray([row["functional_damage_score"] for row in labels])
    summary = {
        "status": "MEASURED_FUNCTIONAL_DAMAGE",
        "path_count": len(labels),
        "request_count": len({row["request_id"] for row in labels}),
        "calibration_paths": sum(row["split"] == "calibration" for row in labels),
        "heldout_paths": sum(row["split"] == "heldout" for row in labels),
        "primary_damage_metric": "max(0,target_logprob_drop_nats)+2*JSD+1[top1_flip]",
        "functional_damage_score": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p80": float(np.percentile(values, 80)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
        },
        "absolute_safe_fraction": float(np.mean([row["absolute_safe"] for row in labels])),
        "top1_flip_fraction": float(np.mean([row["top1_flip"] for row in labels])),
        "local_mixture_vs_functional_spearman": float(spearmanr(
            [row["local_mixture_relative_l2"] for row in labels],
            [row["functional_damage_score"] for row in labels],
        ).statistic),
        "marker_audit": {
            "probe_count": len(marker_audit),
            "all_pass": True,
            "marker_count_distribution": dict(Counter(row["marker_count"] for row in marker_audit)),
            "marker_contract": "one_or_more_matching_materializations; exactly_one_physical_expert_owner",
            "expected_owned_target_per_probe": 1,
        },
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
