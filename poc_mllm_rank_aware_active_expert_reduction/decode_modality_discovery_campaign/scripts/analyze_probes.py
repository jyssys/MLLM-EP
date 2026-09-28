from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import SEED, atomic_json, read_json, write_jsonl_gz


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    return shifted - np.log(np.exp(shifted).sum())


def damage(base: np.ndarray, changed: np.ndarray, target: int) -> dict[str, Any]:
    blp, clp = log_softmax(base), log_softmax(changed)
    bp, cp = np.exp(blp), np.exp(clp)
    middle = np.maximum(0.5 * (bp + cp), 1e-300)
    jsd = 0.5 * float(np.sum(bp * (blp - np.log(middle))))
    jsd += 0.5 * float(np.sum(cp * (clp - np.log(middle))))
    drop = float(blp[target] - clp[target])
    flip = int(np.argmax(base) != np.argmax(changed))
    return {"target_logprob_drop_nats": drop, "jsd": jsd, "top1_flip": flip,
            "damage": max(0.0, drop) + 2.0 * jsd + float(flip)}


def load_logits(run: Path, probe: str, step: int) -> tuple[np.ndarray, int]:
    with np.load(run / "logits" / f"{probe}.step{step}.npz") as payload:
        return payload["logits"], int(payload["target_token_id"])


def bootstrap(values: dict[str, float], repeats: int = 10000) -> dict[str, Any]:
    requests = sorted(values)
    point = float(np.mean([values[key] for key in requests]))
    rng, sampled = np.random.default_rng(SEED), []
    for _ in range(repeats):
        selected = rng.choice(requests, len(requests), replace=True)
        sampled.append(float(np.mean([values[str(value)] for value in selected])))
    return {"mean": point, "ci95": [float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))],
            "requests": len(requests), "bootstrap_unit": "request", "repeats": repeats}


def marker_audit(run: Path, plan: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case in plan["cases"]:
        for variant in case["variants"]:
            if not variant["intervene"]:
                continue
            markers = [json.loads(Path(value).read_text()) for value in glob.glob(str(run / "markers" / f"{variant['probe_id']}.pid*.json"))]
            removed = sum(int(value["owned_removed_count"]) for value in markers)
            expected = 8 - int(variant["keep_k"])
            passed = bool(markers) and removed == expected and all(
                value["fingerprint"] == case["target_hidden_hash"] for value in markers
            )
            rows.append({"probe_id": variant["probe_id"], "markers": len(markers),
                         "removed": removed, "expected": expected, "pass": passed})
    failed = [row for row in rows if not row["pass"]]
    if failed:
        raise RuntimeError(f"marker audit failed: {failed[:5]}")
    return {"all_pass": True, "interventions": len(rows),
            "marker_count_distribution": dict(Counter(row["markers"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan); labels = []
    for case in plan["cases"]:
        baseline = next(value for value in case["variants"] if value["policy"] == "baseline")
        for variant in case["variants"]:
            if not variant["intervene"]:
                continue
            immediate, horizon_values = None, []
            for offset in range(int(case["horizon"])):
                step = int(case["probe_step"]) + offset
                base, target = load_logits(args.run, baseline["probe_id"], step)
                changed, changed_target = load_logits(args.run, variant["probe_id"], step)
                if target != changed_target:
                    raise RuntimeError((variant["probe_id"], step))
                metric = damage(base, changed, target)
                horizon_values.append(metric["damage"])
                if offset == 0: immediate = metric
            labels.append({
                "case_id": case["case_id"], "request_id": case["request_id"], "domain": case["domain"],
                "split": case["split"], "target_layer": case["target_layer"], "probe_step": case["probe_step"],
                "probe_id": variant["probe_id"], "policy": variant["policy"], "keep_k": variant["keep_k"],
                "removed_slots": variant["removed_slots"], "removed_expert_ids": variant["removed_expert_ids"],
                "predicted_wave_closure_events": variant.get("predicted_wave_closure_events"),
                "predicted_bottleneck_rank_removals": variant.get("predicted_bottleneck_rank_removals"),
                **immediate, "horizon_mean_damage": float(np.mean(horizon_values)),
                "horizon_damage": horizon_values,
                "evidence": "MEASURED_EXACT_FUNCTIONAL_INTERVENTION_NOT_PHYSICAL_WORK_REDUCTION",
            })
    write_jsonl_gz(args.labels, labels)
    policies = sorted({row["policy"] for row in labels})
    by = {(row["request_id"], row["policy"], row["keep_k"]): row for row in labels}
    comparisons = {}
    for policy in policies:
        if policy in ("router_tail", "random"):
            continue
        for keep in (6, 4):
            requests = sorted({row["request_id"] for row in labels if row["policy"] == policy and row["keep_k"] == keep})
            vs_tail = {request: by[(request, "router_tail", keep)]["horizon_mean_damage"] - by[(request, policy, keep)]["horizon_mean_damage"] for request in requests}
            vs_random = {request: by[(request, "random", keep)]["horizon_mean_damage"] - by[(request, policy, keep)]["horizon_mean_damage"] for request in requests}
            comparisons[f"{policy}_k{keep}"] = {
                "positive_means_candidate_lower_damage": True,
                "vs_router_tail": bootstrap(vs_tail), "vs_random": bootstrap(vs_random),
                "candidate_damage": float(np.mean([by[(request, policy, keep)]["horizon_mean_damage"] for request in requests])),
                "tail_damage": float(np.mean([by[(request, "router_tail", keep)]["horizon_mean_damage"] for request in requests])),
                "random_damage": float(np.mean([by[(request, "random", keep)]["horizon_mean_damage"] for request in requests])),
                "closure_events": float(np.mean([by[(request, policy, keep)]["predicted_wave_closure_events"] for request in requests])),
            }
    summary = {"schema_version": 1, "split": plan["split"], "requests": len(plan["cases"]),
               "labels": len(labels), "policies": policies, "marker_audit": marker_audit(args.run, plan),
               "comparisons": comparisons}
    atomic_json(args.summary, summary)
    print(json.dumps({"split": plan["split"], "comparisons": comparisons}, indent=2))


if __name__ == "__main__": main()
