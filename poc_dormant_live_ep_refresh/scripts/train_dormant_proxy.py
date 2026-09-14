#!/usr/bin/env python3
"""Train deliberately simple future-free dormant classifiers on layer 16."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_accept(path: Path):
    document = json.loads(path.read_text())
    return {
        tuple(map(int, key.split(":"))): int(value)
        for key, value in document["acceptance_iteration_by_key"].items()
    }


def denoise_by_wave(path: Path):
    prefix = "[LLADA_DENOISE]"
    result = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            where = line.find(prefix)
            if where >= 0:
                record = json.loads(line[where + len(prefix):])
                if not str(record.get("request_id", "")).startswith("warmup"):
                    result[int(record["iteration"])] = record
    return result


def build_rows(campaign: Path, task: str):
    accept = load_accept(campaign / "analysis" / f"{task}_acceptance_plan.json")
    denoise = denoise_by_wave(
        campaign / "logs" / f"shape_{task}_ep4_b32_mini32_r1_g32.log"
    )
    rows = []
    previous = {}
    root = campaign / "raw" / "shape" / task / "r1"
    for path in sorted(root.glob("shape_rank*.jsonl")):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if int(record.get("wave", -1)) < 0 or int(record["layer"]) != 16:
                    continue
                wave = int(record["wave"])
                semantic = denoise[wave]
                global_rows = []
                for seq, block, iteration, masks, conf, margin in zip(
                    semantic["sequence_ids"], semantic["block_starts"],
                    semantic["block_iterations"], semantic["mask_before"],
                    semantic["token_confidence"], semantic["confidence_margin"]
                ):
                    masked_rank = -1
                    for position, masked in enumerate(masks):
                        if masked:
                            masked_rank += 1
                        global_rows.append((
                            (int(seq), int(block), position), int(iteration), bool(masked),
                            masked_rank, float(conf[position]), float(margin[position])
                        ))
                start = int(record["global_row_start"])
                ids_rows = record["topk_ids_local_source"]
                local_rows = global_rows[start:start + len(ids_rows)]
                rank = int(record["rank"])
                for index, (logical, t, masked, masked_rank, conf, margin) in enumerate(local_rows):
                    if not masked:
                        continue
                    ids = set(int(value) for value in ids_rows[index])
                    owners = {value // 64 for value in ids}
                    prior = previous.get((rank, logical))
                    rows.append({
                        "task": task, "sequence_id": logical[0], "block_start": logical[1],
                        "position": logical[2], "iteration": t,
                        "phase_index": min(t / 31.0, 1.0),
                        "confidence": conf, "margin": margin,
                        "masked_rank": masked_rank,
                        "router_entropy": float(record["router_entropy_local_source"][index]),
                        "top1_probability": float(record["top1_probability_local_source"][index]),
                        "topk_mass": float(record["topk_probability_mass_local_source"][index]),
                        "topk_gap": float(record["topk_probability_gap_local_source"][index]),
                        "fanout": len(owners),
                        "route_overlap": 0.0 if prior is None else len(ids & prior[0]) / max(len(ids | prior[0]), 1),
                        "destination_overlap": 0.0 if prior is None else len(owners & prior[1]) / max(len(owners | prior[1]), 1),
                        "dormant": int(accept[logical] - t > 1),
                    })
                    previous[(rank, logical)] = (ids, owners)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    feature_groups = {
        "semantic": ["confidence", "margin", "phase_index", "masked_rank", "position"],
        "expert": ["router_entropy", "top1_probability", "topk_mass", "topk_gap", "fanout", "route_overlap", "destination_overlap"],
        "semantic+expert": ["confidence", "margin", "phase_index", "masked_rank", "position", "router_entropy", "top1_probability", "topk_mass", "topk_gap", "fanout", "route_overlap", "destination_overlap"],
    }
    oracle = pd.read_csv(args.task_root / "DORMANT_ORACLES.csv")
    output = []
    datasets = {task: build_rows(args.campaign, task) for task in ("gsm8k", "humaneval")}
    for task, frame in datasets.items():
        # Sequence IDs are stable within this single submitted batch.  Keep
        # the last quarter held out to prevent row-level leakage.
        split = int(frame.sequence_id.max()) * 3 // 4
        train = frame[frame.sequence_id <= split]
        test = frame[frame.sequence_id > split]
        perfect = float(oracle[
            (oracle.task == task) & (oracle.horizon == 1)
            & (oracle.policy == "perfect_remove")
        ].iloc[0].feasible_e2e_pct)
        for name, features in feature_groups.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=500, class_weight="balanced", random_state=0),
            )
            model.fit(train[features], train.dormant)
            train_prob = model.predict_proba(train[features])[:, 1]
            # Select the lowest train threshold that maintains >=99% dormant
            # precision.  If none exists, make no dormant predictions.
            threshold = 1.01
            for candidate in sorted(np.unique(train_prob)):
                pred = train_prob >= candidate
                if pred.sum() and train.loc[pred, "dormant"].mean() >= 0.99:
                    threshold = float(candidate)
                    break
            probability = model.predict_proba(test[features])[:, 1]
            predicted = probability >= threshold
            truth = test.dormant.to_numpy().astype(bool)
            tp = int((predicted & truth).sum())
            fp = int((predicted & ~truth).sum())
            fn = int((~predicted & truth).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            output.append({
                "task": task, "split": f"sequence_id>{split}", "model": "logistic",
                "features": name, "horizon": 1,
                "false_dormant_rate": fp / max(tp + fp, 1),
                "precision": precision, "recall": recall, "f1": f1,
                "deferred_routed_cost_pct": perfect * recall,
                "oracle_recovery_pct": 100 * recall,
                "post_epoch_e2e_pct": perfect * recall,
                "gate": "KILL" if perfect * recall < 5 else "CHARACTERIZATION",
                "threshold": threshold, "test_rows": len(test),
            })
    fields = list(output[0])
    with (args.task_root / "CLASSIFIER_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(pd.DataFrame(output).to_string(index=False))


if __name__ == "__main__":
    main()
