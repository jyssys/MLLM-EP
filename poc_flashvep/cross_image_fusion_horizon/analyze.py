"""Analyze causal fusion-delay accuracy, logits, and task horizons."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZONS = [0, 4, 8, 12, 16, 24, 32]


def _answer(text: str) -> str:
    match = re.search(r"\b(YES|NO|FIRST|SECOND)\b", text.upper())
    return match.group(1) if match else "UNPARSED"


def _logit_metrics(stock: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    stock = stock.astype(np.float64)
    changed = changed.astype(np.float64)
    cosine = float(np.dot(stock, changed) /
                   (np.linalg.norm(stock) * np.linalg.norm(changed) + 1e-12))
    stock_shifted = stock - stock.max()
    changed_shifted = changed - changed.max()
    p = np.exp(stock_shifted)
    p /= p.sum()
    q = np.exp(changed_shifted)
    q /= q.sum()
    kl = float(np.sum(p * (np.log(np.maximum(p, 1e-300)) -
                           np.log(np.maximum(q, 1e-300)))))
    return {"logit_cosine": cosine, "logit_kl": kl,
            "logit_max_abs": float(np.max(np.abs(stock - changed)))}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    result = args.result_dir
    records = []
    for rank in (0, 1):
        payload = json.loads((result / f"driver.dp{rank}.json").read_text())
        if not payload["ok"]:
            raise RuntimeError(payload["traceback"])
        records.extend(payload["records"])
    frame = pd.DataFrame(records)
    frame["parsed_answer"] = frame.output_text.map(_answer)
    frame["correct"] = frame.parsed_answer == frame.expected_answer
    frame["token_key"] = frame.output_token_ids.map(lambda value: ",".join(map(str, value)))

    logits = {}
    tp_replica_checks = []
    for row in frame.itertuples():
        matches = list((result / "raw_logits").glob(f"{row.capture_id}.dp{row.source_dp_rank}.tp*.npy"))
        if not matches:
            raise AssertionError((row.capture_id, matches))
        matches = sorted(matches)
        reference_path = next((path for path in matches if ".tp0.npy" in path.name), matches[0])
        reference = np.load(reference_path)
        max_replica_error = max(
            (float(np.max(np.abs(reference.astype(np.float32) -
                                 np.load(path).astype(np.float32))))
             for path in matches), default=0.0)
        tp_replica_checks.append({"capture_id": row.capture_id,
                                  "replicas": len(matches),
                                  "max_abs_error": max_replica_error})
        logits[row.capture_id] = reference
    pd.DataFrame(tp_replica_checks).to_csv(result / "tp_logit_replica_check.csv", index=False)
    stock_rows = frame[frame.condition == "stock"].set_index("pair_id")
    metric_rows = []
    for row in frame.itertuples():
        stock = stock_rows.loc[row.pair_id]
        metric_rows.append({
            "pair_id": row.pair_id, "task": row.task, "condition": row.condition,
            "intervention": row.intervention, "horizon": int(row.horizon),
            "expected_answer": row.expected_answer, "parsed_answer": row.parsed_answer,
            "baseline_correct": bool(stock.correct), "correct": bool(row.correct),
            "answer_consistent": row.parsed_answer == stock.parsed_answer,
            "tokens_exact": row.token_key == stock.token_key,
            **_logit_metrics(logits[stock.capture_id], logits[row.capture_id]),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(result / "per_pair_metrics.csv", index=False)
    primary = metrics[metrics.baseline_correct].copy()
    baseline_count = int(primary[primary.condition == "stock"].pair_id.nunique())
    baseline_by_task = (primary[primary.condition == "stock"].groupby("task").pair_id.nunique()
                        .to_dict())

    aggregate = primary.groupby(["intervention", "horizon"], as_index=False).agg(
        samples=("pair_id", "size"), accuracy=("correct", "mean"),
        answer_consistency=("answer_consistent", "mean"),
        exact_generation=("tokens_exact", "mean"),
        median_logit_cosine=("logit_cosine", "median"),
        median_logit_kl=("logit_kl", "median"),
        max_logit_error=("logit_max_abs", "max"))
    aggregate.to_csv(result / "horizon_summary.csv", index=False)
    task_aggregate = primary.groupby(["task", "intervention", "horizon"], as_index=False).agg(
        samples=("pair_id", "size"), accuracy=("correct", "mean"),
        answer_consistency=("answer_consistent", "mean"),
        median_logit_kl=("logit_kl", "median"))
    task_aggregate.to_csv(result / "task_horizon_summary.csv", index=False)

    h0 = metrics[metrics.condition.isin(["visual_h0", "full_h0"])]
    h0_check = {
        "generated_tokens_exact_fraction": float(h0.tokens_exact.mean()),
        "answers_exact_fraction": float(h0.answer_consistent.mean()),
        "max_logit_abs_error": float(h0.logit_max_abs.max()),
        "min_logit_cosine": float(h0.logit_cosine.min()),
    }
    h0_pass = (h0_check["generated_tokens_exact_fraction"] == 1.0 and
               h0_check["max_logit_abs_error"] == 0.0)

    def safe_horizon(intervention: str, task: str | None = None) -> int:
        local = aggregate[aggregate.intervention == intervention]
        if task is not None:
            local = task_aggregate[(task_aggregate.intervention == intervention) &
                                   (task_aggregate.task == task)]
        safe = 0
        for horizon in HORIZONS[1:]:
            row = local[local.horizon == horizon]
            if len(row) and float(row.accuracy.iloc[0]) >= .98:
                safe = horizon
            else:
                break
        return safe

    safe = {kind: safe_horizon(kind) for kind in ("visual", "full")}
    task_safe = {task: {kind: safe_horizon(kind, task) for kind in ("visual", "full")}
                 for task in ("identity", "brightness")}
    full_rows = aggregate[aggregate.intervention == "full"]
    later = full_rows[full_rows.horizon > safe["full"]]
    boundary = bool(len(later) and ((1.0 - later.accuracy) >= .10).any())
    coverage_ok = (baseline_count >= 16 and
                   all(int(baseline_by_task.get(task, 0)) >= 6
                       for task in ("identity", "brightness")))
    go = (h0_pass and coverage_ok and safe["full"] >= 12 and
          all(task_safe[task]["full"] >= 12 for task in task_safe) and boundary)
    hold = (h0_pass and (safe["visual"] >= 12 or safe["full"] in (4, 8) or
                         (safe["full"] >= 12 and not go)))
    status = "GO" if go else "HOLD" if hold else "NO-GO"

    pair_horizons = []
    for pair_id, pair_rows in primary.groupby("pair_id"):
        for intervention in ("visual", "full"):
            local = pair_rows[pair_rows.intervention == intervention]
            pair_safe = 0
            for horizon in HORIZONS[1:]:
                row = local[local.horizon == horizon]
                if len(row) and bool(row.correct.iloc[0]):
                    pair_safe = horizon
                else:
                    break
            pair_horizons.append({"pair_id": pair_id, "task": pair_rows.task.iloc[0],
                                  "intervention": intervention, "safe_horizon": pair_safe})
    pair_horizon_frame = pd.DataFrame(pair_horizons)
    pair_horizon_frame.to_csv(result / "per_pair_safe_horizon.csv", index=False)

    interaction_rows = []
    for path in sorted((result / "interaction").glob("*.json")):
        payload = json.loads(path.read_text())
        interaction_rows.append(payload)
    interaction = pd.DataFrame(interaction_rows)
    interaction_summary = interaction.groupby("layer", as_index=False).mean(numeric_only=True)
    interaction_summary.to_csv(result / "interaction_by_layer.csv", index=False)

    figures = result / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [("intra_image_visual_to_visual", "Intra-image V→V"),
              ("cross_image_visual_to_visual", "Cross-image V→V"),
              ("question_to_visual", "Question→visual")]
    for column, label in labels:
        ax.plot(interaction_summary.layer, interaction_summary[column], label=label)
    ax.set(xlabel="Decoder layer", ylabel="Mean attention mass",
           title="Bounded stock attention interaction characterization")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot1_cross_image_interaction_by_layer.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for intervention, label in (("visual", "Visual isolation"), ("full", "Full late fusion")):
        local = aggregate[aggregate.intervention == intervention]
        ax.plot(local.horizon, local.accuracy, marker="o", label=label)
    ax.axhline(.98, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Fusion horizon H", ylabel="Baseline-correct accuracy",
           ylim=(0, 1.04), title="Accuracy under causal fusion delay")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot2_accuracy_vs_fusion_horizon.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for intervention, label in (("visual", "Visual isolation"), ("full", "Full late fusion")):
        local = aggregate[aggregate.intervention == intervention]
        axes[0].plot(local.horizon, local.median_logit_kl, marker="o", label=label)
        axes[1].plot(local.horizon, local.median_logit_cosine, marker="o", label=label)
    axes[0].set(xlabel="H", ylabel="Median KL", title="First-token logit KL")
    axes[1].set(xlabel="H", ylabel="Median cosine", title="First-token logit cosine")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "plot3_logit_shift_vs_fusion_horizon.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = {"identity": 0, "brightness": 1}
    offsets = {"visual": -.12, "full": .12}
    colors = {"visual": "#4472c4", "full": "#ed7d31"}
    for (task, intervention), rows in pair_horizon_frame.groupby(["task", "intervention"]):
        jitter = np.linspace(-.045, .045, len(rows))
        ax.scatter(positions[task] + offsets[intervention] + jitter,
                   rows.safe_horizon, color=colors[intervention], alpha=.65, s=28)
    for intervention in ("visual", "full"):
        ax.scatter([], [], color=colors[intervention], label=intervention)
    ax.set_xticks([0, 1], ["Identity", "Brightness"])
    ax.set(ylabel="Per-pair contiguous safe horizon", title="Fusion horizon by task")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot4_horizon_by_task.png", dpi=200)
    plt.close(fig)

    summary = {
        "CROSS_IMAGE_FUSION_HORIZON": status,
        "baseline": {"total_pairs": int(frame[frame.condition == "stock"].pair_id.nunique()),
                     "correct_pairs": baseline_count,
                     "correct_by_task": {str(k): int(v) for k, v in baseline_by_task.items()}},
        "h0_correctness": h0_check,
        "maximum_safe_horizon": safe,
        "task_safe_horizon": task_safe,
        "degradation_boundary": boundary,
        "aggregate": aggregate.to_dict("records"),
        "interaction": interaction_summary.to_dict("records"),
        "coverage_ok": coverage_ok,
    }
    _json(result / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
