#!/usr/bin/env python3
"""Analyze fixed-policy EP runtime-tail captures and render the preregistered plots."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize(values: np.ndarray) -> dict[str, float]:
    median = float(np.median(values))
    mean = float(np.mean(values))
    return {
        "median_ms": median,
        "p95_ms": percentile(values, 95),
        "cv_pct": float(np.std(values, ddof=1) / mean * 100),
        "tail_frequency_pct": float(np.mean(values >= 1.15 * median) * 100),
        "max_ms": float(np.max(values)),
    }


def overlap_ms(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def read_context(root: Path, context: str) -> pd.DataFrame:
    records = []
    for path in sorted((root / context / "raw").glob("rank*.jsonl")):
        with path.open() as handle:
            records.extend(json.loads(line) for line in handle)

    rows = []
    for record in records:
        block_m = int(record["runtime_config"]["BLOCK_SIZE_M"])
        histogram = tuple(record["histogram"])
        row = {
            "context": context,
            "iteration": int(record["iteration"]),
            "layer": int(record["layer"]),
            "rank": int(record["rank"]),
            "n": int(record["n"]),
            "g": int(record["g"]),
            "q": int(sum(math.ceil(x / block_m) for x in histogram if x)),
            "histogram": ",".join(map(str, histogram)),
            "block_m": block_m,
            "dispatch_ms": float(record["dispatch"]["comm_ms"]),
            "expert_ms": float(record["expert"]["compute_ms"]),
            "combine_ms": float(record["combine"]["comm_ms"]),
            "dispatch_expert_overlap_ms": overlap_ms(
                record["dispatch"]["comm_start_ms"],
                record["dispatch"]["comm_end_ms"],
                record["expert"]["compute_start_ms"],
                record["expert"]["compute_end_ms"],
            ),
            "expert_combine_overlap_ms": overlap_ms(
                record["expert"]["compute_start_ms"],
                record["expert"]["compute_end_ms"],
                record["combine"]["comm_start_ms"],
                record["combine"]["comm_end_ms"],
            ),
        }
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["iteration", "rank", "layer"])
    frame["previous_expert_ms"] = frame.groupby(["context", "iteration", "rank"])[
        "expert_ms"
    ].shift(1).fillna(0.0)
    frame["previous_combine_ms"] = frame.groupby(["context", "iteration", "rank"])[
        "combine_ms"
    ].shift(1).fillna(0.0)
    group_cols = ["context", "rank", "layer", "histogram"]
    frame["static_median_ms"] = frame.groupby(group_cols)["expert_ms"].transform("median")
    frame["tail"] = frame["expert_ms"] >= 1.15 * frame["static_median_ms"]
    return frame


def grouped_critical_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    model_features = {
        "N": ["n"],
        "N+G": ["n", "g"],
        "N+G+Q": ["n", "g", "q"],
        "N+G+Q+runtime": [
            "n", "g", "q", "dispatch_ms",
            "previous_expert_ms", "previous_combine_ms",
            "dispatch_expert_overlap_ms",
        ],
    }
    context_frames = []
    for _, data in frame.groupby("context"):
        data = data.copy()
        data["group"] = data["iteration"].astype(str)
        for model_name, features in model_features.items():
            data[f"pred_{model_name}"] = np.nan
            splitter = GroupKFold(n_splits=5)
            for train, test in splitter.split(data, groups=data["group"]):
                model = LinearRegression().fit(
                    data.iloc[train][features], data.iloc[train]["expert_ms"]
                )
                data.loc[data.index[test], f"pred_{model_name}"] = model.predict(
                    data.iloc[test][features]
                )
        context_frames.append(data)
    data = pd.concat(context_frames).sort_index()

    outcomes = []
    for (context, iteration, layer), rows in data.groupby(["context", "iteration", "layer"]):
        actual = int(rows.loc[rows["expert_ms"].idxmax(), "rank"])
        out = {"context": context, "iteration": iteration, "layer": layer, "actual_rank": actual}
        for model_name in model_features:
            predicted = int(rows.loc[rows[f"pred_{model_name}"].idxmax(), "rank"])
            out[model_name] = predicted
            out[f"match_{model_name}"] = predicted == actual
        outcomes.append(out)
    result = pd.DataFrame(outcomes)
    accuracy = {
        name: float(result[f"match_{name}"].mean() * 100) for name in model_features
    }
    accuracy["runtime_gain_pp"] = accuracy["N+G+Q+runtime"] - accuracy["N+G+Q"]
    accuracy["by_context"] = {
        context: {
            name: float(part[f"match_{name}"].mean() * 100) for name in model_features
        }
        for context, part in result.groupby("context")
    }
    return result, accuracy


def critical_stability(frame: pd.DataFrame) -> float:
    critical = []
    for (context, iteration, layer), rows in frame.groupby(["context", "iteration", "layer"]):
        critical.append((context, layer, int(rows.loc[rows["expert_ms"].idxmax(), "rank"])))
    values = pd.DataFrame(critical, columns=["context", "layer", "rank"])
    shares = []
    for _, rows in values.groupby(["context", "layer"]):
        shares.append(max(Counter(rows["rank"]).values()) / len(rows))
    return float(np.mean(shares) * 100)


def plot_outputs(root: Path, frame: pd.DataFrame, isolated: np.ndarray,
                 critical: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    selected = frame[(frame["layer"] == 45) & (frame["rank"] == 0)]
    plot1 = pd.DataFrame({"context": "isolated", "expert_ms": isolated})
    plot1 = pd.concat([plot1, selected[["context", "expert_ms"]]], ignore_index=True)
    plt.figure(figsize=(8, 4.8))
    sns.boxplot(data=plot1, x="context", y="expert_ms", order=["isolated", "controlled", "serving"])
    sns.stripplot(data=plot1, x="context", y="expert_ms", order=["isolated", "controlled", "serving"],
                  color="black", alpha=.25, size=2)
    plt.ylabel("Expert CUDA latency (ms)")
    plt.xlabel("Execution context")
    plt.tight_layout()
    plt.savefig(figures / "plot1_latency_distribution_by_context.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    for context, rows in selected.groupby("context"):
        med = rows.expert_ms.median()
        plt.plot(rows.iteration, rows.expert_ms / med, marker="o", ms=3, label=context)
    plt.axhline(1.15, color="crimson", linestyle="--", label="fixed +15% tail threshold")
    plt.xlabel("Measured iteration")
    plt.ylabel("Latency / context median")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "plot2_same_work_tail_events.png", dpi=180)
    plt.close()

    cols = ["expert_ms", "dispatch_ms", "combine_ms", "previous_expert_ms"]
    normalized = frame.copy()
    for col in cols:
        normalized[col] = normalized[col] / normalized.groupby(
            ["context", "rank", "layer", "histogram"]
        )[col].transform("median").replace(0, np.nan)
    melted = normalized.melt(id_vars=["context", "tail"], value_vars=cols,
                             var_name="runtime_factor", value_name="relative_value")
    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=melted, x="runtime_factor", y="relative_value", hue="tail", errorbar=("ci", 95))
    plt.ylabel("Value / same-work median")
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figures / "plot3_tail_vs_runtime_context.png", dpi=180)
    plt.close()

    names = ["N", "N+G", "N+G+Q", "N+G+Q+runtime"]
    acc = [critical[f"match_{name}"].mean() * 100 for name in names]
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(names, acc, color=["#777777", "#5b8ff9", "#61d9a6", "#f6bd16"])
    plt.bar_label(bars, fmt="%.1f%%")
    plt.ylim(0, max(acc) + 12)
    plt.ylabel("Actual critical-rank accuracy")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(figures / "plot4_critical_rank_prediction.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    frames = [read_context(root, context) for context in ("controlled", "serving")]
    frame = pd.concat(frames, ignore_index=True)
    isolated_record = json.loads((root / "isolated" / "isolated.json").read_text())
    isolated = np.asarray(isolated_record["samples_ms"], dtype=float)

    context_summary = {"isolated": summarize(isolated)}
    for context, rows in frame.groupby("context"):
        context_summary[context] = summarize(rows.expert_ms.to_numpy())
        context_summary[context]["exact_static_tail_frequency_pct"] = float(
            rows["tail"].mean() * 100
        )
        group_cv = rows.groupby(["rank", "layer", "histogram"])["expert_ms"].agg(
            lambda x: np.std(x, ddof=1) / np.mean(x) * 100
        )
        context_summary[context]["exact_static_cv_median_pct"] = float(group_cv.median())
        context_summary[context]["exact_static_cv_p95_pct"] = float(group_cv.quantile(.95))
        context_summary[context]["critical_rank_stability_pct"] = critical_stability(rows)
        context_summary[context]["dispatch_median_ms"] = float(rows.dispatch_ms.median())
        context_summary[context]["combine_median_ms"] = float(rows.combine_ms.median())

    selected_summary = {"isolated": summarize(isolated)}
    selected = frame[(frame["layer"] == 45) & (frame["rank"] == 0)]
    for context, rows in selected.groupby("context"):
        selected_summary[context] = summarize(rows.expert_ms.to_numpy())
        selected_summary[context]["n"] = int(rows.n.iloc[0])
        selected_summary[context]["g"] = int(rows.g.iloc[0])
        selected_summary[context]["q"] = int(rows.q.iloc[0])
        selected_summary[context]["histogram_variants"] = int(rows.histogram.nunique())

    critical, accuracy = grouped_critical_predictions(frame)
    normal_tail = {}
    for context, rows in frame.groupby("context"):
        normal_tail[context] = {}
        for factor in ["dispatch_ms", "combine_ms", "previous_expert_ms", "previous_combine_ms"]:
            med = rows.groupby(["rank", "layer", "histogram"])[factor].transform("median")
            relative = rows[factor] / med.replace(0, np.nan)
            normal_tail[context][factor] = {
                "normal_mean_relative": float(relative[~rows["tail"]].mean()),
                "tail_mean_relative": float(relative[rows["tail"]].mean()),
                "difference": float(
                    relative[rows["tail"]].mean() - relative[~rows["tail"]].mean()
                ),
            }

    summary = {
        "fixed_policy": {
            "target_request": "text_18_tui_main",
            "layer": 45,
            "rank": 0,
            "tail_definition": "expert latency >= 1.15 * exact-static-work context median",
            "selection_source": "prior live-prefill trace, fixed before rerun",
        },
        "context_all_layers": context_summary,
        "selected_exact_work": selected_summary,
        "critical_rank_accuracy": accuracy,
        "tail_context_factors": normal_tail,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    frame.to_csv(root / "per_iteration_stage_metrics.csv", index=False)
    critical.to_csv(root / "critical_rank_predictions.csv", index=False)
    plot_outputs(root, frame, isolated, critical)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
