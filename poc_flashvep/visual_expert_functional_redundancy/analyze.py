"""Analyze raw Top-8 expert outputs and apply the preregistered modality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

M_VALUES = (1, 2, 3, 4, 6, 8)
SEED = 20260825


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _token_metrics(outputs: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    outputs = outputs.astype(np.float64)
    weights = weights.astype(np.float64)
    weights /= weights.sum()
    norms = np.linalg.norm(outputs, axis=1)
    normalized = outputs / np.maximum(norms[:, None], 1e-12)
    gram_cos = normalized @ normalized.T
    pairwise = float(gram_cos[np.triu_indices(8, 1)].mean())
    center = outputs.mean(axis=0)
    dispersion = float(np.sqrt(np.mean(np.sum((outputs - center) ** 2, axis=1))) /
                       (norms.mean() + 1e-12))
    eigenvalues = np.maximum(np.linalg.eigvalsh(outputs @ outputs.T), 0)
    participation = float(eigenvalues.sum() ** 2 /
                          (np.square(eigenvalues).sum() + 1e-12))
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    router_effective_k = float(np.exp(entropy))
    cumulative = np.cumsum(weights)
    router_mass_k = int(np.searchsorted(cumulative, .95) + 1)
    y8 = np.sum(weights[:, None] * outputs, axis=0)
    functional_k = 8
    values: dict[str, float] = {}
    for m in M_VALUES:
        local_weights = weights[:m] / weights[:m].sum()
        ym = np.sum(local_weights[:, None] * outputs[:m], axis=0)
        cosine = _cosine(ym, y8)
        relative_l2 = float(np.linalg.norm(ym - y8) / (np.linalg.norm(y8) + 1e-12))
        values[f"cosine_m{m}"] = cosine
        values[f"relative_l2_m{m}"] = relative_l2
        if functional_k == 8 and cosine >= .99 and relative_l2 <= .05:
            functional_k = m
    return {"pairwise_cosine": pairwise, "normalized_dispersion": dispersion,
            "participation_rank": participation, "router_entropy": entropy,
            "router_effective_k": router_effective_k, "router_mass95_k": router_mass_k,
            "functional_k": functional_k, **values}


def _cluster_ci(frame: pd.DataFrame, column: str, seed: int = SEED) -> tuple[float, float, float]:
    paired = frame.pivot_table(index="sample_id", columns="modality", values=column,
                               aggfunc="mean").dropna()
    differences = (paired["text"] - paired["visual"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(differences, len(differences), replace=True).mean())
            for _ in range(2000)]
    return float(differences.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def _region(layer: int) -> str:
    if layer <= 12: return "early"
    if layer <= 28: return "middle"
    return "late"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args(); result = args.result_dir
    manifest = json.loads((result / "manifest.json").read_text())
    raw = result / "raw"
    records = []
    correctness = []
    for meta in manifest["samples"]:
        sample = meta["sample_id"]
        source_dp = next(row["source_dp_rank"] for row in manifest["schedule"]
                         if row["sample_id"] == sample)
        prompt_tokens = int(meta["prompt_tokens"])
        visual_start, visual_end = map(int, meta["visual_span"])
        for layer in manifest["policy"]["layers"]:
            router_parts = [np.load(raw / f"router.{sample}.dp{source_dp}.tp{tp}.layer{layer}.npz")
                            for tp in (0, 1)]
            router = {key: np.concatenate([part[key] for part in router_parts], axis=0)[:prompt_tokens]
                      for key in ("fingerprints", "selected", "token_ids", "topk_ids", "topk_weights")}
            expert_rows: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]]] = {}
            for ep_rank in range(4):
                arrays = np.load(raw / f"experts.{sample}.ep{ep_rank}.layer{layer}.npz")
                for index, fingerprint in enumerate(arrays["fingerprints"]):
                    expert_rows.setdefault(str(fingerprint), []).append((
                        arrays["expert_ids"][index], arrays["router_weights"][index],
                        arrays["local_mask"][index], arrays["raw_outputs"][index],
                        arrays["stock_output"][index]))
            for position in np.flatnonzero(router["selected"]):
                modality = ("visual" if visual_start <= position < visual_end
                            else "text" if position >= visual_end else "excluded_pre_image")
                if modality == "excluded_pre_image":
                    continue
                fingerprint = str(router["fingerprints"][position])
                pieces = expert_rows.get(fingerprint, [])
                outputs = np.zeros((8, 2048), dtype=np.float32)
                coverage = np.zeros(8, dtype=int)
                stock = np.zeros(2048, dtype=np.float32)
                ids = router["topk_ids"][position].astype(int)
                weights = router["topk_weights"][position].astype(np.float32)
                for expert_ids, expert_weights, local_mask, raw_outputs, stock_output in pieces:
                    # DeepEP masks non-local IDs with a rank-specific sentinel;
                    # only locally owned slots retain their original IDs/weights.
                    if not np.array_equal(expert_ids[local_mask].astype(int), ids[local_mask]):
                        raise AssertionError(f"{sample} L{layer}: local route mismatch")
                    if not np.allclose(expert_weights[local_mask], weights[local_mask], atol=1e-6):
                        raise AssertionError(f"{sample} L{layer}: local weight mismatch")
                    stock += stock_output.astype(np.float32)
                    for slot in np.flatnonzero(local_mask):
                        outputs[slot] += raw_outputs[slot].astype(np.float32)
                        coverage[slot] += 1
                if not np.all(coverage == 1):
                    raise AssertionError(f"{sample} L{layer}: incomplete slot coverage {coverage}")
                reconstructed = np.sum((weights / weights.sum())[:, None] * outputs, axis=0)
                correctness.append({"sample_id": sample, "layer": layer,
                                    "modality": modality,
                                    "stock_cosine": _cosine(reconstructed, stock),
                                    "stock_relative_l2": float(np.linalg.norm(reconstructed - stock) /
                                                               (np.linalg.norm(stock) + 1e-12))})
                records.append({"sample_id": sample, "category": meta["category"],
                                "layer": layer, "region": _region(layer),
                                "position": int(position), "modality": modality,
                                **_token_metrics(outputs, weights)})
    frame = pd.DataFrame(records)
    check = pd.DataFrame(correctness)
    frame.to_csv(result / "token_metrics.csv", index=False)
    check.to_csv(result / "raw_output_correctness.csv", index=False)
    aggregate = frame.groupby(["layer", "region", "modality"], as_index=False).agg(
        tokens=("functional_k", "size"), functional_k=("functional_k", "mean"),
        pairwise_cosine=("pairwise_cosine", "mean"),
        normalized_dispersion=("normalized_dispersion", "mean"),
        participation_rank=("participation_rank", "mean"),
        router_effective_k=("router_effective_k", "mean"),
        router_mass95_k=("router_mass95_k", "mean"))
    aggregate.to_csv(result / "layer_summary.csv", index=False)
    region_summary = frame.groupby(["region", "modality"], as_index=False).agg(
        tokens=("functional_k", "size"), functional_k=("functional_k", "mean"),
        pairwise_cosine=("pairwise_cosine", "mean"),
        normalized_dispersion=("normalized_dispersion", "mean"),
        participation_rank=("participation_rank", "mean"),
        router_effective_k=("router_effective_k", "mean"),
        router_mass95_k=("router_mass95_k", "mean"))
    region_summary.to_csv(result / "region_summary.csv", index=False)

    late = frame[frame.region == "late"]
    late_gap, late_ci_low, late_ci_high = _cluster_ci(late, "functional_k")
    late_means = late.groupby("modality").functional_k.mean()
    late_relative_gap = float(late_gap / late_means["text"])
    diversity = {
        metric: _cluster_ci(late, metric)
        for metric in ("pairwise_cosine", "normalized_dispersion", "participation_rank",
                       "router_effective_k", "router_mass95_k")}
    # Interpretable router-adjusted mediation: fit only router effective-K and
    # preregistered layer fixed effects, then compare image-clustered residuals.
    model_frame = frame.copy()
    layer_dummies = pd.get_dummies(model_frame.layer.astype(str), drop_first=True, dtype=float)
    x = np.column_stack([model_frame.router_effective_k.to_numpy(), layer_dummies.to_numpy()])
    regression = LinearRegression().fit(x, model_frame.functional_k)
    model_frame["router_adjusted_residual"] = model_frame.functional_k - regression.predict(x)
    adjusted_gap, adjusted_low, adjusted_high = _cluster_ci(
        model_frame[model_frame.region == "late"], "router_adjusted_residual")
    retention = float(adjusted_gap / late_gap) if abs(late_gap) > 1e-12 else 0.0
    directions = {
        "visual_higher_pairwise_cosine": diversity["pairwise_cosine"][0] < 0,
        "visual_lower_dispersion": diversity["normalized_dispersion"][0] > 0,
        "visual_lower_participation_rank": diversity["participation_rank"][0] > 0,
    }
    strongest_layer = None
    layer_gaps = []
    for layer in manifest["policy"]["layers"]:
        local = frame[frame.layer == layer]
        gap, low, high = _cluster_ci(local, "functional_k", SEED + layer)
        means = local.groupby("modality").functional_k.mean()
        relative = float(gap / means["text"])
        row = {"layer": layer, "text_minus_visual_k": gap, "ci_low": low,
               "ci_high": high, "relative_gap": relative}
        layer_gaps.append(row)
        if strongest_layer is None or relative > strongest_layer["relative_gap"]:
            strongest_layer = row
    pd.DataFrame(layer_gaps).to_csv(result / "layer_modality_gaps.csv", index=False)
    if (late_relative_gap >= .30 and late_ci_low > 0 and all(directions.values())
            and retention >= .50):
        status = "GO"
    elif ((late_relative_gap >= .10 and late_ci_low > 0 and sum(directions.values()) >= 2)
          or (strongest_layer is not None and strongest_layer["relative_gap"] >= .30
              and strongest_layer["ci_low"] > 0)):
        status = "HOLD"
    else:
        status = "NO-GO"

    figures = result / "figures"; figures.mkdir(exist_ok=True)
    colors = {"visual": "#4472c4", "text": "#ed7d31"}
    metrics = [("pairwise_cosine", "Pairwise cosine"),
               ("normalized_dispersion", "Normalized dispersion"),
               ("participation_rank", "Participation rank"),
               ("router_effective_k", "Router effective-K")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (metric, label) in zip(axes.flat, metrics, strict=True):
        for modality in ("visual", "text"):
            local = aggregate[aggregate.modality == modality]
            ax.plot(local.layer, local[metric], marker="o", color=colors[modality], label=modality)
        ax.set(ylabel=label); ax.grid(alpha=.25)
    axes[0, 0].legend(); axes[1, 0].set_xlabel("MoE layer"); axes[1, 1].set_xlabel("MoE layer")
    fig.suptitle("Raw Top-8 expert-output diversity by modality")
    fig.tight_layout(); fig.savefig(figures / "plot1_expert_output_diversity_by_layer.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for modality in ("visual", "text"):
        local = aggregate[aggregate.modality == modality]
        ax.plot(local.layer, local.functional_k, marker="o", linewidth=2,
                color=colors[modality], label=modality)
    ax.set(xlabel="MoE layer", ylabel="Mean functional effective-K",
           title="Top-m approximation of the full Top-8 expert output")
    ax.set_ylim(0, 8.3); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "plot2_functional_k_by_modality.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for modality in ("visual", "text"):
        local = frame[frame.modality == modality]
        axes[0].scatter(local.router_effective_k, local.functional_k, s=8, alpha=.2,
                        color=colors[modality], label=modality)
    axes[0].set(xlabel="Router entropy effective-K", ylabel="Functional effective-K")
    axes[0].legend(); axes[0].grid(alpha=.25)
    xlabels = ["early", "middle", "late"]
    positions = np.arange(3); width = .36
    for offset, modality in ((-.18, "visual"), (.18, "text")):
        local = region_summary.set_index(["region", "modality"])
        axes[1].bar(positions + offset,
                    [local.loc[(region, modality), "functional_k"] for region in xlabels],
                    width, color=colors[modality], label=modality)
    axes[1].set_xticks(positions, xlabels); axes[1].set_ylabel("Functional effective-K")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Router-only K versus raw-output functional K")
    fig.tight_layout(); fig.savefig(figures / "plot3_router_k_vs_functional_k.png", dpi=200); plt.close(fig)

    summary = {
        "VISUAL_FUNCTIONAL_COLLAPSE": status,
        "POC4": "PENDING" if status in ("GO", "HOLD") else "NOT-RUN",
        "tokens": {key: int(value) for key, value in frame.modality.value_counts().items()},
        "samples": int(frame.sample_id.nunique()),
        "raw_output_correctness": {
            "min_cosine": float(check.stock_cosine.min()),
            "median_cosine": float(check.stock_cosine.median()),
            "max_relative_l2": float(check.stock_relative_l2.max()),
            "median_relative_l2": float(check.stock_relative_l2.median()),
        },
        "region_summary": region_summary.to_dict("records"),
        "late_functional_k": {"visual": float(late_means["visual"]),
                              "text": float(late_means["text"]),
                              "text_minus_visual": late_gap,
                              "relative_gap": late_relative_gap,
                              "clustered_ci": [late_ci_low, late_ci_high]},
        "late_diversity_differences_text_minus_visual": {
            key: {"mean": value[0], "ci": [value[1], value[2]]}
            for key, value in diversity.items()},
        "router_adjusted_late_gap": {"mean": adjusted_gap,
                                     "ci": [adjusted_low, adjusted_high],
                                     "raw_gap_retention": retention},
        "diversity_directions": directions,
        "strongest_layer_gap": strongest_layer,
    }
    _json(result / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
