"""Analyze matched within-expert token redundancy and representative oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 20260825
RATIOS = (0.25, 0.50, 0.75, 1.00)
MIN_GROUP = 8
CAP_GROUP = 32


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _distance(x: np.ndarray) -> np.ndarray:
    normalized = _normalize(x.astype(np.float64))
    return np.maximum(1.0 - normalized @ normalized.T, 0.0)


def _greedy_medoids(distance: np.ndarray, count: int) -> np.ndarray:
    """Deterministic output/hidden-space medoid selection.

    The output-space use is a bounded oracle proxy: it sees E_e(x), unlike the
    practical path, while retaining real token outputs as representatives.
    """
    size = len(distance)
    if count >= size:
        return np.arange(size)
    medoids = [int(np.argmin(distance.sum(axis=1)))]
    minimum = distance[:, medoids[0]].copy()
    while len(medoids) < count:
        best_candidate = -1
        best_cost = math.inf
        for candidate in range(size):
            if candidate in medoids:
                continue
            cost = float(np.minimum(minimum, distance[:, candidate]).sum())
            if cost < best_cost:
                best_candidate, best_cost = candidate, cost
        medoids.append(best_candidate)
        minimum = np.minimum(minimum, distance[:, best_candidate])
    return np.asarray(medoids, dtype=int)


def _stable_entries(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    def key(entry: dict[str, Any]) -> str:
        value = f"{entry['sample_id']}:{entry['position']}:{entry['slot']}:{entry['expert']}"
        return hashlib.sha1(value.encode()).hexdigest()
    return sorted(entries, key=key)[:count]


def _diversity(vectors: np.ndarray) -> dict[str, float]:
    vectors = vectors.astype(np.float64)
    normalized = _normalize(vectors)
    cosine = normalized @ normalized.T
    pairwise = float(cosine[np.triu_indices(len(vectors), 1)].mean())
    norms = np.linalg.norm(vectors, axis=1)
    centered = vectors - vectors.mean(axis=0)
    dispersion = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))) /
                       (norms.mean() + 1e-12))
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered @ centered.T), 0)
    participation = float(eigenvalues.sum() ** 2 /
                          (np.square(eigenvalues).sum() + 1e-12))
    relative = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)
    scale = (norms[:, None] + norms[None, :]) / 2
    relative /= np.maximum(scale, 1e-12)
    np.fill_diagonal(relative, np.inf)
    nearest = float(np.min(relative, axis=1).mean())
    return {"pairwise_cosine": pairwise, "normalized_dispersion": dispersion,
            "centered_participation_rank": participation,
            "nearest_neighbor_relative_l2": nearest}


def _bootstrap(values: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean()
                        for _ in range(2000)])
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def _region(layer: int) -> str:
    if layer <= 12:
        return "early"
    if layer <= 24:
        return "middle"
    return "late"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _load_assignments(result: Path, manifest: dict[str, Any]) -> tuple[
        dict[tuple[int, int, str], list[dict[str, Any]]], pd.DataFrame]:
    raw = result / "raw"
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    checks: list[dict[str, Any]] = []
    source_rank = {row["sample_id"]: row["source_dp_rank"] for row in manifest["schedule"]}
    for meta in manifest["samples"]:
        sample = meta["sample_id"]
        prompt_tokens = int(meta["prompt_tokens"])
        visual_start, visual_end = map(int, meta["visual_span"])
        for layer in manifest["policy"]["layers"]:
            router_parts = [np.load(raw / f"router.{sample}.dp{source_rank[sample]}.tp{tp}.layer{layer}.npz")
                            for tp in (0, 1)]
            router = {key: np.concatenate([part[key] for part in router_parts], axis=0)[:prompt_tokens]
                      for key in ("fingerprints", "selected", "topk_ids", "topk_weights")}
            expert_rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
            for ep_rank in range(4):
                arrays = np.load(raw / f"experts.{sample}.ep{ep_rank}.layer{layer}.npz")
                if "hidden_states" not in arrays:
                    raise AssertionError("hidden-state capture missing")
                for index, fingerprint in enumerate(arrays["fingerprints"]):
                    expert_rows[str(fingerprint)].append((
                        arrays["expert_ids"][index], arrays["router_weights"][index],
                        arrays["local_mask"][index], arrays["raw_outputs"][index],
                        arrays["stock_output"][index], arrays["hidden_states"][index]))
            for position in np.flatnonzero(router["selected"]):
                if visual_start <= position < visual_end:
                    modality = "visual"
                elif position >= visual_end:
                    modality = "text"
                else:
                    continue
                fingerprint = str(router["fingerprints"][position])
                ids = router["topk_ids"][position].astype(int)
                weights = router["topk_weights"][position].astype(np.float32)
                coverage = np.zeros(8, dtype=int)
                outputs = np.zeros((8, 2048), dtype=np.float32)
                stock = np.zeros(2048, dtype=np.float32)
                for expert_ids, expert_weights, local_mask, raw_outputs, stock_output, hidden in expert_rows[fingerprint]:
                    stock += stock_output.astype(np.float32)
                    for slot in np.flatnonzero(local_mask):
                        if int(expert_ids[slot]) != int(ids[slot]):
                            raise AssertionError(f"{sample} L{layer}: route mismatch")
                        if not np.isclose(expert_weights[slot], weights[slot], atol=1e-6):
                            raise AssertionError(f"{sample} L{layer}: weight mismatch")
                        coverage[slot] += 1
                        outputs[slot] = raw_outputs[slot].astype(np.float32)
                        groups[(layer, int(ids[slot]), modality)].append({
                            "sample_id": sample, "category": meta["category"],
                            "position": int(position), "visual_index": int(position - visual_start),
                            "slot": int(slot), "expert": int(ids[slot]),
                            "weight": float(weights[slot]),
                            "hidden": hidden.astype(np.float32),
                            "output": raw_outputs[slot].astype(np.float32),
                        })
                if not np.all(coverage == 1):
                    raise AssertionError(f"{sample} L{layer}: incomplete slots {coverage}")
                normalized_weights = weights / weights.sum()
                reconstructed = np.sum(normalized_weights[:, None] * outputs, axis=0)
                checks.append({
                    "sample_id": sample, "layer": layer, "modality": modality,
                    "cosine": float(_cosine_rows(reconstructed[None], stock[None])[0]),
                    "relative_l2": float(np.linalg.norm(reconstructed - stock) /
                                         (np.linalg.norm(stock) + 1e-12)),
                })
    return groups, pd.DataFrame(checks)


def _matched(groups: dict[tuple[int, int, str], list[dict[str, Any]]]) -> dict[
        tuple[int, int], dict[str, list[dict[str, Any]]]]:
    output = {}
    keys = {(layer, expert) for layer, expert, _ in groups}
    for layer, expert in sorted(keys):
        visual = groups.get((layer, expert, "visual"), [])
        text = groups.get((layer, expert, "text"), [])
        count = min(len(visual), len(text), CAP_GROUP)
        if count < MIN_GROUP:
            continue
        output[(layer, expert)] = {
            "visual": _stable_entries(visual, count),
            "text": _stable_entries(text, count),
        }
    return output


def _reconstruct(entries: list[dict[str, Any]], ratio: float,
                 method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.stack([row["hidden"] for row in entries])
    outputs = np.stack([row["output"] for row in entries])
    count = max(1, int(math.ceil(len(entries) * ratio)))
    feature_distance = _distance(outputs if method == "oracle" else hidden)
    medoids = _greedy_medoids(feature_distance, count)
    nearest = np.argmin(feature_distance[:, medoids], axis=1)
    predicted = outputs[medoids[nearest]]
    cosine = _cosine_rows(predicted, outputs)
    relative_l2 = (np.linalg.norm(predicted - outputs, axis=1) /
                   np.maximum(np.linalg.norm(outputs, axis=1), 1e-12))
    return cosine, relative_l2, medoids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    result = args.result_dir
    manifest = json.loads((result / "manifest.json").read_text())
    groups, correctness = _load_assignments(result, manifest)
    correctness.to_csv(result / "raw_output_correctness.csv", index=False)
    matched = _matched(groups)
    if not matched:
        raise RuntimeError("no matched groups")

    diversity_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    group_manifest: list[dict[str, Any]] = []
    for (layer, expert), modalities in matched.items():
        group_manifest.append({"layer": layer, "expert": expert,
                               "matched_count": len(modalities["visual"]),
                               "visual_available": len(groups[(layer, expert, "visual")]),
                               "text_available": len(groups[(layer, expert, "text")]),
                               "visual_images": len({row["sample_id"] for row in
                                                     groups[(layer, expert, "visual")]}),
                               "text_images": len({row["sample_id"] for row in
                                                   groups[(layer, expert, "text")]})})
        for modality, entries in modalities.items():
            outputs = np.stack([row["output"] for row in entries])
            diversity_rows.append({"layer": layer, "region": _region(layer),
                                   "expert": expert, "modality": modality,
                                   "tokens": len(entries), **_diversity(outputs)})
            for method in ("oracle", "practical"):
                for ratio in RATIOS:
                    cosine, relative_l2, medoids = _reconstruct(entries, ratio, method)
                    nonrepresentative = np.ones(len(entries), dtype=bool)
                    nonrepresentative[medoids] = False
                    if bool(nonrepresentative.any()):
                        evaluation_cosine = cosine[nonrepresentative]
                        evaluation_relative_l2 = relative_l2[nonrepresentative]
                    else:
                        evaluation_cosine = np.ones(1)
                        evaluation_relative_l2 = np.zeros(1)
                    reconstruction_rows.append({
                        "layer": layer, "region": _region(layer), "expert": expert,
                        "modality": modality, "method": method, "ratio": ratio,
                        "tokens": len(entries), "representatives": len(medoids),
                        "evaluated_nonrepresentatives": int(nonrepresentative.sum()),
                        "median_cosine": float(np.median(evaluation_cosine)),
                        "mean_cosine": float(np.mean(evaluation_cosine)),
                        "median_relative_l2": float(np.median(evaluation_relative_l2)),
                        "mean_relative_l2": float(np.mean(evaluation_relative_l2)),
                        "pass_fraction": float(np.mean((evaluation_cosine >= .99) &
                                                       (evaluation_relative_l2 <= .10))),
                        "all_token_pass_fraction": float(np.mean(
                            (cosine >= .99) & (relative_l2 <= .10))),
                    })

    diversity = pd.DataFrame(diversity_rows)
    recon = pd.DataFrame(reconstruction_rows)
    group_manifest_frame = pd.DataFrame(group_manifest)
    diversity.to_csv(result / "within_expert_diversity.csv", index=False)
    recon.to_csv(result / "representative_reconstruction.csv", index=False)
    group_manifest_frame.to_csv(result / "matched_group_manifest.csv", index=False)

    required_rows = []
    for keys, local in recon.groupby(["layer", "expert", "modality", "method"]):
        passing = local[(local.median_cosine >= .99) & (local.median_relative_l2 <= .10)]
        required = float(passing.ratio.min()) if len(passing) else 1.0
        required_rows.append(dict(zip(("layer", "expert", "modality", "method"), keys,
                                      strict=True), required_ratio=required))
    required = pd.DataFrame(required_rows)
    required.to_csv(result / "required_representative_ratio.csv", index=False)

    category_rows = []
    for (layer, expert), modalities in matched.items():
        categories = ({row["category"] for row in modalities["visual"]} &
                      {row["category"] for row in modalities["text"]})
        for category in sorted(categories):
            visual = [row for row in groups[(layer, expert, "visual")]
                      if row["category"] == category]
            text = [row for row in groups[(layer, expert, "text")]
                    if row["category"] == category]
            count = min(len(visual), len(text), 16)
            if count < MIN_GROUP:
                continue
            for modality, entries in (("visual", visual), ("text", text)):
                selected = _stable_entries(entries, count)
                category_rows.append({"layer": layer, "expert": expert,
                                      "category": category, "modality": modality,
                                      "tokens": count,
                                      "images": len({row["sample_id"] for row in selected}),
                                      **_diversity(np.stack([row["output"] for row in selected]))})
    category_diversity = pd.DataFrame(category_rows)
    category_diversity.to_csv(result / "category_matched_diversity.csv", index=False)

    diversity_differences = {}
    direction_count = 0
    specifications = {
        "pairwise_cosine": 1,
        "normalized_dispersion": -1,
        "centered_participation_rank": -1,
        "nearest_neighbor_relative_l2": -1,
    }
    for metric, expected_sign in specifications.items():
        pivot = diversity.pivot(index=["layer", "expert"], columns="modality", values=metric).dropna()
        delta = (pivot.visual - pivot.text).to_numpy()
        mean, low, high = _bootstrap(delta, SEED + len(diversity_differences))
        agrees = mean * expected_sign > 0
        direction_count += int(agrees)
        diversity_differences[metric] = {"visual_minus_text": mean,
                                         "ci": [low, high], "agrees": agrees}

    required_summary = {}
    for method in ("oracle", "practical"):
        pivot = required[required.method == method].pivot(
            index=["layer", "expert"], columns="modality", values="required_ratio").dropna()
        gap = (pivot.text - pivot.visual).to_numpy()
        mean, low, high = _bootstrap(gap, SEED + (10 if method == "oracle" else 11))
        required_summary[method] = {
            "visual_median": float(pivot.visual.median()),
            "text_median": float(pivot.text.median()),
            "text_minus_visual_mean": mean,
            "ci": [low, high],
        }

    quality = {}
    for method in ("oracle", "practical"):
        quality[method] = {}
        for ratio in (0.25, 0.50, 0.75):
            quality[method][str(ratio)] = {}
            for modality in ("visual", "text"):
                local = recon[(recon.method == method) & (recon.ratio == ratio) &
                              (recon.modality == modality)]
                quality[method][str(ratio)][modality] = {
                    "median_cosine": float(local.median_cosine.median()),
                    "median_relative_l2": float(local.median_relative_l2.median()),
                    "pass_fraction": float(local.pass_fraction.mean()),
                }

    practical50 = quality["practical"]["0.5"]["visual"]
    oracle50 = quality["oracle"]["0.5"]["visual"]
    practical_gap = required_summary["practical"]["text_minus_visual_mean"]
    oracle_gap = required_summary["oracle"]["text_minus_visual_mean"]
    go = (practical50["median_cosine"] >= .99 and
          practical50["median_relative_l2"] <= .10 and
          practical_gap >= .20 and direction_count >= 3 and
          required_summary["practical"]["ci"][0] > 0)
    hold = ((practical50["median_cosine"] >= .95 and
             practical50["median_relative_l2"] <= .20 and
             practical_gap > 0 and direction_count >= 2) or
            (oracle50["median_cosine"] >= .99 and
             oracle50["median_relative_l2"] <= .10 and
             oracle_gap > 0 and practical_gap > 0))
    status = "GO" if go else "HOLD" if hold else "NO-GO"

    headroom: dict[str, Any] | None = None
    if status in ("GO", "HOLD"):
        practical_required = required[required.method == "practical"]
        ratio_by_group = {(int(row.layer), int(row.expert), row.modality): row.required_ratio
                          for row in practical_required.itertuples()}
        totals = {"all_visual_assignments": 0, "representative_rows": 0,
                  "late_visual_assignments": 0, "late_representative_rows": 0}
        for (layer, expert, modality), entries in groups.items():
            if modality != "visual" or (layer, expert, modality) not in ratio_by_group:
                continue
            ratio = ratio_by_group[(layer, expert, modality)]
            rows = len(entries)
            representatives = int(math.ceil(rows * ratio))
            totals["all_visual_assignments"] += rows
            totals["representative_rows"] += representatives
            if _region(layer) == "late":
                totals["late_visual_assignments"] += rows
                totals["late_representative_rows"] += representatives
        headroom = {
            **totals,
            "gemm_row_reduction": (1 - totals["representative_rows"] /
                                   max(totals["all_visual_assignments"], 1)),
            "late_gemm_row_reduction": (1 - totals["late_representative_rows"] /
                                        max(totals["late_visual_assignments"], 1)),
            "ep_assignment_reduction_oracle": (1 - totals["representative_rows"] /
                                                max(totals["all_visual_assignments"], 1)),
            "spatial_comparison": "not estimated: matched groups pool source images; cross-image spatial distance is undefined",
        }

    figures = result / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"visual": "#4472c4", "text": "#ed7d31"}
    metrics = [("pairwise_cosine", "Pairwise output cosine"),
               ("normalized_dispersion", "Normalized dispersion"),
               ("centered_participation_rank", "Centered participation rank"),
               ("nearest_neighbor_relative_l2", "NN relative L2")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (metric, label) in zip(axes.flat, metrics, strict=True):
        means = diversity.groupby(["layer", "modality"], as_index=False)[metric].mean()
        for modality in ("visual", "text"):
            local = means[means.modality == modality]
            ax.plot(local.layer, local[metric], marker="o", color=colors[modality], label=modality)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    axes[1, 0].set_xlabel("MoE layer")
    axes[1, 1].set_xlabel("MoE layer")
    fig.suptitle("Matched within-expert output redundancy")
    fig.tight_layout()
    fig.savefig(figures / "plot1_within_expert_output_redundancy.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, method in zip(axes, ("oracle", "practical"), strict=True):
        means = recon[recon.method == method].groupby(["ratio", "modality"], as_index=False).agg(
            cosine=("median_cosine", "median"), relative_l2=("median_relative_l2", "median"))
        for modality in ("visual", "text"):
            local = means[means.modality == modality]
            ax.plot(local.ratio, local.cosine, marker="o", color=colors[modality],
                    label=f"{modality} cosine")
            ax.plot(local.ratio, local.relative_l2, marker="x", linestyle="--",
                    color=colors[modality], label=f"{modality} rel-L2")
        ax.axhline(.99, color="black", linewidth=.8, alpha=.4)
        ax.axhline(.10, color="black", linewidth=.8, alpha=.4)
        ax.set(xlabel="Representative ratio", title=method.capitalize())
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Representative reconstruction quality")
    fig.tight_layout()
    fig.savefig(figures / "plot2_reconstruction_vs_rep_ratio.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    width = .36
    for offset, modality in ((-.18, "visual"), (.18, "text")):
        ax.bar(x + offset, [required_summary[m][f"{modality}_median"] for m in ("oracle", "practical")],
               width, color=colors[modality], label=modality)
    ax.set_xticks(x, ["Output-space oracle", "Hidden-state practical"])
    ax.set(ylabel="Required representative ratio", ylim=(0, 1.05),
           title="Representatives required for 0.99 cosine / 0.10 relative-L2")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot3_visual_vs_text_rep_ratio.png", dpi=200)
    plt.close(fig)

    if headroom is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        values = [headroom["gemm_row_reduction"], headroom["late_gemm_row_reduction"],
                  headroom["ep_assignment_reduction_oracle"]]
        ax.bar(["All GEMM rows", "Late GEMM rows", "EP assignments\n(oracle)"], values,
               color=["#4472c4", "#70ad47", "#a5a5a5"])
        ax.set(ylabel="Theoretical reduction", ylim=(0, 1), title="Conditional system headroom")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(figures / "plot4_system_headroom.png", dpi=200)
        plt.close(fig)

    summary = {
        "INTRA_EXPERT_VISUAL_REDUNDANCY": status,
        "POC4": "RUN" if headroom is not None else "NOT-RUN",
        "samples": len(manifest["samples"]),
        "matched_layer_expert_groups": len(matched),
        "matched_assignments_per_modality": int(sum(len(row["visual"]) for row in matched.values())),
        "matched_group_image_coverage": {
            "visual_median": float(group_manifest_frame.visual_images.median()),
            "text_median": float(group_manifest_frame.text_images.median()),
            "visual_min": int(group_manifest_frame.visual_images.min()),
            "text_min": int(group_manifest_frame.text_images.min()),
        },
        "correctness": {"min_cosine": float(correctness.cosine.min()),
                        "median_cosine": float(correctness.cosine.median()),
                        "max_relative_l2": float(correctness.relative_l2.max()),
                        "median_relative_l2": float(correctness.relative_l2.median())},
        "diversity_by_region": diversity.groupby(["region", "modality"], as_index=False).agg(
            groups=("expert", "size"), pairwise_cosine=("pairwise_cosine", "mean"),
            normalized_dispersion=("normalized_dispersion", "mean"),
            centered_participation_rank=("centered_participation_rank", "mean"),
            nearest_neighbor_relative_l2=("nearest_neighbor_relative_l2", "mean")).to_dict("records"),
        "diversity_differences": diversity_differences,
        "category_matched_groups": int(len(category_diversity) // 2),
        "category_matched_by_category": ({
            str(key): int(value // 2) for key, value in
            category_diversity.category.value_counts().items()
        } if len(category_diversity) else {}),
        "quality": quality,
        "required_ratio": required_summary,
        "direction_count": direction_count,
        "headroom": headroom,
    }
    _json(result / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
