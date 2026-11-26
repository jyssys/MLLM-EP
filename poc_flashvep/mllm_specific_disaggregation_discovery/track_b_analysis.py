"""Aggregate request-paired Track B evidence and regenerate scientific figures."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from statistics import mean


def save(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(field for row in rows for field in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def interval(values, seed=20261127):
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    bootstrap = sorted(mean(rng.choices(values, k=len(values))) for _ in range(1500))
    return bootstrap[37], bootstrap[1462]


def summarize_b1(records):
    attention, hidden = defaultdict(list), defaultdict(list)
    for row in records:
        groups, norms = defaultdict(list), defaultdict(list)
        for point in row["b1"]:
            total = sum(point["key_eligible"].values())
            for target, mass in point["key_mass"].items():
                available = point["key_eligible"][target]
                prevalence = available / total
                groups[(point["group"], point["query_type"], target)].append(
                    (mass, prevalence, mass / prevalence if prevalence else None)
                )
            for target, edge in point.get("visual_edges", {}).items():
                prevalence = edge["eligible"] / total
                groups[(point["group"], point["query_type"], target)].append(
                    (edge["mass"], prevalence, edge["mass"] / prevalence if prevalence else None)
                )
            for label, value in point["alignment"].items():
                norms[(point["group"], point["query_type"], label)].append(value)
        for group, values in groups.items():
            enrich = [item[2] for item in values if item[2] is not None]
            summary = (mean(item[0] for item in values), mean(item[1] for item in values),
                       mean(enrich) if enrich else None, len(values))
            for source in dict.fromkeys(("ALL", row["source"], row["dataset"])):
                attention[(source, *group)].append(summary)
        for group, values in norms.items():
            for source in dict.fromkeys(("ALL", row["source"], row["dataset"])):
                hidden[(source, *group)].append((mean(values), len(values)))
    attn_rows, hidden_rows = [], []
    for (source, layer_group, query, key), values in sorted(attention.items()):
        enrich = [item[2] for item in values if item[2] is not None]
        low, high = interval(enrich)
        attn_rows.append({
            "source": source, "layer_group": layer_group, "query_type": query, "key_type": key,
            "n_requests": len(values), "n_with_causal_access": len(enrich),
            "sampled_query_layer_rows": sum(item[3] for item in values),
            "mass": mean(item[0] for item in values), "eligible_key_prevalence": mean(item[1] for item in values),
            "causal_enrichment": mean(enrich) if enrich else None, "enrichment_ci_low": low, "enrichment_ci_high": high,
        })
    for (source, layer_group, query, metric), values in sorted(hidden.items()):
        estimates = [item[0] for item in values]
        low, high = interval(estimates)
        hidden_rows.append({
            "source": source, "layer_group": layer_group, "query_type": query, "metric": metric,
            "n_requests": len(values), "sampled_query_layer_rows": sum(item[1] for item in values),
            "estimate": mean(estimates), "ci_low": low, "ci_high": high,
        })
    return attn_rows, hidden_rows


def summarize_b2(records, frozen, scorer):
    per_arm = defaultdict(list)
    differences = defaultdict(list)
    request_scores = []
    for row in records:
        request_id = row["request_id"]
        reference = frozen[request_id]
        baseline = scorer(reference, row["baseline"]["text"]) if scorer else {}
        pairs = defaultdict(dict)
        for arm in row["b2"]:
            layer = "-".join(map(str, arm["layers"]))
            natural = arm.get("natural")
            result = scorer(reference, natural["text"]) if scorer and natural is not None else {}
            baseline_score = baseline.get("score")
            score = result.get("score")
            delta = score - baseline_score if score is not None and baseline_score is not None else None
            values = {
                "request_id": request_id, "dataset": reference["source"], "study_source": row["source"],
                "arm": arm["arm"], "operation": arm["operation"], "category": arm["category"], "layer_range": layer,
                "input_tokens": row["prompt_tokens"], "visual_tokens": row["counts"]["visual"],
                "image_count": len(row["image_spans"]),
                "evidence": arm["evidence"],
                "controlled_prompt_rows": arm["controlled_prompt_rows"],
                "teacher_target_count": len(arm["teacher_reference_tokens"]),
                "teacher_mean_delta_nll": arm["mean_delta_nll"], "teacher_mean_KL": arm["mean_forward_kl"],
                "teacher_argmax_match": arm["teacher_argmax_match"], "masked_pairs": arm["teacher_masked_pairs_across_layers"],
                "kv_no_mask_relative_L2": arm["kv_no_mask_relative_L2"],
                "theoretical_visual_row_layer_fraction": arm["theoretical_visual_row_layer_fraction"],
                "theoretical_input_row_layer_fraction": (
                    arm["controlled_prompt_rows"] * (arm["layers"][1] - arm["layers"][0] + 1)
                    / (28 * row["prompt_tokens"]) if arm["operation"] != "kv" else None
                ),
                "natural_exact": arm.get("natural_exact"),
                "natural_output_tokens": natural["output_tokens"] if natural is not None else None,
                "baseline_output_tokens": row["baseline"]["output_tokens"],
                "natural_cap_hit": natural["finish_reason"] == "output_cap" if natural is not None else None,
                "baseline_cap_hit": row["baseline"]["finish_reason"] == "output_cap",
                "metric": baseline.get("metric"), "complete_scorer": baseline.get("complete_task_scorer"),
                "official_full_pipeline": baseline.get("official_full_pipeline"),
                "baseline_score": baseline_score, "natural_score": score, "natural_score_delta": delta,
                "new_score_regression": delta < 0 if delta is not None else None,
                "new_wrong_when_previously_correct": (
                    baseline_score == 1 and score < 1 if score is not None and baseline_score is not None else None
                ),
            }
            request_scores.append(values)
            for source in dict.fromkeys(("ALL", row["source"], reference["source"])):
                per_arm[(source, arm["arm"])].append(values)
            if arm["category"] in ("visual", "text"):
                pairs[(arm["operation"], layer)][arm["category"]] = values
        for (operation, layer), pair in pairs.items():
            if set(pair) != {"visual", "text"}:
                raise ValueError("Each intervention needs both modality arms")
            visual, text = pair["visual"], pair["text"]
            if visual["controlled_prompt_rows"] != text["controlled_prompt_rows"] or visual["masked_pairs"] != text["masked_pairs"]:
                raise ValueError("Modality arms changed work geometry")
            for source in dict.fromkeys(("ALL", row["source"], reference["source"])):
                differences[(source, operation, layer)].append({
                    "delta_nll": visual["teacher_mean_delta_nll"] - text["teacher_mean_delta_nll"],
                    "delta_kl": visual["teacher_mean_KL"] - text["teacher_mean_KL"],
                    "natural_score_delta": (
                        visual["natural_score"] - text["natural_score"]
                        if visual["natural_score"] is not None and text["natural_score"] is not None else None
                    ),
                })
    arm_rows = []
    for (source, arm), values in sorted(per_arm.items()):
        scored = [x["natural_score_delta"] for x in values if x["natural_score_delta"] is not None]
        natural = [x for x in values if x["natural_exact"] is not None]
        low, high = interval(scored)
        arm_rows.append({
            "source": source, "arm": arm, "n_requests": len(values), "n_natural": len(natural),
            "n_natural_scored": len(scored), "n_complete_scored": sum(
                item["natural_score_delta"] is not None and bool(item["complete_scorer"]) for item in values
            ), "mean_delta_nll": mean(x["teacher_mean_delta_nll"] for x in values),
            "mean_kl": mean(x["teacher_mean_KL"] for x in values),
            "mean_natural_score_delta_pp": 100 * mean(scored) if scored else None,
            "natural_score_ci_low_pp": 100 * low if low is not None else None,
            "natural_score_ci_high_pp": 100 * high if high is not None else None,
            "natural_exact": sum(item["natural_exact"] for item in natural),
            "new_regressions": sum(item["new_score_regression"] is True for item in values),
            "new_wrong_when_previously_correct": sum(item["new_wrong_when_previously_correct"] is True for item in values),
            "natural_caps": sum(item["natural_cap_hit"] for item in natural),
            "baseline_caps": sum(item["baseline_cap_hit"] for item in natural),
        })
    paired_rows = []
    for (source, operation, layer_range), values in sorted(differences.items()):
        nll = [row["delta_nll"] for row in values]
        scores = [row["natural_score_delta"] for row in values if row["natural_score_delta"] is not None]
        low, high = interval(nll)
        paired_rows.append({
            "source": source, "operation": operation, "layer_range": layer_range, "n_requests": len(values),
            "visual_minus_text_delta_nll": mean(nll), "delta_nll_ci_low": low, "delta_nll_ci_high": high,
            "visual_minus_text_delta_kl": mean(row["delta_kl"] for row in values),
            "n_scored": len(scores), "visual_minus_text_score_pp": 100 * mean(scores) if scores else None,
        })
    return request_scores, arm_rows, paired_rows


def figures(output, attention, hidden, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    groups = sorted({row["layer_group"] for row in attention})
    live = {(row["layer_group"], row["query_type"], row["key_type"]): row
            for row in attention if row["source"] == "ALL"}
    queries = ("visual", "question", "other_prompt_text", "generated")
    colors = ("#30758e", "#a84d28", "#4e8357", "#7c4998")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True, sharex=True)
    for axis, query, color in zip(axes.flat, queries, colors, strict=True):
        selected = [live.get((group, query, "visual"), {}) for group in groups]
        mass = [row.get("mass", np.nan) for row in selected]
        prevalence = [row.get("eligible_key_prevalence", np.nan) for row in selected]
        axis.plot(groups, mass, "o-", color=color, label="Attention mass")
        axis.plot(groups, prevalence, "--", color="#555555", label="Causally eligible share")
        axis.set_title(query.replace("_", " ").title())
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=.2)
        axis.tick_params(axis="x", labelrotation=30)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supylabel("Visual probability / eligible-key fraction")
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"track_b_visual_attention_flow.{ext}", dpi=180)
    plt.close(fig)

    live = {(row["layer_group"], row["query_type"]): row for row in hidden
            if row["source"] == "ALL" and row["metric"] == "hidden_relative_L2"}
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for query, color in zip(queries, colors, strict=True):
        series = [live.get((group, query), {}) for group in groups]
        axis.plot(groups, [row.get("estimate", np.nan) for row in series], "o-", label=query.replace("_", " "), color=color)
    axis.grid(axis="y", alpha=.2)
    axis.legend(frameon=False)
    axis.set_ylabel("Within-layer residual update relative L2")
    axis.set_xlabel("LLM layer group")
    axis.tick_params(axis="x", labelrotation=30)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"track_b_hidden_alignment.{ext}", dpi=180)
    plt.close(fig)

    comparison = {(row["operation"], row["layer_range"]): row for row in paired if row["source"] == "ALL"}
    layers = sorted({key[1] for key in comparison}, key=lambda label: int(label.split("-")[0]))
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8), sharex=True, constrained_layout=True)
    for axis, operation in zip(axes, ("attention", "mlp", "kv"), strict=True):
        displayed = [comparison.get((operation, layer), {}) for layer in layers]
        center = np.asarray([row.get("visual_minus_text_delta_nll", np.nan) for row in displayed])
        lower = np.asarray([row.get("delta_nll_ci_low") if row.get("delta_nll_ci_low") is not None else np.nan
                            for row in displayed])
        upper = np.asarray([row.get("delta_nll_ci_high") if row.get("delta_nll_ci_high") is not None else np.nan
                            for row in displayed])
        axis.axhline(0, color="#777777", linestyle="--")
        axis.plot(layers, center, "o-", color="#30758e")
        axis.fill_between(layers, lower, upper, color="#30758e", alpha=.2)
        axis.set_title({"attention": "Attention residual", "mlp": "MLP residual", "kv": "KV accessibility"}[operation])
    axes[-1].set_xlabel("LLM layer group")
    fig.supylabel("Visual − text teacher-forced ΔNLL (nats)")
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"track_b_matched_interventions.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset-repo", type=Path)
    parser.add_argument("--scorer-dir", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = {row["request_id"]: row for row in json.loads(args.manifest.read_text())["requests"]}
    files = [path for run in args.run for path in sorted((run / "requests").glob("*.json"))]
    records = [json.loads(path.read_text()) for path in files]
    ids = {record["request_id"] for record in records}
    if not records or len(ids) != len(records) or not ids <= frozen.keys():
        raise ValueError("Expected unique completed Track B requests from the frozen manifest")
    scorer = None
    if args.asset_repo and args.scorer_dir:
        from .quality import Scorer
        scorer = Scorer(args.asset_repo.resolve(strict=True), args.scorer_dir.resolve(strict=True))
    attention, hidden = summarize_b1(records)
    scores, arm_rows, paired = summarize_b2(records, frozen, scorer)
    save(args.out / "track_b_attention_flow.csv", attention)
    save(args.out / "track_b_hidden_alignment.csv", hidden)
    save(args.out / "track_b_request_interventions.csv", scores)
    save(args.out / "track_b_intervention_summary.csv", arm_rows)
    save(args.out / "track_b_matched_contrasts.csv", paired)
    figures(args.out, attention, hidden, paired)
    dose = [row for row in scores if row["category"] == "visual_dose55"]
    matched = [len(row["matched_pairs"]) for row in records]
    receipt = {
        "evidence": ["DIAGNOSTIC", "CAUSAL_INTERVENTION"], "manifest_sha256": sha(args.manifest),
        "planned_n": len(frozen), "completed_n": len(records),
        "missing_request_ids": sorted(frozen.keys() - ids),
        "run_configs": [json.loads((path / "config.json").read_text()) for path in args.run],
        "raw": {row["request_id"]: {"relative_file": str(path.relative_to(path.parent.parent)),
                                   "run": path.parent.parent.name, "sha256": sha(path)}
                for path, row in zip(files, records, strict=True)},
        "analysis_sha256": sha(Path(__file__)),
        "scorer": scorer.provenance if scorer else None,
        "primary_interval_unit": "request",
        "sampled_query_layer_rows": sum(len(row["b1"]) for row in records),
        "sampled_query_rows_by_class": {
            name: sum(row["sampled_query_counts"][name] for row in records)
            for name in records[0]["sampled_query_counts"]
        },
        "read_only_teacher_exact_requests": sum(row["attention_read_only_exact"] for row in records),
        "full_model_prompt_checks": [row["original_baseline_preflight"] for row in records
                                     if row["original_baseline_preflight"] is not None],
        "completed_interventions": len(scores),
        "completed_natural_interventions": sum(row["natural_exact"] is not None for row in scores),
        "paired_prompt_rows_min_mean_max": [min(matched), mean(matched), max(matched)],
        "late_visual_dose": {
            "requests": len(dose),
            "exact_natural_requests": sum(row["natural_exact"] is True for row in dose),
            "teacher_argmax_matches": sum(row["teacher_argmax_match"] for row in dose),
            "teacher_target_count": sum(row["teacher_target_count"] for row in dose),
            "mean_theoretical_visual_row_layer_fraction": (
                mean(row["theoretical_visual_row_layer_fraction"] for row in dose) if dose else None
            ),
            "mean_theoretical_input_row_layer_fraction": (
                mean(row["theoretical_input_row_layer_fraction"] for row in dose) if dose else None
            ),
            "pooled_theoretical_input_row_layer_fraction": (
                sum(row["theoretical_input_row_layer_fraction"] * row["input_tokens"] for row in dose)
                / sum(row["input_tokens"] for row in dose) if dose else None
            ),
        },
    }
    (args.out / "track_b_receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print("TRACK_B_ANALYSIS", len(records), len(scores), len(paired), "scorer" if scorer else "unscored", flush=True)


if __name__ == "__main__":
    main()
