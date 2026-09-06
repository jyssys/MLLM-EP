"""Trace-only partial expert completion analysis.

The input is the existing Qwen3-VL functional expert-output capture.  This
script intentionally has no vLLM dependency: it can run in a CPU analysis
environment and keeps the output explicit about what is observed (expert
output error) versus what is not measured (next-block/logit error).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOP_M = (1, 2, 4, 6, 7)
MASS = (0.70, 0.80, 0.90, 0.95, 0.99)
PASS_COS = 0.99
PASS_L2 = 0.05


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def partial_metrics(outputs: np.ndarray, weights: np.ndarray) -> list[dict]:
    """Return quality metrics for fixed-k and router-mass prefixes."""
    outputs = outputs.astype(np.float64)
    weights = np.maximum(weights.astype(np.float64), 0.0)
    order = np.argsort(-weights, kind="stable")
    outputs = outputs[order]
    weights = weights[order]
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones_like(weights) / len(weights)
        total = 1.0
    exact = np.sum((weights / total)[:, None] * outputs, axis=0)
    cumulative = np.cumsum(weights / total)
    rows = []

    def add(method: str, selected: int) -> None:
        selected = max(1, min(int(selected), len(weights)))
        raw = np.sum((weights[:selected] / total)[:, None] * outputs[:selected], axis=0)
        mass = float(weights[:selected].sum() / total)
        renorm = np.sum((weights[:selected] / max(weights[:selected].sum(), 1e-12))[:, None]
                        * outputs[:selected], axis=0)
        for variant, value in (("plain", raw), ("renormalized", renorm)):
            l2 = float(np.linalg.norm(value - exact) / (np.linalg.norm(exact) + 1e-12))
            rows.append({"method": method, "variant": variant, "selected_k": selected,
                         "router_mass": mass, "cosine": cosine(value, exact),
                         "relative_l2": l2,
                         "quality_pass": bool(cosine(value, exact) >= PASS_COS and l2 <= PASS_L2)})

    for selected in TOP_M:
        add(f"top{selected}", selected)
    for threshold in MASS:
        selected = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        add(f"mass{int(threshold * 100)}", selected)
    return rows


def load_sample_records(result: Path) -> tuple[list[dict], list[dict]]:
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    samples = {row["sample_id"]: row for row in manifest["samples"]}
    schedule = manifest.get("schedule", [])
    source_dp = {row["sample_id"]: int(row.get("source_dp_rank", row.get("dp_rank", 0)))
                 for row in schedule}
    if not source_dp:
        for rank, names in enumerate(manifest.get("partition", [])):
            for name in names:
                source_dp[name] = rank
    layers = manifest.get("policy", {}).get("layers")
    if layers is None:
        layers = sorted({int(path.stem.split("layer")[-1]) for path in (result / "raw").glob("router.*.npz")})
    rows: list[dict] = []
    correctness: list[dict] = []
    raw = result / "raw"
    for sample_id, meta in samples.items():
        dp = source_dp.get(sample_id, 0)
        router_files = [raw / f"router.{sample_id}.dp{dp}.tp{tp}.layer{layer}.npz"
                        for tp in (0, 1) for layer in []]
        for layer in layers:
            parts = []
            for tp in (0, 1):
                path = raw / f"router.{sample_id}.dp{dp}.tp{tp}.layer{layer}.npz"
                if path.exists():
                    parts.append(np.load(path))
            if not parts:
                continue
            router = {key: np.concatenate([part[key] for part in parts], axis=0)
                      for key in ("fingerprints", "selected", "topk_ids", "topk_weights")}
            selected_positions = np.flatnonzero(router["selected"])
            expert_rows: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
            for ep in range(4):
                path = raw / f"experts.{sample_id}.ep{ep}.layer{layer}.npz"
                if not path.exists():
                    continue
                arrays = np.load(path)
                for i, fingerprint in enumerate(arrays["fingerprints"]):
                    expert_rows[str(fingerprint)].append((arrays["expert_ids"][i],
                        arrays["router_weights"][i], arrays["local_mask"][i],
                        arrays["raw_outputs"][i], arrays["stock_output"][i]))
            start, end = map(int, meta["visual_span"])
            for position in selected_positions:
                fingerprint = str(router["fingerprints"][position])
                pieces = expert_rows.get(fingerprint, [])
                if not pieces:
                    continue
                ids = router["topk_ids"][position].astype(int)
                weights = router["topk_weights"][position].astype(np.float32)
                outputs = np.zeros((8, pieces[0][3].shape[-1]), dtype=np.float32)
                coverage = np.zeros(8, dtype=int)
                stock = np.zeros(outputs.shape[-1], dtype=np.float32)
                for expert_ids, expert_weights, local_mask, raw_outputs, stock_output in pieces:
                    local = np.flatnonzero(local_mask)
                    if local.size and not np.array_equal(expert_ids[local].astype(int), ids[local]):
                        continue
                    if local.size and not np.allclose(expert_weights[local], weights[local], atol=1e-5):
                        continue
                    stock += stock_output.astype(np.float32)
                    for slot in local:
                        outputs[slot] += raw_outputs[slot].astype(np.float32)
                        coverage[slot] += 1
                if not np.all(coverage == 1):
                    continue
                exact = np.sum((weights / max(float(weights.sum()), 1e-12))[:, None] * outputs, axis=0)
                correctness.append({"sample_id": sample_id, "layer": int(layer),
                                    "modality": "visual" if start <= position < end else "text",
                                    "stock_cosine": cosine(exact, stock),
                                    "stock_relative_l2": float(np.linalg.norm(exact - stock) /
                                                                 (np.linalg.norm(stock) + 1e-12))})
                entropy = float(-np.sum((weights / max(float(weights.sum()), 1e-12)) *
                                       np.log(np.maximum(weights / max(float(weights.sum()), 1e-12), 1e-12))))
                ranks = np.unique(np.clip(ids // 32, 0, 3))
                base = {"sample_id": sample_id, "category": meta.get("category", "unknown"),
                        "layer": int(layer), "position": int(position),
                        "modality": "visual" if start <= position < end else "text",
                        "router_entropy": entropy, "router_effective_k": float(np.exp(entropy)),
                        "active_expert_count": int(np.unique(ids).size),
                        "token_fanout": int(ranks.size)}
                for quality in partial_metrics(outputs, weights):
                    rows.append({**base, **quality})
    return rows, correctness


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["modality"], row["method"], row["variant"])].append(row)
    output = []
    for (modality, method, variant), values in sorted(groups.items()):
        cos = [float(v["cosine"]) for v in values]
        l2 = [float(v["relative_l2"]) for v in values]
        ks = [int(v["selected_k"]) for v in values]
        masses = [float(v["router_mass"]) for v in values]
        output.append({"modality": modality, "method": method, "variant": variant,
                       "tokens": len(values), "cosine_mean": float(np.mean(cos)),
                       "cosine_p50": percentile(cos, .5), "cosine_p90_error": percentile([1-x for x in cos], .9),
                       "relative_l2_mean": float(np.mean(l2)), "relative_l2_p50": percentile(l2, .5),
                       "relative_l2_p90": percentile(l2, .9), "pass_rate": float(np.mean([v["quality_pass"] for v in values])),
                       "selected_k_mean": float(np.mean(ks)), "router_mass_mean": float(np.mean(masses))})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, correctness = load_sample_records(args.result_dir)
    write_csv(args.result_dir / "partial_quality_rows.csv", rows)
    aggs = aggregate(rows)
    write_csv(args.result_dir / "partial_quality_summary.csv", aggs)
    write_csv(args.result_dir / "raw_output_correctness.csv", correctness)
    by_mod = defaultdict(list)
    for row in rows:
        by_mod[(row["modality"], row["method"], row["variant"])].append(row)
    summary = {"rows": len(rows), "tokens": len({(r["sample_id"], r["layer"], r["position"]) for r in rows}),
               "samples": len({r["sample_id"] for r in rows}),
               "correctness": {"min_cosine": min((r["stock_cosine"] for r in correctness), default=None),
                               "median_cosine": percentile([r["stock_cosine"] for r in correctness], .5),
                               "max_relative_l2": max((r["stock_relative_l2"] for r in correctness), default=None),
                               "median_relative_l2": percentile([r["stock_relative_l2"] for r in correctness], .5)},
               "quality_gate": {"cosine": PASS_COS, "relative_l2": PASS_L2}, "summary": aggs,
               "limitations": ["No next-layer hidden propagation or logits were captured.",
                              "No downstream speculative execution or E2E overlap was implemented.",
                              "Expert output is observed from a diagnostic hook; quality is not a model-output equivalence claim."]}
    write_json(args.result_dir / "partial_quality_summary.json", summary)

    figures = args.result_dir / "figures"
    figures.mkdir(exist_ok=True)
    for variant in ("plain", "renormalized"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for modality, color in (("visual", "#4472c4"), ("text", "#ed7d31")):
            points = [r for r in aggs if r["variant"] == variant and r["modality"] == modality and r["method"].startswith("top")]
            points.sort(key=lambda r: int(r["method"][3:]))
            if points:
                ax.plot([int(p["method"][3:]) for p in points], [p["relative_l2_mean"] for p in points], marker="o", color=color, label=modality)
        ax.set(xlabel="Top-m expert contributions", ylabel="Mean relative L2 to exact top-8", title=f"Partial completion quality ({variant})")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / f"partial_topm_{variant}.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for modality, color in (("visual", "#4472c4"), ("text", "#ed7d31")):
        points = [r for r in aggs if r["variant"] == "renormalized" and r["modality"] == modality and r["method"].startswith("mass")]
        points.sort(key=lambda r: int(r["method"][4:]))
        if points:
            ax.plot([int(p["method"][4:]) for p in points], [p["relative_l2_mean"] for p in points], marker="o", color=color, label=modality)
    ax.set(xlabel="Router mass threshold (%)", ylabel="Mean relative L2", title="Mass-threshold provisional hidden quality")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / "partial_mass_threshold.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
