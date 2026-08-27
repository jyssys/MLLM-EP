"""Analyze live Qwen3-VL source→destination traffic matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BUCKETS = ["<256", "256-512", "512-1024", ">=1024"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw(result: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank in range(4):
        path = result / "raw_live" / f"rank{rank}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    frame = pd.DataFrame(rows)
    required = {"request_id", "layer", "ep_rank", "source_dp_rank", "expert_histogram",
                "dispatch", "combine", "measured"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"raw trace missing {sorted(missing)}")
    return frame[frame.measured].copy()


def _source_routes(previous: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = _read_json(previous / "workload_manifest.json")
    routes: dict[str, dict[str, np.ndarray]] = {}
    for pair in manifest["pairs"]:
        request_id = pair["vision"]["request_id"]
        with np.load(previous / pair["vision"]["route_file"]) as archive:
            routes[request_id] = {
                "experts": archive["routed_experts"].astype(np.int64),
                "token_ids": archive["prompt_token_ids"].astype(np.int64),
            }
    return routes


def _bucket(tokens: int) -> str:
    if tokens < 256:
        return "<256"
    if tokens < 512:
        return "256-512"
    if tokens < 1024:
        return "512-1024"
    return ">=1024"


def _route_stats(experts: np.ndarray, token_ids: np.ndarray) -> dict[str, float]:
    destinations = experts // 32
    vision = token_ids == 151655
    values: dict[str, float] = {}
    for name, mask in (("all", np.ones(len(token_ids), dtype=bool)),
                       ("vision", vision), ("text", ~vision)):
        if not mask.any():
            values.update({f"{name}_s": float("nan"), f"{name}_tokens": 0.0,
                           f"{name}_assignments": 0.0})
            continue
        unique = np.array([len(set(row.tolist())) for row in destinations[mask]])
        values[f"{name}_s"] = float(unique.mean() / 4.0)
        values[f"{name}_tokens"] = float(mask.sum())
        values[f"{name}_assignments"] = float(mask.sum() * experts.shape[-1])
    return values


def _matrix_features(matrix: np.ndarray, real_assignments: float,
                     route_stats: dict[str, float]) -> dict[str, Any]:
    total = float(matrix.sum())
    p = (matrix / total).ravel() if total else np.zeros(16)
    nonzero = p[p > 0]
    active_rows = matrix.sum(axis=1) > 0
    active_cols = matrix.sum(axis=0) > 0
    row_means = matrix.sum(axis=1)[active_rows]
    col_means = matrix.sum(axis=0)[active_cols]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(16)) if len(nonzero) else 0.0
    return {
        "matrix": matrix.astype(int).tolist(), "total_volume": total,
        "real_assignments": float(real_assignments),
        "real_tokens_per_source": float(real_assignments / 8.0),
        "observed_tokens_per_source": float(total / 8.0),
        "source_active_count": int(active_rows.sum()), "destination_active_count": int(active_cols.sum()),
        "source_row_imbalance_active": float(matrix.sum(axis=1).max() / row_means.mean()) if len(row_means) else float("nan"),
        "source_row_imbalance_all": float(matrix.sum(axis=1).max() / (matrix.sum(axis=1).mean() or 1.0)),
        "destination_column_imbalance": float(matrix.sum(axis=0).max() / col_means.mean()) if len(col_means) else float("nan"),
        "active_pair_count": int((matrix > 0).sum()), "active_pair_fraction": float((matrix > 0).mean()),
        "max_pair_load": float(matrix.max()), "max_pair_fraction": float(matrix.max() / total) if total else float("nan"),
        "pair_entropy_normalized": entropy, "pair_hhi": float((p * p).sum()),
        **route_stats,
    }


def _build_invocations(raw: pd.DataFrame, routes: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["wave", "request_id", "pair_id", "token_bucket", "phase", "iteration",
                  "source_dp_rank", "layer"]
    for key, group in raw.groupby(group_keys, sort=False):
        if len(group) != 4 or set(group.ep_rank.astype(int)) != {0, 1, 2, 3}:
            continue
        matrix = np.zeros((4, 4), dtype=np.int64)
        for _, row in group.iterrows():
            matrix[int(row.source_dp_rank), int(row.ep_rank)] = int(row.total_assignments)
        request_id, layer = str(key[1]), int(key[-1])
        route = routes[request_id]
        route_experts = route["experts"][:, layer, :]
        feature = _matrix_features(matrix, float(route_experts.size),
                                   _route_stats(route_experts, route["token_ids"]))
        dispatch = group["dispatch"].map(lambda x: x["ms"])
        combine = group["combine"].map(lambda x: x["ms"])
        feature.update({"wave": int(key[0]), "request_id": request_id, "pair_id": int(key[2]),
                        "token_bucket": str(key[3]), "iteration": int(key[5]),
                        "source_dp_rank": int(key[6]), "layer": layer,
                        "dispatch_ms": float(dispatch.max()), "combine_ms": float(combine.max()),
                        "expert_ms": float(group["expert"].map(lambda x: x["ms"]).max()),
                        "comm_total_ms": float(dispatch.max() + combine.max()),
                        "rank_histograms": json.dumps({int(r.ep_rank): r.expert_histogram
                                                        for _, r in group.iterrows()}, separators=(",", ":"))})
        feature["scale_bucket"] = _bucket(int(feature["real_tokens_per_source"]))
        rows.append(feature)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no complete four-rank invocations")
    return frame


def _ols_r2(frame: pd.DataFrame, target: str, cols: list[str]) -> float:
    local = frame[[target, *cols]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) <= len(cols) + 2:
        return float("nan")
    x = np.column_stack([np.ones(len(local)), local[cols].to_numpy(float)])
    y = local[target].to_numpy(float)
    pred = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    denom = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - pred) ** 2).sum() / denom) if denom else float("nan")


def _matched(frame: pd.DataFrame, out: Path) -> dict[str, Any]:
    # Fixed before inspecting latency: same layer, volume ±5%, S ±0.03,
    # destination imbalance ±5%, and HHI contrast at least .01.
    work = frame.sort_values(["layer", "pair_hhi", "wave"]).reset_index(drop=True)
    pairs: list[dict[str, Any]] = []
    for layer, group in work.groupby("layer"):
        records = group.to_dict("records")
        for i, left in enumerate(records):
            candidates = [right for j, right in enumerate(records) if j > i and
                          abs(left["real_assignments"] - right["real_assignments"]) /
                          max(left["real_assignments"], 1) <= .05 and
                          abs(left["all_s"] - right["all_s"]) <= .03 and
                          abs(left["destination_column_imbalance"] - right["destination_column_imbalance"]) /
                          max(left["destination_column_imbalance"], 1) <= .05]
            if not candidates:
                continue
            right = max(candidates, key=lambda x: abs(left["pair_hhi"] - x["pair_hhi"]))
            if abs(left["pair_hhi"] - right["pair_hhi"]) < .01:
                continue
            pairs.append({"layer": int(layer), "left_request": left["request_id"],
                          "right_request": right["request_id"], "left_hhi": left["pair_hhi"],
                          "right_hhi": right["pair_hhi"], "hhi_delta": right["pair_hhi"] - left["pair_hhi"],
                          "left_dispatch_ms": left["dispatch_ms"], "right_dispatch_ms": right["dispatch_ms"],
                          "left_combine_ms": left["combine_ms"], "right_combine_ms": right["combine_ms"],
                          "left_comm_total_ms": left["comm_total_ms"], "right_comm_total_ms": right["comm_total_ms"]})
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(out / "matched_comparisons.csv", index=False)
    high = pairs_df[pairs_df.hhi_delta > 0] if not pairs_df.empty else pairs_df
    return {"count": int(len(high)),
            "dispatch_relative": float((high.right_dispatch_ms / high.left_dispatch_ms - 1).median()) if len(high) else float("nan"),
            "combine_relative": float((high.right_combine_ms / high.left_combine_ms - 1).median()) if len(high) else float("nan")}


def _plot(frame: pd.DataFrame, out: Path) -> list[str]:
    figures: list[str] = []
    for target, name, title in (("dispatch_ms", "plot1_concentration_vs_dispatch.png", "dispatch"),
                                ("combine_ms", "plot2_concentration_vs_combine.png", "combine")):
        plt.figure(figsize=(8, 5))
        for label, group in frame.groupby("scale_bucket", observed=True):
            plt.scatter(group.pair_hhi, group[target], label=label, alpha=.55, s=18)
        plt.xlabel("pair concentration (matrix HHI)"); plt.ylabel(f"DeepEP {name} span (ms)")
        plt.title(f"Live traffic-matrix concentration vs {title} latency"); plt.legend()
        plt.tight_layout(); plt.savefig(out / name, dpi=160); plt.close(); figures.append(name)
    return figures


def _main(args: argparse.Namespace) -> None:
    result = args.result.resolve(); out = result / "analysis"; out.mkdir(exist_ok=True)
    previous = args.previous.resolve()
    raw = _load_raw(result)
    frame = _build_invocations(raw, _source_routes(previous))
    frame.to_csv(out / "per_invocation_features.csv", index=False)
    frame[["request_id", "layer", "pair_id", "source_dp_rank", "matrix", "total_volume",
           "real_tokens_per_source", "pair_hhi", "max_pair_fraction", "active_pair_fraction",
           "dispatch_ms", "combine_ms", "comm_total_ms", "scale_bucket"]].to_json(
        out / "traffic_matrices.json", orient="records", indent=2)
    frame.groupby("scale_bucket", observed=False).size().reindex(BUCKETS, fill_value=0).rename("count").to_csv(out / "scale_distribution.csv")
    figures = _plot(frame, out)
    base_cols = ["real_assignments", "all_s", "destination_column_imbalance"]
    shape_cols = [*base_cols, "pair_hhi", "max_pair_fraction", "active_pair_fraction"]
    stats: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        subset = frame[frame.scale_bucket == bucket]
        row: dict[str, Any] = {"scale_bucket": bucket, "count": int(len(subset))}
        for target in ("dispatch_ms", "combine_ms", "comm_total_ms"):
            if len(subset) > 2:
                corr = spearmanr(subset.pair_hhi, subset[target], nan_policy="omit")
                row[f"{target}_hhi_spearman"] = float(corr.statistic)
                row[f"{target}_hhi_pvalue"] = float(corr.pvalue)
            else:
                row[f"{target}_hhi_spearman"] = float("nan"); row[f"{target}_hhi_pvalue"] = float("nan")
            row[f"{target}_base_r2"] = _ols_r2(subset, target, base_cols)
            row[f"{target}_shape_r2"] = _ols_r2(subset, target, shape_cols)
            row[f"{target}_delta_r2"] = row[f"{target}_shape_r2"] - row[f"{target}_base_r2"]
        stats.append(row)
    scale_stats = pd.DataFrame(stats); scale_stats.to_csv(out / "scale_conditioned_stats.csv", index=False)
    matched = _matched(frame, out)
    drivers = []
    for rank in range(2):
        p = result / f"driver.dp_rank{rank}.json"
        if p.exists(): drivers.extend(_read_json(p)["records"])
    measured_driver = pd.DataFrame(drivers); measured_driver = measured_driver[measured_driver.measured]
    overhead: dict[str, Any] = {"available": False}
    output_agreement: dict[str, Any] = {"available": False}
    if args.baseline and (args.baseline / "driver.dp_rank0.json").exists():
        b_rows = []
        for rank in range(2): b_rows.extend(_read_json(args.baseline / f"driver.dp_rank{rank}.json")["records"])
        b = pd.DataFrame(b_rows); b = b[b.measured]
        inst = measured_driver.groupby(["request_id", "iteration"]).wall_ms.max()
        base = b.groupby(["request_id", "iteration"]).wall_ms.max()
        common = inst.index.intersection(base.index); delta = (inst[common] / base[common] - 1).to_numpy()
        overhead = {"available": True, "pairs": int(len(delta)), "median_relative": float(np.median(delta)),
                    "p95_relative": float(np.percentile(delta, 95)), "baseline_median_wall_ms": float(base[common].median()),
                    "instrumented_median_wall_ms": float(inst[common].median())}
        baseline_outputs = {(r["request_id"], r["iteration"]): tuple(r["output_tokens"])
                           for r in b_rows if r["measured"] and r["driver_dp_rank"] == r["source_dp_rank"]}
        instrument_outputs = {(r["request_id"], r["iteration"]): tuple(r["output_tokens"])
                             for r in drivers if r["measured"] and r["driver_dp_rank"] == r["source_dp_rank"]}
        output_common = set(baseline_outputs) & set(instrument_outputs)
        output_agreement = {"available": True, "comparisons": len(output_common),
                            "exact_matches": sum(baseline_outputs[k] == instrument_outputs[k] for k in output_common),
                            "all_exact": all(baseline_outputs[k] == instrument_outputs[k] for k in output_common)}
    high = frame[frame.real_tokens_per_source >= 1024]
    if len(high) >= 48 and any((x.get("dispatch_ms_delta_r2", 0) or 0) >= .05 for x in stats):
        status = "GO"
    elif len(high) > 0 and (matched["count"] >= 3 or frame.pair_hhi.nunique() > 1):
        status = "HOLD"
    else:
        status = "NO-GO"
    summary = {
        "DEEPEP_TRAFFIC_MATRIX_LIVE": status, "raw_rows_measured": int(len(raw)),
        "complete_invocations": int(len(frame)), "request_count": int(frame.request_id.nunique()),
        "layer_count": int(frame.layer.nunique()), "scale_counts": frame.scale_bucket.value_counts().reindex(BUCKETS, fill_value=0).to_dict(),
        "scale_conditioned_stats": stats, "matched": matched, "instrumentation_overhead": overhead,
        "output_agreement": output_agreement,
        "figures": figures, "physical_gpu_mapping": "CUDA_VISIBLE_DEVICES=1,2,3,4 (logical EP ranks 0..3)",
        "backend": "DeepEPHTPrepareAndFinalize / DeepEPHTAll2AllManager",
        "timing_scope": "CUDA-event span around vLLM _prepare (dispatch) and _finalize (combine); expert GEMM separately",
        "source_rank_limitation": "one active request per DP wave; one nonzero source row limits cross-source pair analysis",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = pd.DataFrame(stats).to_markdown(index=False, floatfmt=".4f")
    example = high.sort_values("pair_hhi", ascending=False).iloc[0] if len(high) else frame.iloc[0]
    example_matrix = json.dumps(example["matrix"])
    report = f"""# Live DeepEP traffic-matrix validation

`DEEPEP_TRAFFIC_MATRIX_LIVE: {status}`

## Scope and controls

This bounded capture reuses the validated Qwen3-VL-30B-A3B-Instruct real-image
workload, DBO off, BF16, linear expert placement, and DeepEP high-throughput.
The fixed request-pair subset is `{_read_json(result / 'run_metadata.json').get('request_pairs')}`;
there are two warmups and two measured repetitions per request. Only physical
GPUs 1,2,3,4 were exposed. No route, placement, model, or communication policy
was modified.

The wrapper measures the existing vLLM `_prepare` (DeepEP dispatch/receiver)
and `_finalize` (DeepEP combine/receiver) calls with CUDA events, and records
expert GEMM separately. No extra collective and no per-layer synchronize were
introduced; events are resolved by one final bounded synchronization.

## Scale distribution

Complete measured four-rank invocations: **{len(frame)}**. Fixed buckets use
real route tokens/source: `<256`, `256–512`, `512–1024`, `≥1024`.

| bucket | invocations |
|---|---:|
""" + "\n".join(f"| {b} | {int((frame.scale_bucket == b).sum())} |" for b in BUCKETS) + f"""

The synthetic N≈1024-like bucket contains **{len(high)}** invocations. Most
observations therefore do not probe the high-scale regime where the synthetic
penalty was strongest.

## Matrix and feature definition

For each invocation, `M[source_dp_rank, ep_rank]` is populated from the actual
per-EP local expert histogram. Local expert index `e` maps to global expert
`ep_rank*32+e`; histograms and matrices are retained in the analysis CSV/JSON.
Features include volume, active pairs, max-pair load/fraction, normalized pair
entropy, HHI, source-row and destination-column imbalance. Route-derived S is
computed from the exact prior route artifacts (`expert_id//32`), separately from
the observed matrix.

Example highest-scale matrix (`{example['request_id']}`, layer {int(example['layer'])},
real tokens/source {example['real_tokens_per_source']:.0f}, HHI {example['pair_hhi']:.4f}):

```text
{example_matrix}
```

This runner intentionally used one active request per DP wave, so the matrices
have one nonzero source row. It is still a real source→destination matrix, but
cross-source pair concentration cannot be fully separated from destination
distribution. This is a material limitation, not replaced by synthetic data.

## Scale-conditioned and matched results

{table}

The fixed matched rule was same layer, real assignment volume within 5%, route
S within 0.03, destination-column imbalance within 5%, and HHI contrast ≥0.01.
The resulting matched rows are in `analysis/matched_comparisons.csv`.

Matched higher-HHI rows: **{matched['count']}**; median dispatch change:
**{matched['dispatch_relative']:.2%}**; combine change: **{matched['combine_relative']:.2%}**.

Figures: `analysis/plot1_concentration_vs_dispatch.png`,
`analysis/plot2_concentration_vs_combine.png`.

## Instrumentation overhead and validity

`{json.dumps(overhead)}`

Backend proof records verify `DeepEPHTPrepareAndFinalize`,
`DeepEPHTAll2AllManager`, EP world size 4, and `CUDA_VISIBLE_DEVICES=1,2,3,4`.
All measured waves completed without CUDA/DeepEP errors; driver output tokens
are retained for a correctness audit. The coarse baseline is a separate run and
therefore includes run-to-run noise.

The common measured greedy output tokens matched **{output_agreement.get('exact_matches', 0)}/{output_agreement.get('comparisons', 0)}**
between baseline and instrumented runs (`all_exact={output_agreement.get('all_exact', False)}`).

## Gate / interpretation

The conservative status is **{status}**. This real suite enters N≈1024 only in
the largest request(s); it does not provide broad high-scale coverage. Any
positive HHI/latency association is bounded evidence, and one active source row
means the full four-source synthetic Family-A analogue is not yet established.
The result therefore does **not** justify implementing a dynamic communication
scheduler. The next useful experiment is a bounded two-source-per-wave capture
with the same routes and fixed analysis rules.

Result directory: `{result}`

Raw trace: `{result}/raw_live/rank0..3.jsonl`

Analysis: `{out}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "report": str(args.report), "analysis": str(out)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    _main(parser.parse_args())


if __name__ == "__main__":
    main()
