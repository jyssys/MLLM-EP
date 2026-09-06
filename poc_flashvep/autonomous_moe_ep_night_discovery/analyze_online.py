"""Analyze low-perturbation online serving traces for the night sprint."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VALID_COUNTERFACTUAL = {
    "H03", "H04", "H05", "H06", "H07", "H08", "H09", "H10", "H11",
    "H12", "H25", "H28", "H37",
}

# A worker can execute before the client-side context file is observed by the
# hook (or the server may intentionally use a different context path).  The
# request id still carries the hypothesis/variant/pair identity, so retain
# those records instead of silently dropping an otherwise valid trace.
EXPERIMENT_BY_HYPOTHESIS = {
    "H03": "within_vs_cross_dp_homogeneous",
    "H04": "within_vs_cross_dp_heterogeneity",
    "H05": "prefill_decode_composition",
    "H06": "large_wave_completion_spread",
    "H07": "within_vs_cross_dp_heterogeneity",
    "H08": "request_order",
    "H09": "chunk_residue",
    "H10": "few_long_vs_many_short",
    "H11": "separated_vs_staggered_mixed_phase",
    "H12": "steady_vs_bursty",
    "H14": "short_vs_long_attention_context",
    "H15": "text_vs_vision_predecessor",
    "H23": "one_image_vs_two_images",
    "H25": "dense_vs_paced_arrival",
    "H28": "stable_set_vs_turnover",
    "H31": "async_scheduler_control",
    "H32": "dp_sync_backend_control",
    "H33": "metadata_host_path_control",
    "H36": "idle_dp_vs_short_participant",
    "H37": "one_dp_output_churn",
    "H38": "metadata_step_recurrence",
    "H43": "pinned_vs_unpinned_dp",
    "H45": "common_regime_transition_repeat",
    "H46": "turnover_repeat",
    "H47": "cohort_completion_spread_repeat",
    "H48": "mixed_phase_repeat",
}


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def load_requests(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    df = pd.DataFrame(rows)
    # H03 from v2 is valid; v2 H04 was interrupted before the control audit.
    # All resumed blocks carry v3_final and are retained.
    protocol = df.get("protocol_version", pd.Series(index=df.index))
    df = df[
        ((protocol == "v2_matched") & df.hypothesis.eq("H03"))
        | ((protocol == "v3_final") & df.hypothesis.eq("H04"))
        | protocol.isin(["v4_dummy", "v5_dummy", "v6_dummy", "v7_dummy"])
    ]
    df = df[df.status.eq(200) & df.error.isna()].copy()
    keys = ["hypothesis", "experiment", "variant", "pair_index"]
    pairs = []
    for key, group in df.groupby(keys, dropna=False):
        arrivals = group.arrival_ns.astype(np.int64)
        finishes = group.finish_ns.astype(np.int64)
        pairs.append({
            **dict(zip(keys, key, strict=True)),
            "requests": len(group),
            "prompt_words": int(group.prompt_words.sum()),
            "output_tokens_requested": int(group.output_tokens_requested.sum()),
            "image_count": int(group.get("image_count", 0).sum()),
            "wave_e2e_ms": (int(finishes.max()) - int(arrivals.min())) / 1e6,
            "sum_request_e2e_ms": float(group.e2e_ms.sum()),
            "request_e2e_p50_ms": float(group.e2e_ms.median()),
            "request_e2e_p90_ms": percentile(group.e2e_ms, 90),
            "request_e2e_p99_ms": percentile(group.e2e_ms, 99),
            "ttft_p50_ms": float(group.ttft_ms.median()),
            "ttft_p99_ms": percentile(group.ttft_ms.dropna(), 99),
        })
    pair_df = pd.DataFrame(pairs)
    summaries = []
    for (hyp, exp, var), group in pair_df.groupby(
        ["hypothesis", "experiment", "variant"], dropna=False
    ):
        # First two waves are per-hypothesis warmup and excluded.
        steady = group[group.pair_index >= 2]
        if steady.empty:
            steady = group
        summaries.append({
            "hypothesis": hyp,
            "experiment": exp,
            "variant": var,
            "waves": len(steady),
            "wave_e2e_p50_ms": float(steady.wave_e2e_ms.median()),
            "wave_e2e_mean_ms": float(steady.wave_e2e_ms.mean()),
            "wave_e2e_p90_ms": percentile(steady.wave_e2e_ms, 90),
            "wave_e2e_cv": float(steady.wave_e2e_ms.std(ddof=1) / steady.wave_e2e_ms.mean()),
            "request_e2e_p50_ms": float(steady.request_e2e_p50_ms.median()),
            "request_e2e_p99_ms": float(steady.request_e2e_p99_ms.median()),
            "ttft_p50_ms": float(steady.ttft_p50_ms.median()),
            "ttft_p99_ms": float(steady.ttft_p99_ms.median()),
        })
    return pair_df, pd.DataFrame(summaries)


def load_steps(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in paths:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                ctx = item.get("context") or {}
                if not ctx:
                    # Example: chatcmpl-H36-B-0007-003-<nonce>.  This is
                    # only an identity fallback; no timing synchronization or
                    # additional instrumentation is introduced.
                    request_ids = item.get("request_ids") or []
                    match = re.search(r"(H\d+)-([AB])-([0-9]+)-", str(request_ids[0])) if request_ids else None
                    if match:
                        hyp, variant, pair_index = match.groups()
                        ctx = {
                            "hypothesis": hyp,
                            "experiment": EXPERIMENT_BY_HYPOTHESIS.get(hyp, "unknown"),
                            "variant": variant,
                            "pair_index": int(pair_index),
                            "protocol_version": "v6_dummy",
                        }
                if not (
                    (ctx.get("protocol_version") == "v2_matched" and ctx.get("hypothesis") == "H03")
                    or (ctx.get("protocol_version") == "v3_final" and ctx.get("hypothesis") == "H04")
                    or ctx.get("protocol_version") in {"v4_dummy", "v5_dummy", "v6_dummy", "v7_dummy"}
                ):
                    continue
                sums = defaultdict(float)
                maxima = defaultdict(float)
                for stage in item.get("stages", []):
                    value = stage.get("cuda_ms")
                    if value is None:
                        continue
                    sums[stage["stage"]] += float(value)
                    maxima[stage["stage"]] = max(maxima[stage["stage"]], float(value))
                rows.append({
                    "hypothesis": ctx.get("hypothesis"),
                    "experiment": ctx.get("experiment"),
                    "variant": ctx.get("variant"),
                    "pair_index": ctx.get("pair_index"),
                    "dp_rank": item.get("dp_rank"),
                    "tp_rank": item.get("tp_rank"),
                    "step_id": item.get("step_id"),
                    "phase": item.get("phase"),
                    "num_requests": item.get("num_requests"),
                    "scheduled_tokens": item.get("scheduled_tokens"),
                    "host_gap_ms": item.get("host_gap_ms"),
                    "host_execute_ms": item.get("host_execute_ms"),
                    "step_cuda_ms": item.get("step_cuda_ms"),
                    "moe_sample_ms": sums["moe_block"],
                    "attention_sample_ms": sums["attention"],
                    "decoder_sample_ms": sums["decoder_layer"],
                    "layout_sample_ms": sums["deepep_layout_enqueue"],
                    "dispatch_sample_ms": sums["deepep_dispatch_enqueue"],
                    "combine_sample_ms": sums["deepep_combine_enqueue"],
                    "max_moe_layer_ms": maxima["moe_block"],
                    "max_attention_layer_ms": maxima["attention"],
                    "max_dispatch_layer_ms": maxima["deepep_dispatch_enqueue"],
                })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw
    # TP ranks execute the same logical DP step. Keep rank-critical values once.
    index = ["hypothesis", "experiment", "variant", "pair_index", "dp_rank", "step_id", "phase"]
    logical = raw.groupby(index, as_index=False).agg({
        "num_requests": "max", "scheduled_tokens": "max", "host_gap_ms": "max",
        "host_execute_ms": "max", "step_cuda_ms": "max", "moe_sample_ms": "max",
        "attention_sample_ms": "max", "decoder_sample_ms": "max",
        "layout_sample_ms": "max", "dispatch_sample_ms": "max",
        "combine_sample_ms": "max", "max_moe_layer_ms": "max",
        "max_attention_layer_ms": "max", "max_dispatch_layer_ms": "max",
    })
    return raw, logical


def hypothesis_effects(req_summary: pd.DataFrame, logical: pd.DataFrame) -> pd.DataFrame:
    gpu = []
    if not logical.empty:
        # Sum rank-local critical spans for a request wave, then take the slower DP rank.
        per_dp = logical.groupby(
            ["hypothesis", "experiment", "variant", "pair_index", "dp_rank"],
            as_index=False,
        ).agg({
            "step_cuda_ms": "sum", "moe_sample_ms": "sum",
            "attention_sample_ms": "sum", "dispatch_sample_ms": "sum",
            "combine_sample_ms": "sum", "host_execute_ms": "sum",
        })
        per_pair = per_dp.groupby(
            ["hypothesis", "experiment", "variant", "pair_index"], as_index=False
        ).agg({c: "max" for c in [
            "step_cuda_ms", "moe_sample_ms", "attention_sample_ms",
            "dispatch_sample_ms", "combine_sample_ms", "host_execute_ms",
        ]})
        for (hyp, exp, var), group in per_pair.groupby(
            ["hypothesis", "experiment", "variant"]
        ):
            steady = group[group.pair_index >= 2]
            if steady.empty:
                steady = group
            row = {"hypothesis": hyp, "experiment": exp, "variant": var, "gpu_waves": len(steady)}
            for col in ["step_cuda_ms", "moe_sample_ms", "attention_sample_ms",
                        "dispatch_sample_ms", "combine_sample_ms", "host_execute_ms"]:
                row[col + "_p50"] = float(steady[col].median())
                row[col + "_p90"] = percentile(steady[col], 90)
            gpu.append(row)
    gpu_df = pd.DataFrame(gpu)
    merged = req_summary.merge(gpu_df, how="left", on=["hypothesis", "experiment", "variant"])
    effects = []
    for (hyp, exp), group in merged.groupby(["hypothesis", "experiment"]):
        by = {row.variant: row for row in group.itertuples()}
        if "A" not in by or "B" not in by:
            continue
        a, b = by["A"], by["B"]
        row = {
            "hypothesis": hyp, "experiment": exp,
            "counterfactual_valid": hyp in VALID_COUNTERFACTUAL,
        }
        for col in ["wave_e2e_p50_ms", "request_e2e_p50_ms", "request_e2e_p99_ms",
                    "ttft_p50_ms", "step_cuda_ms_p50", "moe_sample_ms_p50",
                    "attention_sample_ms_p50", "dispatch_sample_ms_p50",
                    "combine_sample_ms_p50", "host_execute_ms_p50"]:
            av, bv = getattr(a, col, np.nan), getattr(b, col, np.nan)
            row[col + "_A"] = av
            row[col + "_B"] = bv
            row[col + "_B_vs_A_pct"] = (bv / av - 1) * 100 if av and np.isfinite(av) else np.nan
        slow = max(a.wave_e2e_p50_ms, b.wave_e2e_p50_ms)
        fast = min(a.wave_e2e_p50_ms, b.wave_e2e_p50_ms)
        row["direct_perfect_oracle_pct"] = (slow - fast) / slow * 100 if hyp in VALID_COUNTERFACTUAL else np.nan
        effects.append(row)
    return pd.DataFrame(effects)


def request_bootstrap(pair: pd.DataFrame, seed: int = 20260907) -> pd.DataFrame:
    """Independent-wave bootstrap for the randomized A/B median effect."""
    rng = np.random.default_rng(seed)
    out = []
    for (hyp, exp), group in pair.groupby(["hypothesis", "experiment"]):
        a = group[(group.variant == "A") & (group.pair_index >= 2)].wave_e2e_ms.to_numpy()
        b = group[(group.variant == "B") & (group.pair_index >= 2)].wave_e2e_ms.to_numpy()
        if len(a) < 3 or len(b) < 3:
            continue
        samples = []
        for _ in range(2000):
            am = np.median(rng.choice(a, len(a), replace=True))
            bm = np.median(rng.choice(b, len(b), replace=True))
            samples.append((bm / am - 1) * 100)
        out.append({
            "hypothesis": hyp, "experiment": exp, "waves_A": len(a), "waves_B": len(b),
            "B_vs_A_median_pct": (np.median(b) / np.median(a) - 1) * 100,
            "bootstrap_p2_5_pct": percentile(samples, 2.5),
            "bootstrap_p97_5_pct": percentile(samples, 97.5),
        })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--request-path", type=Path)
    ap.add_argument("--trace-dir", type=Path)
    args = ap.parse_args()
    analysis = args.root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    request_path = args.request_path or (args.root / "live_deferred/client/requests.jsonl")
    trace_dir = args.trace_dir or (args.root / "live_deferred/trace")
    pair, req_summary = load_requests(request_path)
    raw, logical = load_steps(sorted(trace_dir.glob("steps_pid*.jsonl")))
    effects = hypothesis_effects(req_summary, logical)
    bootstrap = request_bootstrap(pair)
    pair.to_csv(analysis / "request_pair_summary.csv", index=False)
    req_summary.to_csv(analysis / "hypothesis_request_summary.csv", index=False)
    logical.to_csv(analysis / "logical_step_summary.csv", index=False)
    effects.to_csv(analysis / "hypothesis_effects.csv", index=False)
    bootstrap.to_csv(analysis / "request_effect_bootstrap.csv", index=False)
    summary = {
        "request_rows": int(pair.requests.sum()) if not pair.empty else 0,
        "request_waves": len(pair), "logical_steps": len(logical),
        "hypotheses": sorted(effects.hypothesis.unique().tolist()) if not effects.empty else [],
        "max_valid_direct_oracle_pct": None if effects.empty else float(
            effects.direct_perfect_oracle_pct.max(skipna=True)
        ),
    }
    (analysis / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
