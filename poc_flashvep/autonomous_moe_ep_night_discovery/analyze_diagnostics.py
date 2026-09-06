"""Fresh-trace diagnostic tests that are distinct from request A/B probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def safe_effect(low: float, high: float) -> float:
    return (high / low - 1) * 100 if low else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path)
    args = ap.parse_args()
    trace_dir = args.trace_dir or (args.root / "live_deferred/trace")
    rows, layers = [], []
    for path in sorted(trace_dir.glob("steps_pid*.jsonl")):
        pid = int(path.stem.split("pid")[-1])
        previous = None
        for line in path.open():
            item = json.loads(line)
            ctx = item.get("context") or {}
            if not (
                (ctx.get("protocol_version") == "v2_matched" and ctx.get("hypothesis") == "H03")
                or (ctx.get("protocol_version") == "v3_final" and ctx.get("hypothesis") == "H04")
                or ctx.get("protocol_version") in {"v4_dummy", "v5_dummy", "v6_dummy", "v7_dummy"}
            ):
                continue
            base = {
                "pid": pid, "hypothesis": ctx.get("hypothesis"),
                "experiment": ctx.get("experiment"), "variant": ctx.get("variant"),
                "pair_index": ctx.get("pair_index"), "dp_rank": item.get("dp_rank"),
                "tp_rank": item.get("tp_rank"), "step_id": item.get("step_id"),
                "phase": item.get("phase"), "scheduled_tokens": item.get("scheduled_tokens"),
                "num_requests": item.get("num_requests"), "host_gap_ms": item.get("host_gap_ms"),
                "host_execute_ms": item.get("host_execute_ms"), "step_cuda_ms": item.get("step_cuda_ms"),
            }
            stage = {}
            for rec in item.get("stages", []):
                if rec.get("cuda_ms") is None:
                    continue
                stage[(rec["layer"], rec["stage"])] = float(rec["cuda_ms"])
            moe = sum(v for (layer, name), v in stage.items() if name == "moe_block")
            attn = sum(v for (layer, name), v in stage.items() if name == "attention")
            base.update(
                moe_sample_ms=moe, attention_sample_ms=attn,
                shape_changed=previous is not None and previous != (
                    base["phase"], base["scheduled_tokens"], base["num_requests"]),
            )
            rows.append(base)
            for layer in [0, 12, 24, 36, 47]:
                layers.append({
                    **base, "layer": layer,
                    "attention_ms": stage.get((layer, "attention"), np.nan),
                    "moe_ms": stage.get((layer, "moe_block"), np.nan),
                    "dispatch_ms": stage.get((layer, "deepep_dispatch_enqueue"), np.nan),
                    "combine_ms": stage.get((layer, "deepep_combine_enqueue"), np.nan),
                })
            previous = (base["phase"], base["scheduled_tokens"], base["num_requests"])
    steps, layer_df = pd.DataFrame(rows), pd.DataFrame(layers)
    out_dir = args.root / "analysis"
    out_dir.mkdir(exist_ok=True)
    # Remove TP duplicates by taking rank-critical value for a DP step.
    group = ["hypothesis", "experiment", "variant", "pair_index", "dp_rank", "step_id", "phase",
             "scheduled_tokens", "num_requests"]
    logical = steps.groupby(group, as_index=False).agg({
        "step_cuda_ms": "max", "host_execute_ms": "max", "host_gap_ms": "max",
        "moe_sample_ms": "max", "attention_sample_ms": "max", "shape_changed": "max",
    })
    logical.to_csv(out_dir / "diagnostic_logical_steps.csv", index=False)
    layer_group = group + ["layer"]
    logical_layers = layer_df.groupby(layer_group, as_index=False).agg({
        "attention_ms": "max", "moe_ms": "max", "dispatch_ms": "max", "combine_ms": "max",
        "shape_changed": "max",
    })
    logical_layers.to_csv(out_dir / "diagnostic_layer_steps.csv", index=False)

    result = {"logical_steps": len(logical), "logical_layer_rows": len(logical_layers)}
    clean = logical[(logical.moe_sample_ms > 0) & (logical.attention_sample_ms > 0)]
    rho, p = spearmanr(clean.attention_sample_ms, clean.moe_sample_ms)
    result["H24_non_moe_cotail"] = {
        "spearman_attention_moe": float(rho), "p": float(p),
        "attention_tail_overlap": float(np.mean(
            (clean.attention_sample_ms >= np.percentile(clean.attention_sample_ms, 95))
            & (clean.moe_sample_ms >= np.percentile(clean.moe_sample_ms, 95))
        )),
        "independent_tail_expected": 0.0025,
    }
    # Within matched current shape, compare shape-transition and steady steps.
    transition_pairs = []
    for key, g in logical.groupby(["phase", "scheduled_tokens", "num_requests"]):
        a = g[~g.shape_changed].moe_sample_ms
        b = g[g.shape_changed].moe_sample_ms
        if len(a) >= 20 and len(b) >= 20:
            transition_pairs.append((key, float(a.median()), float(b.median()), len(a), len(b)))
    effects = [safe_effect(a, b) for _, a, b, _, _ in transition_pairs]
    result["H26_shape_transition"] = {
        "matched_shapes": len(effects),
        "median_transition_effect_pct": float(np.median(effects)) if effects else None,
        "p90_abs_effect_pct": float(np.percentile(np.abs(effects), 90)) if effects else None,
    }
    # Layer persistence and attention->MoE coupling after coarse current-shape matching.
    layer_stats = []
    for layer, g in logical_layers.dropna().groupby("layer"):
        rho, p = spearmanr(g.attention_ms, g.moe_ms)
        layer_stats.append({
            "layer": int(layer), "samples": len(g), "attention_moe_rho": float(rho),
            "p": float(p), "moe_p50_ms": float(g.moe_ms.median()),
            "moe_p99_ms": float(np.percentile(g.moe_ms, 99)),
        })
    result["H16_layer_attention_dependency"] = layer_stats
    # Approximate decode age within each A/B wave; current M/request width is matched.
    dec = logical[logical.phase.eq("decode")].copy()
    dec["age"] = dec.groupby([
        "hypothesis", "variant", "pair_index", "dp_rank"
    ]).cumcount()
    age_effects = []
    for key, g in dec.groupby(["scheduled_tokens", "num_requests"]):
        if len(g) < 100:
            continue
        q1, q3 = np.percentile(g.age, [25, 75])
        early = g[g.age <= q1].moe_sample_ms
        late = g[g.age >= q3].moe_sample_ms
        if len(early) >= 20 and len(late) >= 20:
            age_effects.append(safe_effect(float(early.median()), float(late.median())))
    result["H27_decode_age"] = {
        "matched_shapes": len(age_effects),
        "median_late_vs_early_pct": float(np.median(age_effects)) if age_effects else None,
        "p90_abs_pct": float(np.percentile(np.abs(age_effects), 90)) if age_effects else None,
    }
    (out_dir / "fresh_diagnostics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
