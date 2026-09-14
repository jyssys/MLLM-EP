#!/usr/bin/env python3
"""Map measured DeepEP envelopes and hardware floors to request E2E oracles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


CLEAN_MS = {"gsm8k": 5925.0, "humaneval": 7261.0}
HIDDEN = 4096
# Same physical H100 4--7 DeepEP-normal M=4096 sweep: two-way remote BF16
# payload divided by dispatch+combine median.
SUSTAINABLE_AGGREGATE_GBPS = 854.7
O2_IRREDUCIBLE_MS = 0.020  # 10 us issue/progress floor per direction.
O3_CREDIBLE_CONTROL_MS = 0.050  # 25 us per direction; ~2.4x below measured LL tiny floor.


def isotonic_predict(bench, policy, values):
    part = bench[bench.policy == policy].groupby("global_fresh_m", as_index=False).comm_semantics_ms.median()
    part = part.sort_values("global_fresh_m")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(part.global_fresh_m, part.comm_semantics_ms)
    predicted = model.predict(np.maximum(values, 1))
    return np.where(values > 0, predicted, 0.0)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.9f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("poc_refineep"))
    parser.add_argument("--layer-metrics", type=Path, default=Path("poc_dllm_ep_discovery/LAYER_WAVE_METRICS.csv"))
    parser.add_argument("--fragmentation", type=Path, default=Path("poc_dllm_ep_discovery/FRAGMENTATION_ORACLE_BY_LAYER.csv"))
    args = parser.parse_args()
    atlas = pd.read_csv(args.root / "REFINEMENT_EP_SHAPES.csv")
    bench = pd.read_csv(args.root / "EXISTING_KERNEL_BENCH.csv")
    layers = pd.read_csv(args.layer_metrics)
    frag = pd.read_csv(args.fragmentation)

    atlas["normal_fresh_ms"] = isotonic_predict(bench, "normal_fresh", atlas.fresh_m.to_numpy())
    atlas["normal_cached_ms"] = isotonic_predict(bench, "normal_cached", atlas.fresh_m.to_numpy())
    atlas["low_latency_ms"] = isotonic_predict(bench, "low_latency", atlas.fresh_m.to_numpy())
    atlas["best_existing_ms"] = np.minimum(atlas.normal_fresh_ms, atlas.low_latency_ms)
    atlas["best_existing_path"] = np.where(
        atlas.low_latency_ms <= atlas.normal_fresh_ms, "low_latency", "normal"
    )
    atlas.loc[atlas.fresh_m == 0, "best_existing_path"] = "skip_empty"
    both_bytes = atlas.remote_bytes_one_way_bf16 * 2
    atlas["payload_lower_bound_ms"] = both_bytes / (SUSTAINABLE_AGGREGATE_GBPS * 1e6)
    nonempty = (atlas.fresh_m > 0).astype(float)
    atlas["o1_control_free_ms"] = atlas.payload_lower_bound_ms
    atlas["o2_physical_lower_ms"] = atlas.payload_lower_bound_ms + O2_IRREDUCIBLE_MS * nonempty
    atlas["o3_refineep_target_ms"] = np.minimum(
        atlas.best_existing_ms,
        atlas.payload_lower_bound_ms + O3_CREDIBLE_CONTROL_MS * nonempty,
    )
    atlas["normal_to_o0_saved_ms"] = atlas.normal_fresh_ms - atlas.best_existing_ms
    atlas["o0_to_o3_saved_ms"] = atlas.best_existing_ms - atlas.o3_refineep_target_ms
    write_csv(args.root / "REFINEMENT_EP_SHAPES.csv", atlas)

    representative = bench.pivot_table(
        index=["case_id", "dataset", "wave", "layer", "phase", "shape_class", "global_fresh_m"],
        columns="policy", values="comm_semantics_ms"
    ).reset_index()
    tax = representative.merge(
        atlas[["dataset", "wave", "layer", "remote_bytes_one_way_bf16",
               "payload_lower_bound_ms", "o2_physical_lower_ms", "o3_refineep_target_ms"]],
        on=["dataset", "wave", "layer"], how="left"
    )
    tax["cached_handle_upper_bound_ms"] = tax.normal_fresh - tax.normal_cached
    tax["normal_tax_above_payload_ms"] = tax.normal_fresh - tax.payload_lower_bound_ms
    tax["low_latency_tax_above_payload_ms"] = tax.low_latency - tax.payload_lower_bound_ms
    tax["ll_speedup_vs_normal_percent"] = 100 * (tax.normal_fresh - tax.low_latency) / tax.normal_fresh
    tax["refineep_increment_vs_ll_percent"] = 100 * (tax.low_latency - tax.o3_refineep_target_ms) / tax.low_latency
    tax["measurement_scope"] = "real route; communication semantics; no expert GEMM"
    write_csv(args.root / "RUNTIME_TAX.csv", tax)

    oracle_rows = []
    sensitivity_rows = []
    for dataset in ("gsm8k", "humaneval"):
        a = atlas[atlas.dataset == dataset]
        l = layers[layers.dataset == dataset]
        f = frag[frag.dataset == dataset]
        clean = CLEAN_MS[dataset]
        actual_comm = float((l.dispatch_critical_ms + l.combine_critical_ms).sum())
        actual_expert = float(l.expert_critical_ms.sum())
        compact_expert = float(f.compact_predicted_expert_critical_ms_p99_trimmed.sum())
        unchanged = clean - actual_comm - actual_expert + compact_expert
        scenarios = {
            "CURRENT_BEST_STATIC_MEASURED": clean,
            "POST_COMPACTION_NORMAL_PROJECTED": unchanged + a.normal_fresh_ms.sum(),
            "O0_BEST_EXISTING_PATH": unchanged + a.best_existing_ms.sum(),
            "O1_CONTROL_FREE_PAYLOAD_BOUND": unchanged + a.o1_control_free_ms.sum(),
            "O2_PHYSICAL_LOWER_BOUND": unchanged + a.o2_physical_lower_ms.sum(),
            "O3_CREDIBLE_REFINEEP_TARGET": unchanged + a.o3_refineep_target_ms.sum(),
        }
        normal = scenarios["POST_COMPACTION_NORMAL_PROJECTED"]
        o0 = scenarios["O0_BEST_EXISTING_PATH"]
        for name, total in scenarios.items():
            oracle_rows.append({
                "dataset": dataset,
                "scenario": name,
                "request_ms": total,
                "gain_vs_current_clean_percent": 100 * (clean - total) / clean,
                "gain_vs_postcomp_normal_percent": 100 * (normal - total) / normal,
                "incremental_gain_vs_o0_percent": 100 * (o0 - total) / o0,
                "evidence": "measured" if name == "CURRENT_BEST_STATIC_MEASURED" else "analytical_oracle",
                "baseline_scope": "best-static mini32; compacted rows are sensitivity, not measured Epoch",
            })
        for control_us in (25, 50, 75, 100, 125):
            target = np.minimum(
                a.best_existing_ms,
                a.payload_lower_bound_ms + (control_us / 1000.0) * (a.fresh_m > 0),
            )
            total = unchanged + target.sum()
            sensitivity_rows.append({
                "dataset": dataset,
                "refineep_two_direction_control_us": control_us,
                "projected_request_ms": total,
                "incremental_gain_vs_o0_percent": 100 * (o0 - total) / o0,
            })
    oracle = pd.DataFrame(oracle_rows)
    write_csv(args.root / "HEADROOM_ORACLE.csv", oracle)
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_csv(args.root / "CONTROL_FLOOR_SENSITIVITY.csv", sensitivity)

    e2e = oracle[oracle.scenario.isin([
        "CURRENT_BEST_STATIC_MEASURED", "O0_BEST_EXISTING_PATH", "O3_CREDIBLE_REFINEEP_TARGET"
    ])].copy()
    e2e["prototype_status"] = np.where(
        e2e.scenario == "O3_CREDIBLE_REFINEEP_TARGET", "NOT_RUN_GATE_BELOW_12_PERCENT", "not_applicable"
    )
    write_csv(args.root / "E2E_RESULTS.csv", e2e)
    pd.DataFrame([{
        "status": "NOT_RUN_GATE_BELOW_12_PERCENT",
        "contract": "EP4_topk8_hidden4096_BF16_forward",
        "reason": "credible O3 incremental request oracle below CUDA implementation gate",
        "measured_kernel_speedup_percent": np.nan,
        "correctness": "no new kernel exists",
    }]).to_csv(args.root / "REFINEEP_MICROBENCH.csv", index=False)

    figures = args.root / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"gsm8k": "#1976d2", "humaneval": "#ef6c00"}
    for dataset, group in atlas.groupby("dataset"):
        wave = group.groupby("wave").fresh_m.median()
        plt.plot(wave.index, wave.values, label=dataset, color=colors[dataset])
    plt.xlabel("refinement wave"); plt.ylabel("fresh M (median over layers)"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "01_fresh_m_over_refinement.png", dpi=180); plt.close()

    for dataset, group in atlas.groupby("dataset"):
        plt.hist(group.fresh_m, bins=40, alpha=.5, label=dataset, color=colors[dataset])
    plt.xlabel("fresh M per layer-wave"); plt.ylabel("frequency"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "02_fresh_m_histogram.png", dpi=180); plt.close()

    for dataset, group in atlas.groupby("dataset"):
        values = np.sort(group.fresh_m); plt.plot(values, np.linspace(0,1,len(values)), label=dataset)
    plt.xlabel("fresh M"); plt.ylabel("CDF"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "03_fresh_m_cdf.png", dpi=180); plt.close()

    for y, name, ylabel in [
        ("active_experts", "04_active_experts_vs_m.png", "active experts"),
        ("rows_per_active_expert_p50", "05_rows_per_expert_vs_m.png", "p50 rows/active expert"),
        ("destination_fanout_mean", "06_fanout_vs_m.png", "mean destination fanout"),
        ("rank_load_cv", "07_rank_cv_vs_m.png", "rank-load CV"),
    ]:
        for dataset, group in atlas.groupby("dataset"):
            sample = group.sample(min(800, len(group)), random_state=7)
            plt.scatter(sample.fresh_m, sample[y], s=5, alpha=.25, label=dataset)
        plt.xlabel("fresh M"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(figures/name,dpi=180); plt.close()

    for policy, group in bench.groupby("policy"):
        plt.scatter(group.global_fresh_m, group.comm_semantics_ms, label=policy, s=22)
    plt.xscale("log"); plt.xlabel("fresh M"); plt.ylabel("measured communication-semantics ms")
    plt.legend(); plt.tight_layout(); plt.savefig(figures/"08_existing_path_envelope.png",dpi=180); plt.close()

    winner_counts = atlas[atlas.fresh_m>0].groupby(["shape_class","best_existing_path"]).size().unstack(fill_value=0)
    winner_counts.plot(kind="bar", stacked=True); plt.ylabel("layer-wave count"); plt.tight_layout()
    plt.savefig(figures/"09_existing_winner_mass.png",dpi=180); plt.close()

    plt.scatter(tax.global_fresh_m, tax.low_latency_tax_above_payload_ms, label="LL tax above payload")
    plt.scatter(tax.global_fresh_m, tax.normal_tax_above_payload_ms, label="normal tax above payload")
    plt.xscale("log"); plt.xlabel("fresh M"); plt.ylabel("ms"); plt.legend(); plt.tight_layout()
    plt.savefig(figures/"10_runtime_tax_atlas.png",dpi=180); plt.close()

    plt.scatter(atlas.fresh_m.clip(lower=1), atlas.best_existing_ms-atlas.o3_refineep_target_ms, s=4, alpha=.3)
    plt.xscale("log"); plt.xlabel("fresh M"); plt.ylabel("O0 minus credible O3 (ms/layer-wave)"); plt.tight_layout()
    plt.savefig(figures/"11_per_shape_o3_headroom.png",dpi=180); plt.close()

    display = oracle[oracle.scenario.isin(["POST_COMPACTION_NORMAL_PROJECTED","O0_BEST_EXISTING_PATH","O3_CREDIBLE_REFINEEP_TARGET"])]
    display.pivot(index="scenario",columns="dataset",values="request_ms").plot(kind="bar")
    plt.ylabel("projected request ms"); plt.tight_layout(); plt.savefig(figures/"12_request_oracles.png",dpi=180); plt.close()

    gains = oracle[oracle.scenario.isin(["O0_BEST_EXISTING_PATH","O3_CREDIBLE_REFINEEP_TARGET"])]
    gains.pivot(index="scenario",columns="dataset",values="gain_vs_postcomp_normal_percent").plot(kind="bar")
    plt.ylabel("gain vs post-compaction normal (%)"); plt.tight_layout(); plt.savefig(figures/"13_mode_switch_vs_third_path.png",dpi=180); plt.close()

    by_phase = atlas.groupby(["dataset","phase"])[["normal_to_o0_saved_ms","o0_to_o3_saved_ms"]].sum().reset_index()
    by_phase["series"] = by_phase.dataset+"/"+by_phase.phase
    by_phase.set_index("series")[["normal_to_o0_saved_ms","o0_to_o3_saved_ms"]].plot(kind="bar")
    plt.ylabel("cumulative projected saved ms"); plt.tight_layout(); plt.savefig(figures/"14_phase_savings.png",dpi=180); plt.close()

    error = bench.groupby(["shape_class","policy"]).max_relative_l2.max().unstack()
    error.plot(kind="bar"); plt.ylabel("max relative L2 vs normal"); plt.tight_layout()
    plt.savefig(figures/"15_existing_path_correctness.png",dpi=180); plt.close()

    sensitivity.pivot(index="refineep_two_direction_control_us",columns="dataset",values="incremental_gain_vs_o0_percent").plot(marker="o")
    plt.xlabel("credible two-direction control floor (us)"); plt.ylabel("incremental gain vs O0 (%)")
    plt.tight_layout(); plt.savefig(figures/"16_control_floor_sensitivity.png",dpi=180); plt.close()

    mass = atlas.groupby(["dataset","shape_class"]).size().unstack(fill_value=0)
    (100*mass.div(mass.sum(axis=1),axis=0)).T.plot(kind="bar")
    plt.ylabel("layer-wave mass (%)"); plt.tight_layout(); plt.savefig(figures/"17_shape_class_mass.png",dpi=180); plt.close()

    for dataset, group in atlas.groupby("dataset"):
        wave = group.groupby("wave").remote_assignments.median()
        plt.plot(wave.index, wave.values, label=dataset, color=colors[dataset])
    plt.xlabel("refinement wave"); plt.ylabel("fresh remote assignments (layer median)")
    plt.legend(); plt.tight_layout(); plt.savefig(figures/"18_remote_assignments_over_refinement.png",dpi=180); plt.close()

    for policy, column in (("normal_fresh", "normal_fresh"), ("low_latency", "low_latency")):
        plt.scatter(tax.remote_bytes_one_way_bf16, tax[column], s=25, label=policy)
    plt.xscale("symlog", linthresh=65536); plt.xlabel("one-way remote BF16 bytes")
    plt.ylabel("communication-semantics ms"); plt.legend(); plt.tight_layout()
    plt.savefig(figures/"19_existing_latency_vs_bytes.png",dpi=180); plt.close()

    controlled_path = args.root / "CONTROLLED_KERNEL_BENCH.csv"
    if controlled_path.exists():
        controlled = pd.read_csv(controlled_path)
        for policy, group in controlled.groupby("policy"):
            plt.scatter(group.global_fresh_m, group.comm_semantics_ms, s=28, label=policy)
        plt.xscale("log", base=2); plt.xlabel("controlled fresh M")
        plt.ylabel("communication-semantics ms"); plt.legend(); plt.tight_layout()
        plt.savefig(figures/"20_controlled_path_envelope.png",dpi=180); plt.close()

    print(oracle.to_string(index=False))
    print("\nRuntime-tax medians by class:\n", tax.groupby("shape_class")[["normal_fresh","low_latency","payload_lower_bound_ms","o3_refineep_target_ms"]].median())


if __name__ == "__main__":
    main()
