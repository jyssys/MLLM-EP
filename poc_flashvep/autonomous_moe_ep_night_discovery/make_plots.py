"""Small, reproducible plots for the autonomous night-campaign report."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eff = pd.read_csv(args.analysis / "hypothesis_effects.csv")
    logical = pd.read_csv(args.analysis / "logical_step_summary.csv")

    if not logical.empty:
        g = logical.groupby(["hypothesis", "phase"], as_index=False)["step_cuda_ms"].median()
        fig, ax = plt.subplots(figsize=(10, 4))
        for phase, d in g.groupby("phase"):
            ax.plot(d.hypothesis, d.step_cuda_ms, marker="o", label=phase)
        ax.set_ylabel("median rank-critical step CUDA (ms)")
        ax.set_title("Live phase/step medians")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "phase_vs_hypothesis.png", dpi=150)
        plt.close(fig)

    if not eff.empty:
        cols = ["wave_e2e_p50_ms_B_vs_A_pct", "request_e2e_p50_ms_B_vs_A_pct"]
        d = eff.melt(["hypothesis", "experiment"], value_vars=cols,
                     var_name="metric", value_name="B_vs_A_pct")
        fig, ax = plt.subplots(figsize=(10, 4))
        for metric, x in d.groupby("metric"):
            ax.plot(x.hypothesis, x.B_vs_A_pct, marker="o", label=metric.replace("_B_vs_A_pct", ""))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("B vs A (%)")
        ax.set_title("Wave versus request-level paired effects")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "wave_vs_request.png", dpi=150)
        plt.close(fig)

        stage = ["moe_sample_ms_p50_B_vs_A_pct", "attention_sample_ms_p50_B_vs_A_pct",
                 "dispatch_sample_ms_p50_B_vs_A_pct", "combine_sample_ms_p50_B_vs_A_pct"]
        d = eff[["hypothesis"] + stage].set_index("hypothesis")
        fig, ax = plt.subplots(figsize=(10, 5))
        d.rename(columns=lambda x: x.replace("_p50_B_vs_A_pct", "")).plot.bar(ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("B vs A (%)")
        ax.set_title("Sampled stage breakdown")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(args.out / "stage_breakdown.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
