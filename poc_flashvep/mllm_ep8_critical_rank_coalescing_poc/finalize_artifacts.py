#!/usr/bin/env python3
"""Create self-contained manifests, gate summary, and report figures.

This is a documentation/analysis step only.  It never edits routes or model
state; the coalescing numbers remain trace-driven count/cost proxies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_CONFIG = Path(
    "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/"
    "snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c/config.json"
)
RUN_COMMAND = (
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_NO_USAGE_STATS=1 "
    "VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 "
    "NVSHMEM_DIR=/home/esjung/.cache/flashvep-deepep-v020/nvshmem "
    "LD_LIBRARY_PATH=/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-} "
    "PYTHONPATH=/home/esjung/MLLM-EP-github/poc_flashvep/mllm_ep8_critical_rank_coalescing_poc/hooks:"
    "/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/hooks:"
    "/home/esjung/MLLM-EP-github "
    "/home/esjung/.venvs/flashvep-deepep-v020/bin/python "
    "/home/esjung/MLLM-EP-github/poc_flashvep/mllm_ep8_critical_rank_coalescing_poc/"
    "run_multimodal.py --model "
    "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/"
    "snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c "
    "--output <TRACE_DIR> --reps 4"
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def q(s: pd.Series, p: float) -> float:
    return float(s.quantile(p)) if len(s) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    args = ap.parse_args()
    trace = args.trace; analysis = args.analysis
    inv = pd.read_csv(analysis / "invocation_features.csv")
    pairs = pd.read_csv(analysis / "hidden_similarity_pairs.csv")
    coal = pd.read_csv(analysis / "coalescing_results.csv")
    matched_path = analysis / "matched_pair_budget_results.csv"
    matched = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
    prior_summary = json.loads((analysis / "analysis_summary.json").read_text())
    prior_pairs = prior_summary.get("hidden_pair_summary", {})

    cfg = json.loads(MODEL_CONFIG.read_text())
    text_cfg = cfg.get("text_config", {})
    cfg_audit = {
        "checkpoint": str(MODEL_CONFIG.parent),
        "config_sha256": hashlib.sha256(MODEL_CONFIG.read_bytes()).hexdigest(),
        "architecture": cfg.get("architectures"), "model_type": cfg.get("model_type"),
        "text_config": {k: text_cfg.get(k) for k in
                        ("hidden_size", "num_hidden_layers", "num_experts",
                         "num_experts_per_tok", "moe_intermediate_size", "dtype")},
        "derived": {"experts_per_ep8_rank": 16, "placement": "expert_id // 16"},
    }
    for out in (trace, analysis):
        write_json(out / "model_config_audit.json", cfg_audit)
        write_json(out / "placement_map.json", {"ep_size": 8, "experts": 128,
                                                 "experts_per_rank": 16,
                                                 "map": {str(e): e // 16 for e in range(128)}})
        write_json(out / "environment.json", {
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
            "physical_gpu_mapping": [0, 1, 2, 3, 4, 5, 6, 7],
            "topology": "TP2/DP4/EP8/PP1", "dtype": "bfloat16",
            "backend": "deepep_high_throughput", "expert_backend": "TritonExperts",
            "placement": "linear", "eplb": False, "dbo": False,
            "prefix_caching": False, "eager": True,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "run_command": RUN_COMMAND,
        })
        source_audit = Path(__file__).with_name("source_audit.md")
        if source_audit.exists():
            (out / "source_audit.md").write_text(source_audit.read_text())
    (trace / "experiment_command.txt").write_text(
        RUN_COMMAND.replace("<TRACE_DIR>", str(trace)) + "\n"
    )
    (analysis / "analysis_command.txt").write_text(
        "python3 poc_flashvep/mllm_ep8_critical_rank_coalescing_poc/analyze_coalescing.py "
        f"--trace {trace} --output {analysis}\n"
        "python3 poc_flashvep/mllm_ep8_critical_rank_coalescing_poc/finalize_artifacts.py "
        f"--trace {trace} --analysis {analysis}\n"
    )
    (trace / "trace_portability.md").write_text(
        "# EP8 trace portability\n\n"
        "The route capture is self-contained for offline analysis: each measured "
        "wave/layer/TP shard stores token positions, token IDs, Vision/Text label, "
        "top-k logical expert IDs, router weights, destination EP rank, and sampled "
        "FP16 hidden vectors (layers 16/24/40). Per-rank DeepEP/Triton timing, model "
        "config, placement map, workload manifest, and environment are included.\n\n"
        "The mapping is EP8-specific. Reuse on four GPUs requires an explicit remap "
        "and must not be presented as an EP8 rerun. All coalescing values are "
        "trace-driven oracle/count-cost estimates; no route or model execution was "
        "changed and no EP8 expert-output equivalence was captured.\n"
    )

    strat = {}
    for b, g in coal.groupby(["budget", "strategy"]):
        strat[f"{b[0]:.2f}_{b[1]}"] = {
            "actual_removed_fraction_median": float(g.actual_removed_fraction.median()),
            "max_rank_load_reduction_median": float(g.max_rank_load_reduction.median()),
            "critical_rank_load_reduction_median": float(g.critical_rank_load_reduction.median()),
        }
    matched_summary = {}
    if not matched.empty:
        for (fraction, strategy), g in matched.groupby(["pair_budget_fraction", "strategy"]):
            matched_summary[f"{fraction:.2f}_{strategy}"] = {
                "max_rank_load_reduction_median": float(g.max_rank_load_reduction.median()),
                "actual_removed_fraction_median": float(g.actual_removed_fraction.median()),
                "critical_pair_fraction_median": float(g.critical_pair_fraction.median()),
            }
    ratios = inv.expert_max_mean_ratio.dropna()
    gate = {
        "final_status": "NO_GO",
        "decision_basis": "quality-preserving critical-rank coalescing headroom was not demonstrated",
        "model": "Qwen3-VL-30B-A3B-Instruct",
        "topology": "TP2/DP4/EP8/PP1", "experts_per_gpu": 16,
        "trace": {"route_files": 6912, "invocations": int(len(inv)),
                  "timing_rows": 9216, "measured_repetitions": 3,
                  "request_count": int(inv.request_id.nunique()), "layers": 48},
        "stage_b_straggler_current_multimodal": {
            "expert_cuda_max_mean": {"median": float(ratios.median()),
                                     "p90": q(ratios, .9), "max": float(ratios.max())},
            "rank_assignment_max_mean": {"median": float(inv.rank_ratio.median()),
                                          "p90": q(inv.rank_ratio, .9),
                                          "max": float(inv.rank_ratio.max())},
            "interpretation": "rank pressure is visible, but expert CUDA imbalance is mostly mild in this bounded image workload",
        },
        "critical_excess": {"vision_median": float(inv.vision_critical_excess.median()),
                            "vision_mean": float(inv.vision_critical_excess.mean()),
                            "text_median": float(inv.text_critical_excess.median()),
                            "text_mean": float(inv.text_critical_excess.mean())},
        "hidden_similarity": {"sample_assignments": int(prior_pairs.get("hidden_sample_assignments", 0)),
                              "pairs_cosine_ge_0_90": int(len(pairs)),
                              "pair_fraction": float(prior_pairs.get("pair_fraction_cosine_ge_0_90", 0.0)),
                              "median_cosine": float(pairs.cosine.median()) if len(pairs) else None,
                              "critical_pair_fraction": float(pairs.critical_pair.mean()) if len(pairs) else None,
                              "expert_output_similarity": "NOT_CAPTURED_EP8"},
        "requested_assignment_budget_results": strat,
        "matched_available_pair_results": matched_summary,
        "quality_preserving_headroom": "NOT_DEMONSTRATED; sampled-layer max-rank proxy median 0.47%, max targeted-layer median 1.00%",
        "critical_rank_advantage": "small absolute advantage at matched pair budgets, not a >=3% latency headroom result",
        "trace_reusable_on_4gpu": True,
        "measurement_mode_caveat": "same real request submitted to all four DP engines so multimodal EP collectives participate; canonical route metrics use DP0, timing includes four replicated copies",
    }
    write_json(analysis / "gate_summary.json", gate)
    write_json(trace / "gate_summary.json", gate)

    # Additional figures make the positive and negative evidence easy to audit.
    if not inv.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        order = inv.groupby("request_id").vision_critical_excess.median().sort_values().index
        inv.assign(request_id=pd.Categorical(inv.request_id, categories=order, ordered=True)).boxplot(
            column="vision_critical_excess", by="request_id", ax=ax, grid=False, rot=25)
        ax.set_title("Vision critical-rank excess by request (EP8 route)"); ax.set_xlabel(""); ax.set_ylabel("assignments above mean")
        fig.suptitle(""); fig.tight_layout(); fig.savefig(analysis / "critical_excess_by_request.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(inv.expert_max_mean_ratio, bins=30, alpha=.8)
        ax.axvline(1.15, color="tab:orange", ls="--", label="1.15 gate"); ax.axvline(1.25, color="tab:red", ls="--", label="1.25 gate")
        ax.set(xlabel="expert CUDA max/mean", ylabel="invocations", title="EP8 expert CUDA imbalance"); ax.legend(); fig.tight_layout(); fig.savefig(analysis / "expert_cuda_ratio_distribution.png", dpi=160); plt.close(fig)
    if not pairs.empty:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(pairs.cosine, bins=25, alpha=.8)
        ax.set(xlabel="same-expert hidden cosine (sampled pairs)", ylabel="pairs", title="Candidate redundancy similarity")
        fig.tight_layout(); fig.savefig(analysis / "similarity_distribution.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
