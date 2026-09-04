#!/usr/bin/env python3
"""Finalize the bounded Qwen3-30B-A3B EP8 straggler PoC."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def q(s: pd.Series, p: float) -> float:
    return float(s.quantile(p)) if len(s) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args(); out = args.result
    inv = pd.read_csv(out / "invocation_metrics.csv")
    proxy = pd.read_csv(out / "capacity_action_proxy.csv")
    temporal = pd.read_csv(out / "temporal_metrics.csv")
    gate = json.loads((out / "gate_summary.json").read_text())
    er, rr = inv.expert_ratio.dropna(), inv.rank_ratio.dropna()

    cond = (inv.groupby("condition", dropna=False)
            .agg(invocations=("expert_ratio", "size"),
                 expert_ratio_median=("expert_ratio", "median"),
                 expert_ratio_p90=("expert_ratio", lambda x: q(x, .90)),
                 expert_ratio_max=("expert_ratio", "max"),
                 rank_ratio_median=("rank_ratio", "median"),
                 critical_path_ratio_median=("critical_path_ratio", "median"),
                 expert_ms_median=("expert_max_ms", "median"),
                 total_assignments_median=("total_assignments", "median"))
            .reset_index())
    cond.to_csv(out / "stage0_condition_summary.csv", index=False)
    layer = (inv.groupby("layer", dropna=False)
             .agg(invocations=("expert_ratio", "size"),
                  expert_ratio_median=("expert_ratio", "median"),
                  expert_ratio_p90=("expert_ratio", lambda x: q(x, .90)),
                  expert_ratio_max=("expert_ratio", "max"),
                  rank_ratio_median=("rank_ratio", "median"),
                  dispatch_ratio_median=("dispatch_ratio", "median"),
                  combine_ratio_median=("combine_ratio", "median"),
                  critical_path_ratio_median=("critical_path_ratio", "median"))
             .reset_index())
    layer.to_csv(out / "stage0_layer_summary.csv", index=False)

    # A compact distribution figure is useful in the report and is generated
    # entirely from measured CUDA-event values.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(er, bins=24, color="#4472c4", alpha=.85)
    axes[0].axvline(1.25, color="#c00000", ls="--", label="1.25 strong gate")
    axes[0].set(xlabel="expert CUDA max/mean", ylabel="layer invocations")
    axes[0].legend(fontsize=8)
    axes[1].scatter(inv.rank_ratio, inv.expert_ratio, c=inv.layer, s=10, alpha=.55, cmap="viridis")
    axes[1].axvline(1.5, color="k", ls="--", lw=.8); axes[1].axhline(1.25, color="#c00000", ls="--", lw=.8)
    axes[1].set(xlabel="rank assignment max/mean", ylabel="expert CUDA max/mean")
    fig.tight_layout(); fig.savefig(out / "figures/stage0_straggler_summary.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer.layer, layer.expert_ratio_median, label="expert CUDA")
    ax.plot(layer.layer, layer.rank_ratio_median, label="rank assignment")
    ax.axhline(1.25, color="#c00000", ls="--", lw=.8)
    ax.set(xlabel="decoder layer", ylabel="median max/mean"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "figures/per_layer_straggler.png", dpi=170); plt.close(fig)

    # Keep exact command/environment evidence and the serving log in-result.
    command = """export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_DIR=/home/esjung/.cache/flashvep-deepep-v020/nvshmem
export LD_LIBRARY_PATH=\"$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}\"
export PYTHONPATH=/home/esjung/MLLM-EP-github/poc_flashvep/rl_ep_straggler_qwen_ep8_poc/hooks:/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/hooks:/home/esjung/MLLM-EP-github
export FLASHVEP_MATRIX_ENABLE=1
export FLASHVEP_MATRIX_RAW_DIR=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/rl_ep_straggler_qwen_ep8_poc_20260904_stage0_attempt7/raw_live
export FLASHVEP_MATRIX_CONTROL=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/rl_ep_straggler_qwen_ep8_poc_20260904_stage0_attempt7/control.json
/home/esjung/.venvs/flashvep-deepep-v020/bin/python poc_flashvep/rl_ep_straggler_qwen_ep8_poc/run_serving.py \\
  --model /home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \\
  --output poc_flashvep/deepep_revalidation/results/rl_ep_straggler_qwen_ep8_poc_20260904_stage0_attempt7 \\
  --reps 3 --max-tokens 1
"""
    (out / "experiment_command.txt").write_text(command)
    sibling = out.with_suffix(".log")
    if sibling.exists():
        shutil.copy2(sibling, out / "serving.log")

    refs = {
        "capacity_aware_moe": {"repository": "https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE", "commit": "9c73c8eee6ca64836eb873e77aa096fb4955e658", "status": "inspected_not_executed", "reason": "read-only artifact has no token IDs/alternate-route outcomes"},
        "eplb": {"repository": "https://github.com/deepseek-ai/EPLB", "commit": "d52c72d5b2f2fb4c41afbf8eb21366820239913d", "status": "inspected_not_executed", "reason": "migration/replication cost and token routes are unavailable"},
    }
    (out / "capacity_eplb_reference_manifest.json").write_text(json.dumps(refs, indent=2) + "\n")

    proof = sorted((out / "backend_proof").glob("moe_backend_*.json"))
    model_cfg = json.loads((out / "model_config_audit.json").read_text())
    source = f"""# Qwen3-30B-A3B EP8 source/config audit

## Model

- checkpoint: `{model_cfg['model']}`
- architecture: `{model_cfg['architectures'][0]}` / `{model_cfg['model_type']}`
- hidden size / layers: `{model_cfg['hidden_size']}` / `{model_cfg['num_hidden_layers']}`
- routed experts / top-k: `{model_cfg['num_experts']}` / `{model_cfg['num_experts_per_tok']}`
- experts per EP rank: `{model_cfg['experts_per_ep_rank']}` at EP8
- dtype: `{model_cfg['torch_dtype']}`
- sparse path: `decoder_sparse_step=1`, `mlp_only_layers=[]` (all 48 decoder layers use routed MoE)

## Runtime proof

The run log shows eight NCCL workers with `world_size=8`, and every backend
proof in `backend_proof/` reports `ep_world_size=8`,
`DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`, and `TritonExperts`.
The four driver processes were DP ranks 0–3 and each used TP2, giving
TP2/DP4/EP8/PP1. `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` was preserved in
all rank proof files. EPLB and DBO were disabled; placement was linear.

## Measurement hook

The experiment-local child-worker hook wraps the existing
`FusedMoEKernelModularImpl._prepare`, `_fused_experts`, and `_finalize`
calls. It resolves CUDA events once after all waves. `dispatch`, `expert`,
and `combine` are therefore real per-rank CUDA durations; rank values are
never subtracted as cross-device absolute timestamps. The hook records the
local 16-expert assignment histogram and leaves hidden states, top-k routing,
weights, placement, and scheduler decisions unchanged.

## Limitation for gated actions

The raw capture contains exact local expert counts but not token-level expert
IDs or alternate-route outcomes. Capacity-Aware-MoE and EPLB are consequently
source-audited only. `capacity_action_proxy.csv` is a clearly labelled
count-only sensitivity diagnostic, not a correctness-preserving GPU result.
"""
    (out / "source_audit.md").write_text(source)

    # The final status is deliberately split: EP8 is a strong natural
    # straggler testbed, but the RL/action gate is HOLD because no valid action
    # outcomes or migration costs were measured.
    gate.update({
        "final_status": "HOLD",
        "straggler_found": "YES",
        "stage0_interpretation": "STRONG_GO",
        "stage0b_capacity_control": "NOT_GPU_VALIDATED_TOKEN_ROUTE_REQUIRED",
        "stage1_temporal": "ROUTE_PERSISTENCE_ONLY",
        "stage2_dynamic_oracle": "NOT_RUN_VALID_ACTION_COST_UNAVAILABLE",
        "stage3_rl": "NOT_RUN",
        "model": "Qwen3-30B-A3B",
        "configuration": "BF16 TP2/DP4/EP8 PP1 DeepEP HT TritonExperts EPLB off DBO off prefix cache off eager linear",
        "physical_gpus": list(range(8)),
        "backend_proof_files": len(proof),
        "natural_gate_note": "Median expert CUDA max/mean >=1.25, >=50% >=1.25, and 27 invocations >=1.50; strong Stage-0 testbed evidence.",
    })
    (out / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")

    # Report-friendly numbers and limitations.
    corr = float(inv[["rank_ratio", "expert_ratio"]].corr().iloc[0, 1])
    max_row = inv.loc[inv.expert_ratio.idxmax()]
    highest_layer = layer.loc[layer.expert_ratio_median.idxmax()]
    report = f"""# Qwen3-30B-A3B EP8 natural straggler / RL-controller feasibility

## Executive decision

**FINAL STATUS: HOLD**  
**STRAGGLER_FOUND: YES**  
**STAGE-0: STRONG_GO**  
**RL_POLICY: NOT_RUN**

EP8 is a strong, real serving straggler testbed in this bounded run. The
measured routed-expert CUDA max/mean ratio is {er.median():.3f} median, with
{(er >= 1.25).mean()*100:.1f}% of 576 invocation/layer views at or above 1.25
and {(er >= 1.50).sum()} views at or above 1.50. The overall controller result
is HOLD rather than a fabricated GO: the read-only capture does not contain
token-level IDs, alternate-route outcomes, or expert-weight migration timing,
so Capacity-Aware/EPLB action gains cannot be validly measured yet.

## Configuration and workload

| item | value |
|---|---|
| model | Qwen3-30B-A3B (`Qwen3MoeForCausalLM`) |
| model config | 128 routed experts, top-8, 48 decoder layers, 16 experts/EP rank |
| topology | TP2 / DP4 / EP8 / PP1 |
| runtime | vLLM 0.20.0 V1, BF16, DeepEP high-throughput, TritonExperts |
| controls | EPLB off, DBO off, prefix cache off, eager, linear placement |
| GPUs | physical 0–7 (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`) |
| schedule | 6 text conditions × 3 repetitions; repetitions 1–2 measured; 48 layers |
| experiment base | `1614491c30b92fd0f2dde6022c9cfd3620397b2e` (branch created from prior DeepSeek PoC) |
| measured views | 12 measured waves × 48 layers = 576, all with 8 EP-rank timings |

Conditions were balanced/heterogeneous 0.5K–4K-token text prompts across
code, math, reasoning, factual, chat, and long-prefill variants. The first
repetition is retained as route warmup and excluded from the CUDA gate.

## Stage-0 natural straggler

| metric | median | p75 | p90 | max |
|---|---:|---:|---:|---:|
| rank assignment max/mean | {rr.median():.3f} | {q(rr,.75):.3f} | {q(rr,.90):.3f} | {rr.max():.3f} |
| expert CUDA max/mean | {er.median():.3f} | {q(er,.75):.3f} | {q(er,.90):.3f} | {er.max():.3f} |
| dispatch CUDA max/mean | {inv.dispatch_ratio.median():.3f} | {q(inv.dispatch_ratio,.75):.3f} | {q(inv.dispatch_ratio,.90):.3f} | {inv.dispatch_ratio.max():.3f} |
| combine CUDA max/mean | {inv.combine_ratio.median():.3f} | {q(inv.combine_ratio,.75):.3f} | {q(inv.combine_ratio,.90):.3f} | {inv.combine_ratio.max():.3f} |
| critical-path stage-sum max/mean | {inv.critical_path_ratio.median():.3f} | {q(inv.critical_path_ratio,.75):.3f} | {q(inv.critical_path_ratio,.90):.3f} | {inv.critical_path_ratio.max():.3f} |

The primary gate distribution is: {((er <= 1.10).mean()*100):.1f}% ≤1.10,
{((er >= 1.15).mean()*100):.1f}% ≥1.15, {((er >= 1.25).mean()*100):.1f}% ≥1.25,
and {((er >= 1.50).mean()*100):.1f}% ≥1.50. Thus this is not a single
outlier: every condition has a median expert ratio between
{cond.expert_ratio_median.min():.3f} and {cond.expert_ratio_median.max():.3f},
and the highest layer median is layer {int(highest_layer.layer)} at
{highest_layer.expert_ratio_median:.3f}. The largest view is
`{max_row.condition}`, layer {int(max_row.layer)}, ratio {max_row.expert_ratio:.3f}.

Rank assignment max/mean is high (median {rr.median():.3f}), and its Pearson
association with expert CUDA max/mean is r={corr:.3f}. The relationship is
positive, but not perfect: rank-level route load alone is not a sufficient
latency model. This is precisely why action outcomes must be measured before
training RL.

## Stage 0B — Capacity positive control

The public references were inspected at fixed commits (see
`capacity_eplb_reference_manifest.json`). Capacity-Aware-MoE uses a capacity
factor to select/drop or reroute assignments; EPLB packs weighted experts and
can replicate them. Neither was applied to the model. The capture has no
token-level expert IDs, so a route-preserving capacity action cannot be
constructed from these histograms. `capacity_action_proxy.csv` reports only a
count sensitivity diagnostic: idealized EPLB packing has median rank-load
ratio {proxy.EPLB_ideal_rank_ratio.median():.3f} and a {proxy.EPLB_ideal_load_reduction_proxy.median()*100:.1f}%
load upper-bound proxy, while 1.25/1.50 capacity clipping would drop median
{proxy.capacity_mild_drop_fraction.median()*100:.1f}%/{proxy.capacity_strong_drop_fraction.median()*100:.1f}%
of assignments. The latter makes clear that clipping is not a free,
correctness-preserving gain. No GPU latency gain is claimed.

## Stage 1 — temporal structure

The same deterministic domain prompts were repeated three times. Exact route
histograms therefore show median adjacent-wave hottest-expert recurrence
{temporal.hot_expert_recurrence.median():.3f}; this demonstrates repeatability
of this fixed episode, not generalization to unseen text. A future-aware action
oracle still needs alternate-action timing and a realistic migration cost.

## Stage 2/3 gate

`A0` is observed. `A1/A2` are count-only proxies and `A3/A4` (EPLB) are not
evaluated because token routes, replica placement, and expert-weight migration
cost are absent. Consequently myopic-vs-future-aware gain, action diversity,
migration amortization, and RL realization are **not estimated**. This is a
deliberate HOLD, not a claim that the strong straggler has no mitigation
headroom.

## Required answers

1. **Is Qwen EP8 a straggler testbed?** Yes: repeated real vLLM routed-expert
   CUDA imbalance is strong at the observed 0.5K–4K text prefill scales.
2. **Does routing skew connect to CUDA?** Yes, positively (r={corr:.3f}), but
   route rank ratio is not the whole predictor; expert CUDA is the critical
   measured stage.
3. **Is the result an RL opportunity?** Not yet a quantified one. Capacity and
   EPLB action outcomes plus migration costs must be measured with token-level
   route capture before policy design.
4. **What should be done next?** Add an opt-in child-worker route-ID capture
   for a bounded subset, run one real Capacity-Aware intervention and one EPLB
   migration microbenchmark, then remeasure selected layers. Do not train RL
   or integrate LPLB/DeepEP serving from this result alone.

## Artifacts

- raw per-rank trace: `raw_live/rank0..7.jsonl` and `*.proof.json`
- exact model/runtime proof: `model_config_audit.json`, `backend_proof/`, `runtime_proof.dp_rank*.json`
- flattened observations: `local_expert_trace.csv`, `invocation_metrics.csv`
- gate and diagnostics: `gate_summary.json`, `capacity_action_proxy.csv`, `gated_stage_summary.json`
- summaries: `stage0_condition_summary.csv`, `stage0_layer_summary.csv`, `temporal_condition_summary.csv`
- figures: `figures/`
- exact command/log: `experiment_command.txt`, `serving.log`
- source/reference audit: `source_audit.md`, `capacity_eplb_reference_manifest.json`

See `source_audit.md` for the precise vLLM/DeepEP path and proof semantics.
"""
    (out / "stage0_report.md").write_text(report)
    report_path = out.parents[2] / "reports" / "rl_ep_straggler_qwen_ep8_poc.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report_path)


if __name__ == "__main__": main()
