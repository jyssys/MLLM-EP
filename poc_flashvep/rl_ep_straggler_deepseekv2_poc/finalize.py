#!/usr/bin/env python3
"""Finalize the preregistered Stage-0 natural-straggler PoC.

This deliberately stops after a failed Stage-0 gate.  It does not fabricate a
capacity-control result: the user protocol says to terminate immediately when
natural expert CUDA imbalance is not present.
"""
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


def _q(x: pd.Series, p: float) -> float:
    return float(x.quantile(p)) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    out = args.result
    inv = pd.read_csv(out / "invocation_metrics.csv")
    gate = json.loads((out / "gate_summary.json").read_text())

    # Paired summaries are useful even though the natural gate is NO-GO.
    cond = (inv.groupby("condition", dropna=False)
            .agg(invocations=("expert_ratio", "size"),
                 expert_ratio_median=("expert_ratio", "median"),
                 expert_ratio_p90=("expert_ratio", lambda x: _q(x, .90)),
                 expert_ratio_max=("expert_ratio", "max"),
                 rank_ratio_median=("rank_ratio", "median"),
                 dispatch_ratio_median=("dispatch_ratio", "median"),
                 combine_ratio_median=("combine_ratio", "median"),
                 critical_moe_ms_median=("critical_moe_ms", "median"))
            .reset_index())
    cond.to_csv(out / "stage0_condition_summary.csv", index=False)
    layer = (inv.groupby("layer", dropna=False)
             .agg(invocations=("expert_ratio", "size"),
                  expert_ratio_median=("expert_ratio", "median"),
                  expert_ratio_p90=("expert_ratio", lambda x: _q(x, .90)),
                  expert_ratio_max=("expert_ratio", "max"),
                  rank_ratio_median=("rank_ratio", "median"),
                  critical_moe_ms_median=("critical_moe_ms", "median"))
             .reset_index())
    layer.to_csv(out / "stage0_layer_summary.csv", index=False)

    # Distribution figure is independent of the analyzer's ratio scatter.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(inv.expert_ratio.dropna(), bins=24, color="#4472c4", alpha=.85)
    ax.axvline(1.10, color="black", ls="--", label="1.10 noise gate")
    ax.axvline(1.15, color="#c00000", ls="--", label="1.15 meaningful gate")
    ax.set(xlabel="expert CUDA max/mean", ylabel="layer invocations",
           title="DeepSeek-V2-Lite natural routed-MoE imbalance")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "figures/expert_ratio_distribution.png", dpi=160)
    plt.close(fig)

    # Keep the exact run command alongside the raw trace.
    command = """CUDA_VISIBLE_DEVICES=1,2,3,4 \\
PYTHONPATH=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/hooks:/home/esjung/MLLM-EP-github \\
PATH=/home/esjung/anaconda3/envs/flashvep-poc/bin:/home/esjung/.venvs/flashvep-deepep-v020/bin:$PATH \\
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m \\
poc_flashvep.rl_ep_straggler_deepseekv2_poc.run_serving \\
--model /home/esjung/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7 \\
--output poc_flashvep/deepep_revalidation/results/rl_ep_straggler_deepseekv2_poc_20260904_stage0_v6 \\
--reps 3 --max-tokens 1
"""
    (out / "experiment_command.txt").write_text(command)
    sibling_log = out.with_suffix(".log")
    if sibling_log.exists():
        shutil.copy2(sibling_log, out / "serving.log")

    # The references are inspected for provenance, but no unvalidated route
    # mutation is run after the preregistered NO-GO gate.
    refs = {
        "capacity_aware_moe": {
            "repository": "https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE",
            "commit": "9c73c8eee6ca64836eb873e77aa096fb4955e658",
            "inspected_path": "capacity_aware/capacity_patch.py",
            "mechanism": "capacity-factor token selection by score/device; may drop or reroute selected assignments",
        },
        "eplb": {
            "repository": "https://github.com/deepseek-ai/EPLB",
            "commit": "d52c72d5b2f2fb4c41afbf8eb21366820239913d",
            "inspected_path": "eplb.py",
            "mechanism": "weighted expert packing and optional replication; placement-level action",
        },
        "execution_status": "NOT_RUN_STAGE0_NO_GO",
        "reason": "User gate requires immediate termination when most actual expert CUDA ratios are <=1.10; no positive-control data are claimed.",
    }
    (out / "capacity_eplb_reference_manifest.json").write_text(json.dumps(refs, indent=2) + "\n")

    model_cfg = json.loads((out / "model_config_audit.json").read_text())
    proof = sorted((out / "backend_proof").glob("moe_backend_*.json"))
    vllm_version = "0.20.0 (run log / installed package)"
    source_audit = f"""# DeepSeek-V2-Lite source/config audit

## Model configuration

- model type: `{model_cfg['model_type']}` (`DeepseekV2ForCausalLM`)
- hidden size: `{model_cfg['hidden_size']}`
- decoder layers: `{model_cfg['num_hidden_layers']}`
- routed experts: `{model_cfg['n_routed_experts']}`
- top-k per token: `{model_cfg['num_experts_per_tok']}`
- shared experts: `{model_cfg['n_shared_experts']}`
- MoE frequency: every layer (`moe_layer_freq={model_cfg['moe_layer_freq']}`)
- first dense replacement: layer 0 (`first_k_dense_replace={model_cfg['first_k_dense_replace']}`); measured routed layers 1–26
- dtype: `{model_cfg['torch_dtype']}`

## vLLM path inspected

Installed vLLM `{vllm_version}` source at
`/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm`.
`DeepseekV2DecoderLayer.forward` (deepseek_v2.py:1043+) executes self-attention,
post-attention RMSNorm, then `self.mlp`; routed layers construct
`DeepseekV2MoE`, whose `FusedMoE` uses `top_k=config.num_experts_per_tok`.
The experiment-local hook wraps the existing modular `_prepare`,
`_fused_experts`, and `_finalize` calls with CUDA events and reads
`expert_num_tokens_cpu`; it does not alter routes, weights, placement, or
scheduler behavior.

## Backend proof

All four EP rank proof files report `DeepEPHTPrepareAndFinalize`,
`DeepEPHTAll2AllManager`, `TritonExperts`, `ep_world_size=4`, and
`visible_devices=1,2,3,4`. The runtime metadata records TP2/DP2/EP4,
BF16, DBO off, prefix cache off, and linear placement.

## Capacity/EPLB references

Public references were inspected at the commits in
`capacity_eplb_reference_manifest.json`. Capacity-Aware-MoE's patch applies
capacity-factor token selection; EPLB's `rebalance_experts` packs weighted
experts and can replicate them. Neither was executed because the
preregistered Stage-0 natural-straggler gate failed.
"""
    (out / "source_audit.md").write_text(source_audit)

    gate.update({
        "final_verdict": "NO_GO",
        "stop_after_stage0": True,
        "stage0_gate_interpretation": "natural routed-expert CUDA imbalance is not repeated; most invocations are <=1.10",
        "stage0b_capacity_control": "NOT_RUN_EARLY_STOP",
        "stage1_temporal": "NOT_RUN_EARLY_STOP",
        "stage2_dynamic_oracle": "NOT_RUN_EARLY_STOP",
        "stage3_rl": "NOT_RUN_EARLY_STOP",
        "n_backend_proof_files": len(proof),
        "model": "DeepSeek-V2-Lite-Chat",
        "configuration": "BF16 TP2/DP2/EP4 PP1 DeepEP HT TritonExperts DBO off",
        "physical_gpus": [1, 2, 3, 4],
    })
    (out / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")

    er = inv.expert_ratio.dropna(); rr = inv.rank_ratio.dropna()
    max_case = inv.loc[inv.expert_ratio.idxmax()]
    report = f"""# RL EP straggler feasibility — DeepSeek-V2-Lite

## Decision

**FINAL STATUS: NO_GO**  
**NATURAL_STRAGGLER: NO**  
**RL_POLICY: NOT_RUN (preregistered early stop)**

The real text-only vLLM run does not provide the required natural routed-MoE
CUDA straggler. The strict Stage-0 gate was fixed before this run: PASS
requires median expert CUDA max/mean ≥1.15 and at least 50% of measured
prefill layer invocations ≥1.15. Stage 0 failed, so Stage 0B, temporal
episodes, action oracles, migration microbenchmarks, and RL training were not
run. This preserves the requested early-stop rule and avoids claiming a
capacity method result without a qualifying workload.

## Model and execution

| item | value |
|---|---|
| model | DeepSeek-V2-Lite-Chat (`DeepseekV2ForCausalLM`) |
| model config | 64 routed experts, top-6, 2 shared experts, 27 decoder layers |
| measured routed layers | 1–26 (layer 0 is dense under `first_k_dense_replace=1`) |
| precision / parallelism | BF16, TP2 / DP2 / EP4 / PP1 |
| backend | DeepEP high-throughput + TritonExperts |
| placement / controls | linear, DBO off, prefix cache off, eager |
| visible GPUs | `CUDA_VISIBLE_DEVICES=1,2,3,4` |
| measured workload | 6 real text domain pairs × 2 measured repetitions; 12 waves, max 1 decode token |
| measured invocations | {len(inv)} layer-level prefill views (4 EP ranks per invocation) |

The largest prefill invocation per rank was selected when duplicate engine
calls (chunk/profile/decode) existed. This keeps the primary comparison on
the real routed prefill while retaining raw JSONL rows for audit.

## Stage-0 natural straggler metrics

| metric | median | p25 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| rank assignment max/mean | {rr.median():.3f} | {_q(rr,.25):.3f} | {_q(rr,.75):.3f} | {_q(rr,.90):.3f} | {rr.max():.3f} |
| expert CUDA max/mean | {er.median():.3f} | {_q(er,.25):.3f} | {_q(er,.75):.3f} | {_q(er,.90):.3f} | {er.max():.3f} |
| dispatch CUDA max/mean | {inv.dispatch_ratio.median():.3f} | {_q(inv.dispatch_ratio,.25):.3f} | {_q(inv.dispatch_ratio,.75):.3f} | {_q(inv.dispatch_ratio,.90):.3f} | {inv.dispatch_ratio.max():.3f} |
| combine CUDA max/mean | {inv.combine_ratio.median():.3f} | {_q(inv.combine_ratio,.25):.3f} | {_q(inv.combine_ratio,.75):.3f} | {_q(inv.combine_ratio,.90):.3f} | {inv.combine_ratio.max():.3f} |

Expert CUDA ratio distribution: **{float((er <= 1.10).mean())*100:.1f}% ≤1.10**, **{float((er >= 1.15).mean())*100:.1f}% ≥1.15**, **{float((er >= 1.25).mean())*100:.1f}% ≥1.25**, and **{float((er >= 1.50).mean())*100:.1f}% ≥1.50**. The maximum 1.583 case was an isolated layer invocation (`condition={max_case.condition}`, layer {int(max_case.layer)}), not a repeated heavy regime.

Per-condition medians span `{cond.expert_ratio_median.min():.3f}`–`{cond.expert_ratio_median.max():.3f}`; only the `code_vs_math` pair crosses 1.15 at the median ({cond.loc[cond.expert_ratio_median.idxmax(),'expert_ratio_median']:.3f}). Per-layer medians are mostly near 1.02–1.11; the highest is layer {int(layer.loc[layer.expert_ratio_median.idxmax(),'layer'])} at {layer.expert_ratio_median.max():.3f}. Thus routing-count imbalance (rank median {rr.median():.3f}) does not translate into a repeated expert-kernel critical-path imbalance.

The event timing confirms the important distinction: expert CUDA max/mean has only a weak relationship to rank assignment ratio (Pearson r = {inv[['rank_ratio','expert_ratio']].corr().iloc[0,1]:.3f}), while expert ratio tracks the combined critical MoE timing (r = {inv[['expert_ratio','critical_moe_ms']].corr().iloc[0,1]:.3f}). The latter is a timing sanity check, not evidence of a natural straggler.

## Temporal and action stages

Not run by design. Stage 0 failed the preregistered natural-straggler gate,
so there is no qualifying heavy/transient hotspot on which to evaluate
CAPACITY_MILD/STRONG, EPLB_SMALL/LARGE, migration cost, myopic versus
future-aware oracle, or a learned policy. `capacity_eplb_reference_manifest.json`
records the inspected public references and their exact commits; it does not
claim an unmeasured gain.

## Strongest positive and counter-evidence

- **Positive:** assignment spread exists (rank max/mean median {rr.median():.3f}, p90 {_q(rr,.90):.3f}), and isolated expert-timing outliers reach {er.max():.3f}.
- **Counter-evidence:** the primary CUDA metric is median {er.median():.3f}; {float((er <= 1.10).mean())*100:.1f}% of invocations are at or below 1.10 and only {float((er >= 1.15).mean())*100:.1f}% reach 1.15. Heavy ≥1.50 appears in {float((er >= 1.50).sum()):.0f}/{len(er)} invocation(s), so no robust natural straggler is present.
- The v5 one-repetition pilot had a higher median (1.186), but the preregistered repeated v6 run (two measured repeats per condition, 312 layer views) does not reproduce it; the pilot is retained only as exploratory evidence and is not used for the gate.

## Interpretation and next action

**SEQUENTIAL_EFFECT: REJECTED/UNTESTED** — temporal persistence was not
assessed after early stop; the qualifying prerequisite did not exist.  
**DYNAMIC_ORACLE_HEADROOM: NOT_ESTIMATED.**  
**RL METHOD DESIGN: NO.**

Within this DeepSeek-V2-Lite EP4 + vLLM configuration and tested real text
workload, a future-aware RL controller has no demonstrated straggler headroom.
The next single action, if this direction is revisited, is to collect a
separate much larger/longer prompt burst only after defining and preregistering
a workload scale target; do not train RL or integrate LPLB/EPLB into serving
until a repeated actual expert CUDA ratio ≥1.15 is first observed.

## Artifacts

- raw per-rank trace: `raw_live/rank0..3.jsonl`
- backend proof: `backend_proof/`
- model config: `model_config_audit.json`
- invocation metrics: `invocation_metrics.csv`
- condition/layer summaries: `stage0_condition_summary.csv`, `stage0_layer_summary.csv`
- figures: `figures/`
- exact command: `experiment_command.txt`
- reference provenance: `capacity_eplb_reference_manifest.json`
- gate: `gate_summary.json`
"""
    (out / "stage0_report.md").write_text(report)
    report_dir = out.parents[2] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "rl_ep_straggler_deepseekv2_poc.md").write_text(report)


if __name__ == "__main__":
    main()
