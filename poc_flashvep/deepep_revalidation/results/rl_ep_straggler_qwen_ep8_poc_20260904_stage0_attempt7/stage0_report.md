# Qwen3-30B-A3B EP8 natural straggler / RL-controller feasibility

## Executive decision

**FINAL STATUS: HOLD**  
**STRAGGLER_FOUND: YES**  
**STAGE-0: STRONG_GO**  
**RL_POLICY: NOT_RUN**

EP8 is a strong, real serving straggler testbed in this bounded run. The
measured routed-expert CUDA max/mean ratio is 1.287 median, with
60.9% of 576 invocation/layer views at or above 1.25
and 27 views at or above 1.50. The overall controller result
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
| rank assignment max/mean | 1.931 | 2.193 | 2.412 | 3.030 |
| expert CUDA max/mean | 1.287 | 1.372 | 1.458 | 1.878 |
| dispatch CUDA max/mean | 1.153 | 1.227 | 1.319 | 1.546 |
| combine CUDA max/mean | 1.442 | 1.545 | 1.649 | 2.011 |
| critical-path stage-sum max/mean | 1.051 | 1.072 | 1.099 | 1.173 |

The primary gate distribution is: 12.5% ≤1.10,
81.4% ≥1.15, 60.9% ≥1.25,
and 4.7% ≥1.50. Thus this is not a single
outlier: every condition has a median expert ratio between
1.080 and 1.388,
and the highest layer median is layer 16 at
1.506. The largest view is
`vision_proxy_long`, layer 44, ratio 1.878.

Rank assignment max/mean is high (median 1.931), and its Pearson
association with expert CUDA max/mean is r=0.577. The relationship is
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
ratio 1.000 and a 48.2%
load upper-bound proxy, while 1.25/1.50 capacity clipping would drop median
63.8%/59.1%
of assignments. The latter makes clear that clipping is not a free,
correctness-preserving gain. No GPU latency gain is claimed.

## Stage 1 — temporal structure

The same deterministic domain prompts were repeated three times. Exact route
histograms therefore show median adjacent-wave hottest-expert recurrence
1.000; this demonstrates repeatability
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
2. **Does routing skew connect to CUDA?** Yes, positively (r=0.577), but
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
