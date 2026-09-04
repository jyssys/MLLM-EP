# RL EP straggler feasibility — DeepSeek-V2-Lite

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
| measured invocations | 312 layer-level prefill views (4 EP ranks per invocation) |

The largest prefill invocation per rank was selected when duplicate engine
calls (chunk/profile/decode) existed. This keeps the primary comparison on
the real routed prefill while retaining raw JSONL rows for audit.

## Stage-0 natural straggler metrics

| metric | median | p25 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|
| rank assignment max/mean | 1.177 | 1.130 | 1.260 | 1.333 | 1.592 |
| expert CUDA max/mean | 1.029 | 1.018 | 1.049 | 1.183 | 1.583 |
| dispatch CUDA max/mean | 1.147 | 1.110 | 1.189 | 1.387 | 1.618 |
| combine CUDA max/mean | 1.054 | 1.032 | 1.099 | 1.168 | 1.369 |

Expert CUDA ratio distribution: **83.7% ≤1.10**, **12.8% ≥1.15**, **3.5% ≥1.25**, and **0.3% ≥1.50**. The maximum 1.583 case was an isolated layer invocation (`condition=math_vs_reasoning`, layer 21), not a repeated heavy regime.

Per-condition medians span `1.019`–`1.159`; only the `code_vs_math` pair crosses 1.15 at the median (1.159). Per-layer medians are mostly near 1.02–1.11; the highest is layer 21 at 1.114. Thus routing-count imbalance (rank median 1.177) does not translate into a repeated expert-kernel critical-path imbalance.

The event timing confirms the important distinction: expert CUDA max/mean has only a weak relationship to rank assignment ratio (Pearson r = -0.055), while expert ratio tracks the combined critical MoE timing (r = 0.861). The latter is a timing sanity check, not evidence of a natural straggler.

## Temporal and action stages

Not run by design. Stage 0 failed the preregistered natural-straggler gate,
so there is no qualifying heavy/transient hotspot on which to evaluate
CAPACITY_MILD/STRONG, EPLB_SMALL/LARGE, migration cost, myopic versus
future-aware oracle, or a learned policy. `capacity_eplb_reference_manifest.json`
records the inspected public references and their exact commits; it does not
claim an unmeasured gain.

## Strongest positive and counter-evidence

- **Positive:** assignment spread exists (rank max/mean median 1.177, p90 1.333), and isolated expert-timing outliers reach 1.583.
- **Counter-evidence:** the primary CUDA metric is median 1.029; 83.7% of invocations are at or below 1.10 and only 12.8% reach 1.15. Heavy ≥1.50 appears in 1/312 invocation(s), so no robust natural straggler is present.
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
