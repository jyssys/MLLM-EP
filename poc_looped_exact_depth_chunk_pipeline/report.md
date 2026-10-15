# Exact recurrent-depth × sequence-chunk pipeline

Final verdict: `QUALITY_BASELINE_WEAK`.

## Decision

The dependency graph admits a mathematically legal chunk schedule.
It is isomorphic to generic PP on the tied-unrolled 192-layer operation graph.
No independent recurrent-specific scheduling or memory advantage passes that comparison.
No distributed candidate serving engine or integrated generic PP engine runs in this campaign.
We claim no measured pipeline E2E speedup or candidate-quality PASS.

Native quality gate: `QUALITY_BASELINE_WEAK`. Component reference gate: `FAIL_UNDER_FROZEN_TOLERANCE`.

The independent novelty verdict is `GENERIC_PP_EQUIVALENT`. The reference-executor verdict is `CORRECTNESS_BLOCKED`.
Mathematical graph equivalence does not establish BF16 execution parity.

## Model, runtime, and physical devices

The checkpoint is ByteDance/Ouro-2.6B-Thinking at `f1edd81e7ac41355db670500ceaf204e0f73af68`.
The native engine is vLLM 0.26.0+cu129 with PyTorch 2.11.0+cu129 and Transformers 5.16.1.
It uses BF16, FlashAttention3, ordinary CUDA Graphs, and native TP4.
Only physical H100 GPUs4–7 execute kernels. Logical CUDA ranks0–3 map to those devices in order.
The topology audit records NV18 between every assigned GPU pair. Foreign processes remain untouched.
Each native TP4 rank has 192 cache slots and four local KV heads.
The loaded model has 48 physical layers and four passes. The card says 24 layers.
We preserve the loaded pinned model. We do not change its layer count to fit the card.

The config permits 65,536 positions. This campaign caps total positions at 32,768, consistent with the documented Thinking context.
Prefix caching, asynchronous scheduling, and EP remain disabled.

See [architecture provenance](model_architecture_vs_reference.md), [environment](results/environment.json), and [runtime audit](results/audit.json).

## Native public quality, not forced-token accuracy

All quality outputs are real natural generation. The quality lane keeps EOS enabled.
The official tokenizer chat template uses `add_generation_prompt=True`.
R remains four. The early-exit threshold remains 1.0. Native vLLM executes all four passes.
Published-style sampling uses temperature1.0 and top-p0.7. Seeds are 41015, 41016, and 41017.
The greedy lane uses temperature0. Quality batches contain at most eight requests.
Every problem, answer, prompt, and seed freezes before quality output.

| Dataset/lane | Cap | Correct completed / total | Exact-answer score, problem-bootstrap 95% CI | EOS / total | Parsed / total |
|---|---:|---:|---|---:|---:|
| aime24/greedy | 4096 | 6/30 | 20.00% [6.67%, 33.33%] | 6/30 | 6/30 |
| aime24/greedy | 8192 | 11/30 | 36.67% [20.00%, 53.33%] | 12/30 | 14/30 |
| aime24/sampling | 4096 | 16/90 | 17.78% [6.67%, 31.11%] | 16/90 | 23/90 |
| aime24/sampling | 8192 | 39/90 | 43.33% [28.89%, 57.78%] | 42/90 | 45/90 |
| math100/greedy | 4096 | 75/100 | 75.00% [66.00%, 83.00%] | 75/100 | 81/100 |
| math100/greedy | 8192 | 85/100 | 85.00% [78.00%, 92.00%] | 87/100 | 89/100 |

The final analysis uses the largest complete preregistered cap in each lane.
A cap increase repeats the entire lane. It does not select only failed or favorable problems.
Capped generations remain in the denominator and cannot count as completed correct answers.
The raw ledger separately records matching extracted answers inside capped outputs. That diagnostic is not completed accuracy.
AIME has 30 independent problems. The bootstrap retains all seeds of each selected problem together.
MATH100 is a subject-stratified subset. It is not the full MATH-500 score.

### Final-cap seed results

| Lane | Seed | Correct / total |
|---|---:|---:|
| aime24_greedy | 41015 | 11/30 |
| aime24_sampling | 41015 | 11/30 |
| aime24_sampling | 41016 | 15/30 |
| aime24_sampling | 41017 | 13/30 |
| math100_greedy | 41015 | 85/100 |

### Final-cap output diagnostics

| Lane | Mean tokens | Median tokens | P95 tokens | Capped / total | Mean repeated 16-gram fraction |
|---|---:|---:|---:|---:|---:|
| aime24_greedy | 6952.3 | 8192.0 | 8192.0 | 18/30 | 0.1285 |
| aime24_sampling | 6685.8 | 8192.0 | 8192.0 | 48/90 | 0.0445 |
| math100_greedy | 2996.6 | 1942.5 | 8192.0 | 13/100 | 0.0565 |

The repetition statistic counts duplicate overlapping 16-token sequences.
It is a descriptive diagnostic, not a quality judge or a model-selection rule.

### Comparison with the authors

The authors report AIME2024 pass@1 64.7%, AIME2025 50.3%, and OlympiadBench 76.44%.
Their in-house prompt/harness and LLM judge do not match this public deterministic evaluator.
The card example omits explicit do_sample and top-k settings. This run explicitly samples with top-k disabled.
The comparison is `REFERENCE_PROTOCOL_NOT_IDENTICAL`, not an official-score reproduction.
The official repository address returned HTTP404 during this audit. Its complete evaluation recipe remains unavailable.
The project page links to LoopLM. That repository describes a reimplementation, not the original private evaluation harness.
A low bounded score with substantial truncation does not establish that the pretrained model lacks the reported capability.
No prompt, temperature, top-p, seed, recurrence, precision, or answer key changes based on observed answers.
Only the preregistered complete-lane cap rule changes the output budget.

The exact AIME grader accepts only a final integer in [0,999].
MATH uses Math-Verify0.8.0 with LaTeX extraction, strict equivalence, and two-second limits.
All 100 frozen MATH references pass the parser self-check. Preset equivalence and wrong-answer controls also pass.
No manual answer adjudication changes a score. Two assistant inspections check EOS versus truncation only.
The first two execution attempts fail before benchmark generation because of tokenizer preflight checks.
The corrected path preserves every frozen token sequence. Two secondary prompts have different Transformers5 space tokenization but identical decoded prompt strings.
The seed check preserves both short same-batch repetitions. It is not a full-output universal determinism proof.

See [quality protocol](quality_protocol.md), [raw scores](results/quality_generations.csv), and [native quality JSON](results/native_quality_r4.json).

### Cross-cap prefix diagnostic

208/220 paired outputs preserve their common token prefix across the two caps.
All 120 AIME pairs match. Twelve of 100 MATH greedy pairs diverge within their common prefix.
This diagnostic changes max_tokens. It is not an identical full-configuration repeat.
The cause remains unresolved. The MATH score change cannot be attributed only to the larger budget.
No full-output reproducibility PASS follows from the short seed check.

See [per-problem first divergence](results/cap_prefix_reproducibility.json).

## Exact DAG and generic PP

Task F[r,j] includes a full 48-layer pass and its boundary norm.
It needs the preceding-depth hidden state for chunk j and committed same-depth prefix KV.
Intra-chunk masks use original causal positions. RoPE positions never restart at a chunk boundary.
The macro graph is a sufficient whole-stack schedule. It is not a minimal operator schedule.
The micro graph separates KV production, publication, layer output, and the four native pass norms.
Flattening loop/layer to `48*r+layer` exactly preserves its edge set.
An independent generic tied-unrolled construction matches that relabeled graph.
Finer operator streaming therefore also exists in generic PP. It is not a recurrence-only opportunity.

Each depth stage owns its KV. Interstage prefill traffic contains hidden activations, not remote KV reads.
For a single new output token, all four depths remain sequential. Unknown future tokens cannot fill the pipeline without speculation.

See [dependency contract](dependency_contract.md), [edge-set audit](results/dependency_gate.json), and [ownership](stage_memory_ownership.md).

## Fresh optimized native TP4 baseline

This hardware-only lane uses B1 and 64 frozen output IDs. It is not the quality lane.
Each configuration uses three warmups per prompt length. Each request then uses three technical repeats.
The analysis averages repeats before resampling requests with 5,000 draws.
Prefill means the native scheduled-to-first-token envelope. It is not attention-only time.
Cached TPOT uses 63 intervals. Clean E2E includes output handling after the loaded engine.

| Prompt tokens | n | Prefill ms | Cached TPOT ms | Clean E2E ms, 95% CI | Peak allocated GiB/rank |
|---:|---:|---:|---:|---|---:|
| 1024 | 8 | 175.882 | 14.722 | 1106.012 [1103.977, 1108.012] | 66.389 |
| 8192 | 8 | 217.292 | 16.034 | 1231.996 [1228.784, 1235.525] | 66.586 |
| 16384 | 8 | 408.792 | 17.101 | 1492.107 [1490.166, 1494.123] | 66.813 |
| 32704 | 3 | 972.900 | 20.703 | 2286.764 [2281.502, 2295.924] | 67.266 |

Historical absolute times never supply this baseline.
Peak allocated memory is the request maximum across ranks. It is not the same as the device-wide nvidia-smi memory counter.
The separate trace records four actual forwards on each of four ranks. Its 192-slot execution-count check is True.
Each rank records three actual CUDA Graph replays and four sampling calls.
The host counter records five execute_model calls. The GPU trace contains four active forwards, not five 192-layer executions.
Profiler observer overhead is 4.16%. Timing attribution is `DIAGNOSTIC_ONLY`.
The execution audit counts every observed forward. It does not multiply the final graph replay by output length.
The native PP class does not provide the required recurrent-depth stage interface.
A PP option or physical-layer partition does not establish exact 192-layer pipeline serving.

## Physical activation transfer

| Chunk rows | One-edge hidden bytes | Maximum PP4-edge median ms | Duplicated PP2×TP2 boundary median ms |
|---:|---:|---:|---:|
| 256 | 1048576 | 0.3772 | 0.3938 |
| 512 | 2097152 | 0.2258 | 0.1700 |
| 1024 | 4194304 | 0.1369 | 0.1233 |
| 2048 | 8388608 | 0.1215 | 0.1228 |
| 4096 | 16777216 | 0.1482 | 0.1506 |

These are physical NCCL send/receive paths on the four assigned GPUs.
The interval includes device completion and the final coordination barrier. It is not pure transport bandwidth.
The benchmark uses three warmups and 30 repeats. Those repeats are not independent serving requests.
It does not measure an integrated overlapped pipeline or assert free transfer.

The transfer sweep uses a preset order. Nonmonotonic medians include host/world-barrier overhead.
They do not establish a transport-size scaling law.

## Stage economics and E2E bounds

Cost-model status: `NOT_RUN`.

The native-quality gate stops the broad one-pass and native-chunk sweep.
One-pass stage costs, the CP sweep, calibrated PP improvements, and candidate E2E improvements are `NOT_RUN`.
The tiny reference and activation-transfer diagnostics do not replace that missing physical comparison.
No structural DAG headroom becomes measured speedup.

The chunk-size sensitivity, physical pipeline Gantt, and generated-length headroom sweeps are `NOT_RUN`.
Their absence follows the native-quality and generic-equivalence stop gates. Missing values do not mean zero headroom.

## Component correctness and conditional full-model tests

The official BF16 HF reference executes 4 full-versus-chunk comparisons on two frozen inputs.
Its frozen component gate is `FAIL_UNDER_FROZEN_TOLERANCE`.
Frozen limits are 0.125 for maximum absolute logit error and 0.01 for hidden and per-slot KV relative L2.
Selected greedy top-1 IDs must also match.
Every task checks all 48 same-depth prefix-cache lengths. Completed tasks preserve 192 independent slots.
The reference is single-GPU SDPA. Its wall is not an optimized serving benchmark.
Cross-engine native TP4 comparisons preserve full logits and selected written KV under identical token prefixes.
BF16 cross-engine numerical differences receive explicit records. Short probes cannot certify complete natural-output parity.

| Reference input | Chunk | Max absolute logit difference | Hidden relative L2 | Worst-slot KV relative L2 | Selected top-1 equal | Frozen gate |
|---|---:|---:|---:|---:|---|---|
| coding_00 | 64 | 0.593750 | 0.060459 | 0.103663 | True | False |
| coding_00 | 128 | 0.750000 | 0.041658 | 0.089789 | True | False |
| coding_01 | 64 | 0.750000 | 0.041361 | 0.062192 | True | False |
| coding_01 | 128 | 0.500000 | 0.025114 | 0.047327 | True | False |

The tested chunk executor does not satisfy the frozen BF16 numerical contract.
Matching selected top-1 IDs cannot override hidden/KV/logit failures.
The recorded errors grow across logical depth. Their exact numerical cause remains unresolved.
The logarithmic KV plot displays zero errors at 1e-7. The raw JSON retains their exact zero values.
This campaign does not prove that every exact chunk implementation must fail. It does not relax limits or promote this executor.

| Native/HF probe | Four teacher-aligned top-1 IDs equal | Independent four-token output equal | Full-logit relative L2 | Hidden relative L2 | Sampled-KV relative L2 |
|---|---|---|---:|---:|---:|
| 60 | True | True | 0.018504 | 0.027796 | 0.009890 |
| 61 | True | True | 0.021257 | 0.033133 | 0.011394 |

Native KV registration and HF cache indices use different iteration orders.
The comparison maps each native registered cache name to its logical index before comparing tensor values.

| Test | Status |
|---|---|
| Native R4 public quality | QUALITY_BASELINE_WEAK |
| HF exact-chunk component check | FAIL_UNDER_FROZEN_TOLERANCE |
| Integrated candidate numerical/KV test | NOT_RUN |
| Candidate 24-request full greedy parity | NOT_RUN |
| Candidate AIME/MATH quality | NOT_RUN |
| Integrated generic PP timing | NOT_RUN |
| Measured candidate E2E improvement | NOT_RUN |

No quality-preserving candidate speedup claim follows from these component probes.
The earlier retained-graph natural-output mismatch remains preserved in the preceding PoC.
This campaign does not reinterpret it as a universal parity PASS.

## Required decision checklist

1. The model/runtime pins, GPU UUIDs, and NV18 topology are in the environment and ownership records.
2. Same-chunk preceding-depth hidden states and contiguous same-depth prefix KV constrain overlap.
3. The proposed graph is identical to generic tied-unrolled PP, including native pass norms.
4. No shared-weight mechanism saves work beyond the matched generic graph.
5. The small official reference reports its frozen numerical/cache gate separately from graph validity. The current executor does not establish BF16 parity.
6. Runnable macro tasks follow the diagonal frontier. The dependency-only ideal has M+3 slots for four equal stages.
7. One-pass stage costs exist only if the quality-gated broad sweep runs. Missing values are unknown, not zero.
8. Actual BF16 hidden transfers use `rows*2048*2` bytes per boundary edge. The component benchmark measures their full-path times.
9. Stage-local KV stays local during prefill. Native TP4 decode would need a real cache transition.
10. Only already-produced activations can transfer during independent work on another chunk.
11. Startup/drain and resource constraints remain in the schedule. A dependency-only utilization is not measured GPU utilization.
12. Layer-level publication dependencies exist. The generic micro-graph retains those same edges.
13. Integrated generic PP does not run. A model cannot establish that it beats optimized native chunked TP4.
14. No measured candidate E2E headroom exists. Any available forecast has explicit transition and contention limits.
15. Weights, logical KV, allocated cache pools, and workspace remain separate memory accounts.
16. The native public benchmark produces real outputs. Published protocol reproduction and candidate quality remain distinct gates.
17. No independent advantage survives the generic PP equivalence objection.
18. Final verdict is `QUALITY_BASELINE_WEAK`. Do not implement a loop-specific framework from these results.

## Evidence and next action

All benchmark outputs, input IDs, seeds, caps, and failure logs remain preserved.
Existing dirty/staged changes and earlier PoCs remain unchanged.
Git includes the bounded numerical arrays and heavy traces. A SHA256 manifest also records external backup copies.
The scoped GPU guard restores project burners. No foreign process receives a signal.
If native quality remains weak, first obtain an adequate complete-output evaluation protocol. Do not tune the model to match a reference score.
Even a complete quality baseline would not remove the generic PP equivalence.
See [reproduction](reproduce.md), [timing contract](timing_contract.md), [prior boundary](prior_overlap.md), and [verdict JSON](results/verdict.json).

Branch: `flashvep/looped-exact-depth-chunk-poc`.
The delivered commit is the revision containing this report. The final handoff provides its exact SHA.

### Figures

- [exact_depth_chunk_dag](figures/exact_depth_chunk_dag.png)
- [memory_ownership](figures/memory_ownership.png)
- [native_baselines](figures/native_baselines.png)
- [native_quality_r4](figures/native_quality_r4.png)
- [reference_KV_relative_error](figures/reference_KV_relative_error.png)
