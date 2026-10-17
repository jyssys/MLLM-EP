# PARPP final report

Verdict: `QUALITY_FAIL`.
Highest completed stage: Stage 3, paired free-running pilot screening.
The tested causal predictors materially harm reasoning quality before distributed implementation.
The experiment therefore stops before full-benchmark expansion, latency-oracle measurement, or asynchronous PP4 engineering.

## What actually executed

The checkpoint is `ByteDance/Ouro-1.4B-Thinking`, revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`.
The runtime uses the unmodified official model code, PyTorch 2.8.0+cu128, Transformers 4.57.1, and SDPA.
The model uses BF16 and fixed R4.
Its 24 physical layers execute 96 logical layers per position.
Each recurrent depth and physical layer has independent KV storage.
The candidate partition contains four groups of six physical layers.

Only physical H100 GPUs 4, 5, 6, and 7 execute this PoC.
They host independent single-GPU Exact, Stale, Delta, and W0 quality workers respectively.
Each worker sees its assigned physical GPU as logical `cuda:0`.
This is serial numerical emulation of concurrent-round inputs, not a physical PP4 engine.
The environment name `llada2-mini-reference` does not identify the loaded model.
Restored vLLM processes are utilization burners, not Ouro stage workers.

The original requested specification destination did not exist initially.
The exact PARPP source was available at `poc_flashvep/reports/async_recurrent_pp_poc_spec.md`.
Its SHA-256 is `d89b206c419aa8bcd644db889596ce6c8d3aa6e2b5d4dc589d30628e564dbea1`.
The experiment copied that source into `reports/spec.md`, as its repository-location clause requires.

## Exact baseline and causal validity

The exact-hook reference passes bitwise logits and all-slot KV parity on 16 smoke prompts.
Four complete single-request natural generations match the official driver token for token.
A separate four-request native batch also passes complete token and EOS parity with exact stage hooks.
The valid approximation diagnostic repeats bitwise exact-reference parity over two 32-position continuations.
All checked candidate positions write each of the 96 depth/layer slots exactly once.

All relaxed inputs freeze before current-round stage computation.
The emulator retains computed preceding-stage outputs from the previous round.
It does not recycle input predictions or use current-round exact activations as predictors.
Candidate KV persists into later generated tokens.
All task arms use exact prefill and official fixed-batch greedy generation.

The CPU regression suite passes 11 tests in the pinned environment.
Failed bookkeeping, generation-driver, and wrong-interpreter attempts remain preserved and excluded where appropriate.
Their scope appears in [evidence_scope.md](evidence_scope.md).
These implementation checks do not establish quality preservation.

## Main measured quality result

The primary pilot contains 16 questions per dataset, with a common 8,192-token cap and EOS enabled.
The table reports actual free-running final-answer accuracy.
The bootstrap uses 10,000 paired question resamples.

| Arm | Warm-up | GSM8K correct/n | GSM8K accuracy | MATH-500 correct/n | MATH-500 accuracy |
|---|---:|---:|---:|---:|---:|
| Exact R4 | Not applicable | 15/16 | 93.75% | 14/16 | 87.50% |
| Stale previous-round output | 1 | 8/16 | 50.00% | 3/16 | 18.75% |
| Calibrated delta | 1 | 7/16 | 43.75% | 3/16 | 18.75% |
| Causal initializer + delta | 0 | 6/16 | 37.50% | 1/16 | 6.25% |

For delta versus exact, GSM8K loses 50.00 pp with paired 95% CI [-75.00, -25.00].
MATH-500 loses 68.75 pp with paired 95% CI [-87.50, -43.75].
Every candidate's upper CI bound is below -10 pp on both datasets and both accuracy endpoints.
Both the controller and independent final scoring return `QUALITY_FAIL`.

The exact arm truncates 2/32 responses.
Stale, delta, and W0 truncate 19/32, 19/32, and 20/32 responses respectively.
The candidates also produce EOS-completed wrong answers.
The losses are not solely missing-parser or output-cap artifacts.
The [accuracy report](accuracy_report.md) includes all intervals, lengths, parsing failures, transitions, and concrete counterexamples.

## Why activation similarity was insufficient

Eight calibration requests supply 136 observed positions.
Exact adjacent-loop states have mean cosine 0.915495 and mean relative L2 change 0.340335.
These are descriptive calibration statistics, not substitution guarantees.

Repeated delta substitution produces mean effective stage-input relative L2 error 0.545706 in the small forced-prefix diagnostic.
Its mean next-token KL is 0.258074 nats, with 55/64 top-1 agreements.
The actual free-running pilot still loses many correct answers.
A low local discrepancy or high cosine does not preserve the complete reasoning trajectory.

There is also a short-horizon information-path constraint.
Under stale execution, a newly computed value can propagate as `F1(R2) -> F2(R3) -> F3(R4) -> F4(R5)`.
Fixed R4 does not execute R5.
This symbolic lag helps explain the risk. It is not a proof that every predictive architecture fails.
The experiment does not isolate one unique causal explanation for every wrong answer.

## Stage completion and latency scope

| Stage | Outcome |
|---|---|
| 0: official baseline and independent depth KV | PASS on the stated integrity probes |
| 1: activation and actual input-target analysis | Measured calibration diagnostics |
| 2: stale, delta, and W0 causal predictors | Implemented and causality-checked |
| 3: free-running paired quality pilot | Complete, `QUALITY_FAIL` |
| 4: full GSM8K-1,319 and MATH-500-500 | NOT_RUN, no pilot survivor |
| 5a: measured-cost multi-GPU latency oracle | NOT_RUN, quality gate failed |
| 5b: real asynchronous PP4 prototype | NOT_RUN, quality gate failed |
| 6: dynamic depth and early exit | NOT_RUN, quality gate failed |

The experiment measures no optimized Exact PP4, Exact TP4, DP4 serving frontier, or asynchronous PP4 latency comparison.
The experiment presents no simulated schedule as GPU evidence.
The specification's 16/7 and 16/4 examples remain conditional zero-cost bounds.
They do not establish achievable speedup against optimized TP4.
Single-GPU quality wall times do not support an architecture speedup claim.

## Decision and next action

Do not implement the distributed engine for these predictors.
Do not add dynamic depth, extra exact warm-up rounds, or a policy to conceal the quality failure.
The result closes this tested stale/scalar-delta/W0 feasibility attempt.
It does not reject every future asynchronous recurrent architecture.

This evidence justifies no further distributed experiment.
Any separate future attempt must first demonstrate a materially better causal input predictor under the same free-running quality gate.
That condition is not evidence that such a predictor exists.

AsyncDiff supplies the dependency-relaxation motivation, not a transferable quality guarantee.
CDB addresses depth-adaptive cross-request execution, not these within-round input predictions.
The [prior audit](prior_overlap.md) records pinned upstream sources and the novelty boundary.
The experiment claims no new parallelism contribution or speedup.

## Artifacts and publication

- [Reproduction commands](reproduce_final.md).
- [Exact and relaxed execution DAG](execution_dag.md).
- [Exact baseline validation](baseline_validation.md).
- [Activation diagnostics](activation_analysis.md) and [margin follow-up](activation_followup.md).
- [Measured quality, paired intervals, and counterexamples](accuracy_report.md).
- `outputs/final_analysis/final_verdict.json` and the companion CSV/JSON files.
- `outputs/final_plots/` for final accuracy and boundary-error figures.
- `outputs/validation/evidence_final.json` for source, mask, KV, and EOS checks.

Worktree: `/home/esjung/MLLM-EP-async-recurrent-pp`.
Branch: `flashvep/async-recurrent-pp-poc`.
The final Git commit contains only `poc_async_recurrent_pp/` changes.
The original checkout and unrelated staged work remain outside this commit.
The final response provides the exact published SHA.

The final device audit confirms verified utilization burners on physical GPUs 4, 5, 6, and 7.
Each GPU reports 100% utilization at that observation.
No experiment GPU context remains.
The audit appears in `outputs/validation/final_gpu_state.json`.
