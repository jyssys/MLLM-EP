# Numerical correctness and B performance diagnosis

Status: COMPLETE_MATCHED_RUNTIME_CORRECTNESS_RELEASE.
Both models pass the documented internal output and state qualification.
This is not a claim of original-HF equivalence or benchmark answer accuracy.

## Current model decisions

| Model | Candidate | Internal PD parity | Original-HF equivalence | Answer accuracy | Current decision |
|---|---|---|---|---|---|
| Ouro R4 | Down256 plus small-row RMSNorm | Old and independent400-request seeds each pass400/400 for repeat and B comparisons | Eight-prompt triage8/8 only | NOT_MEASURED | FINITE_INTERNAL_QUALIFICATION_PASS |
| Huginn R32 | Down256 plus small-row RMSNorm | New200 seed fails199/200. Closed-loop400 repeat fails393/400 | Eight-prompt triage5/8 | NOT_MEASURED | NO_GO_CORRECTNESS for this candidate |
| Huginn R32 | Above plus adapter256 | Two200 and two400 cohorts pass repeat and B comparisons completely. Final byte/order checks pass | Eight-prompt triage5/8 only | NOT_MEASURED | FINITE_INTERNAL_QUALIFICATION_PASS |

Both models remain mandatory targets.
No RCC-PD, A/C mechanism or native-vLLM Loop PP implementation started.

## What the diagnosis established

The issue exists in ordinary PD_G1 repeats, not only B.
Three same-input numerical counterexamples are confirmed in the actual runtime stack:

1. Ouro MLP Down GEMM changes with row geometry.
2. RLT RMSNorm FP32 reduction changes with row geometry, then crosses BF16 rounding boundaries.
3. Huginn's recurrent adapter also changes with row geometry on real failure operands.

The reference LoopServing reports helped identify Down sensitivity.
Its historical Down-only correction does not solve the complete RLT operator stack.
The new correction targets the confirmed operations instead of padding every Linear.
It preserves original weights, BF16, attention semantics and fixed R4/R32 work.
It can change numerical reduction order. No original-model-lossless claim follows automatically.

## Current B performance evidence

The strongest saved matched-output Ouro W1 pair has400/400 sequence agreement.
B increases p95 TTFT from876.059ms to1082.067ms.
Its mean exposed handoff changes only from21.029ms to21.865ms.
The larger mean TTFT change lies mainly in prefill/backpressure and decode-resume intervals.
Those are host phases, not exclusive GPU kernel percentages.

The new small quiet-lane experiment has three independent starts of eight prompts with output cap8.
Ordinary B makespan ratios versus PD_G1 are0.864× for Ouro and0.946× for Huginn.
Both95% intervals include1.000×.
These small cohorts do not establish a universal slowdown or a sustainable-capacity result.

Allowing immutable KV fragments to remain queued improves Ouro's finite makespan against ordinary PD_G1.
However, its additional gain against the same generic PD_G1 readiness change remains uncertain.
That diagnostic has not passed full200/400 correctness qualification.
It is not an accepted loop-specific method.

Actual Nsight P2P bytes match transport counters in all four quiet profiles.
The actual copy timeline differs from cumulative host transfer observations.
Heavy profiles remain separate from clean latency.
No measured HBM bandwidth or fully isolated contention attribution is available.

## Completed expanded Huginn acceptance

The minimal adapter correction passes both newly isolated failure histories.
The tests ran independently on four physical pairs:

| Physical GPUs | Requests | Arrival rule | Seed | Purpose |
|---|---:|---|---:|---|
| 0,1 | 200 | Finite Poisson,0.2req/s | 2026103001 | Original failure-seed qualification |
| 2,3 | 200 | Finite Poisson,0.2req/s | 2026109001 | Failed independent-seed qualification |
| 4,5 | 400 | Closed-loop,capacity8 | 2026109001 | Reproduce and remove the failed400 gate |
| 6,7 | 400 | Closed-loop,capacity8 | 2026109002 | New independent-seed qualification |

Each cohort runs PD_G1, a new PD_G1 repeat, then B.
Each cohort stops before B if the baseline repeat fails.
Full-R32 work counts must also pass.
These concurrent runs measure correctness, not isolated performance.
Every cohort passes its repeat, B and full-depth accounting gates.
The controller restored only owned burners when each pair finished.

Final Huginn release checks additionally pass4/4 valid KV/hidden transfers in each PD_G1 and B arm.
Reversed admission with capacity3 passes8/8 output agreement in both arms.
The new geometry replay passes all inspected tensors and prefix hashes at batch1/2/3/4/8/16.
The installed RLT and imported reference remain unchanged.

The source addition for Huginn does not change Ouro's selected Down modules or Down/normalization function bodies.
Twelve archived Ouro qualification starts pass the CPU AST/module-selection equivalence check.
That source check is not a newly measured400-request rerun.
`configs/final_runtime_hashes.json` records the operator hashes and exact scope.

The baseline is ready for new matched-runtime experiments with strict per-mechanism qualification.
The user supplied a separate RCC-PD specification after this work.
No RCC-PD GPU result is included in this report.

## Reports and raw evidence

- [Engine comparison](looped_pd_b_loopserving_comparison.md).
- [Numerical counterexamples and acceptance](looped_pd_b_numerical_correctness.md).
- [B performance root causes and measured tables](looped_pd_b_performance_root_cause.md).
- [Reproduction commands](looped_pd_b_numerical_reproduce.md).
- [Raw namespace and resume instructions](looped_pd_b_numerical_diagnosis/next_actions.md).
- [E2E component percentages](looped_pd_b_numerical_diagnosis/e2e_component_shares.csv).
- [Clean finite latency records](looped_pd_b_numerical_diagnosis/quiet_latency_raw.csv).
- [Paired finite comparisons and95% intervals](looped_pd_b_numerical_diagnosis/quiet_paired_comparisons.csv).
- [Actual operand values and BF16 bits](looped_pd_b_numerical_diagnosis/first_divergent_tensor_values.csv).

The branch is `flashvep/looped-pd-b-numerical-diagnosis`.
The baseline is `fea20edef840ed357ee0493c2f0dd69668c66c1f`.
The unrelated staged-index SHA256 remains `4fff05df83173f2604af9b8486619d286ca9ff11f53dfd62bb57c6a38b815a18` at the recorded preservation check.
All1637 locked original source files remain unchanged at that check.
Frozen failures and prior negative performance evidence remain unchanged.
The final source commit is the published tip of this diagnosis branch at handoff.
The subsequent RCC prerequisite manifest must record its resolved immutable SHA.
