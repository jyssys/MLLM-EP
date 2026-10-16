# Ouro reasoning fidelity and real-workload multi-GPU audit

Status: COMPLETE under the specification's weak-quality early-exit rule.
The native quality collection, bounded reference checks, and small hardware audit are complete.
The campaign does not run the gated broad systems study or implement a new framework.

## Decision

The campaign does not establish a quality baseline for a new multi-GPU framework.
The primary AIME sampling cohort misses the frozen 90% natural-termination target.
The separate greedy AIME cohort also misses that target.
The original strict reference-state limits remain unmet.
Functional next-token distributions agree within the separate declared JSD limit.

| Decision axis | Final result | Scope |
|---|---|---|
| Quality | `QUALITY_BASELINE_WEAK` | Public evaluation works, but AIME censoring remains substantial |
| Correctness | `REFERENCE_RUNTIME_PARTIAL` | Functional checks pass, strict state checks do not |
| Systems | `PARTIAL_PROFILE_ONLY` | Broad TP×DP profiling remains gated |
| New framework | NO-GO at this gate | No recurrence-specific method or speedup claim |

This decision does not establish that the checkpoint is defective.
It also does not establish that recurrence-specific physical headroom is absent.
The experiment follows the stop rule instead of manufacturing a new method.

## Model and physical execution

The model is `ByteDance/Ouro-2.6B-Thinking` at revision `f1edd81e7ac41355db670500ceaf204e0f73af68`.
The checkpoint contains 48 physical layers and four recurrent passes.
Each forward executes 192 logical layers with 192 independent KV slots.
The hidden dimension is 2,048, with 16 Q heads and 16 KV heads.
The FFN intermediate dimension is 5,632.

The paper's architecture table lists 48 layers.
The pinned model card lists 24 layers.
The loaded config agrees with the paper, not that model-card line.
The campaign does not edit the architecture to match the card.

The native runtime uses BF16, FlashAttention3, and ordinary CUDA Graphs.
Its environment contains vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129, Transformers 5.16.1, and CUDA 12.9.
The exact installed vLLM VCS commit is UNKNOWN.
Installed source hashes identify the executed implementation more precisely than the version string alone.

The reference uses PyTorch 2.8.0+cu128, Transformers 4.57.1, and SDPA.
The card-recommended Transformers 4.54.1 fails the pinned cache constructor.
That failure remains in the evidence.
The functioning 4.57.1 reference is an explicit version deviation.
Neither shared interpreter changes to satisfy the other runtime.

Only physical GPUs4–7 execute model or burner work.
They are H100 80GB HBM3 devices with NV18 pair connectivity.
Primary TP1 uses GPU4, TP2 uses GPUs4–5, and TP4 uses GPUs4–7.
Logical ranks follow the selected mask in order.
The separately labeled HF and operator diagnostics can use another permitted device.

| Physical GPU | UUID without GPU- prefix |
|---:|---|
| 4 | 6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 5 | a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 6 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 7 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

Ouro is a dense Transformer, not an MoE model.
Native layouts use TP1/TP2/TP4 with EP1, DP1, PP1, and CP1.
There is no expert Dispatch/Combine All-to-All in this experiment.
Native TP reductions and vocabulary gathering remain real communication.
FlashAttention kernels with Combine in their names are not expert Combine.

## Public reasoning quality

Every result below uses natural generation with EOS enabled.
No forced 64-token output supplies accuracy evidence.
P1 is the source-selected explicit-thinking prompt, not the better-scoring diagnostic arm.
The sampled policy uses temperature 1.0, top-p 0.7, unlimited top-k, and seeds 41015–41017.
The separate greedy policy uses temperature 0.
All requests stay within 32,768 total positions.

| Lane | Cap | Completed correct / all | Problem-bootstrap 95% CI | EOS / all | Capped |
|---|---:|---:|---:|---:|---:|
| AIME2024 sampling | 8,192 | 37/90 = 41.11% | 25.56%–56.67% | 41/90 | 49/90 |
| AIME2024 sampling | 16,384 | 51/90 = 56.67% | 41.11%–72.22% | 61/90 | 29/90 |
| AIME2024 sampling | 24,576 | 52/90 = 57.78% | 43.33%–72.22% | 79/90 | 11/90 |
| AIME2024 greedy | 24,576 | 19/30 = 63.33% | 46.67%–80.00% | 21/30 | 9/30 |
| Frozen MATH100 greedy | 8,192 | 87/100 = 87.00% | 80.00%–93.00% | 91/100 | 9/100 |

MATH100 has 9% length stops, so its frozen rule does not trigger another cap.
AIME executes every declared whole-lane extension through 24,576 tokens.
The campaign never retries only failures or removes capped outputs from the denominator.
All 460 quality generations remain available across the executed caps.
Repeated caps do not create more independent benchmark problems.

The bootstrap uses 5,000 problem resamples and preserves all seeds within each problem.
AIME has 30 independent problems, not 90 independent problems.
MATH100 is not full MATH-500 accuracy.
Pass@10 is NOT_RUN because only three sampled seeds exist.
No optional ten-seed or full-500 expansion runs after this gate failure.

At the final sampled cap, the seed scores are 19/30, 16/30, and 17/30.
The median sampled AIME output contains 10,129.5 tokens.
The greedy median is 9,230 tokens.
Both AIME P95 lengths equal the 24,576-token cap.
These are actual long-reasoning workloads, not long prompts with short forced outputs.

The authors report R4 AIME2024 pass@1 64.7% and pass@10 90.0%.
They use an in-house harness and LLM judge.
The exact private AIME prompt, grading rubric, cap, and seed stream remain unavailable.
The present public integer grader therefore carries `REFERENCE_PROTOCOL_NOT_IDENTICAL`.
The published score lies within the sampled problem-bootstrap interval.
That fact establishes neither exact reproduction nor statistical equivalence.

Sources: [Ouro Table 9 and Appendix C](https://arxiv.org/html/2510.25741v2) and the [author project](https://ouro-llm.github.io/).
The paper's separate 8,192-token safety experiment does not specify its AIME evaluation cap.
See [author_protocol_comparison.md](author_protocol_comparison.md) for the protocol matrix.
See [quality_scorecard.md](quality_scorecard.md) for every cap, seed, length, and censoring field.

## Falsification and forensic checks

The omitted input thinking marker is not a sufficient explanation for the old result.
P0 naturally produces the opening thinking token in all eight diagnostic requests.
Seven of eight also produce the following newline.
The overlapping diagnostic completes correctly in 3/8 P0 and 2/8 P1 requests.
Those eight questions are not an untouched confirmation cohort.
The campaign retains P1 because of its source-based freeze, not those scores.

The completed-cap audit finds no EOS handling failure.
Every observed natural stop ends with ID2.
No length-stopped output contains ID2.
No generation exceeds its effective cap or the 32,768-position limit.
Two completed answers in the 24,576-token sampled cohort fail the accepted integer-range check.
They are 1024 and an irrational expression, not valid AIME integers lost through formatting.

Both reference tokenizer versions reproduce all 260 frozen prompt encodings.
They also reproduce all 462 checked native output strings and final-answer extractions.
Those outputs comprise 460 quality generations and two continuity completions.
Native re-encoding differs on four MATH prompt arms under Transformers 5.16.1.
The engine receives frozen token IDs, so those re-encoding differences do not enter the benchmark.

Changing the cap also changes some generated prefixes.
The sampled 8,192→16,384 comparison changes 20/90 common prefixes.
The sampled 16,384→24,576 comparison also changes 20/90.
The paired score changes are +15.56 pp and +1.11 pp respectively.
Their problem-bootstrap 95% intervals are +7.78–+24.44 pp and −4.44–+7.78 pp.

The preceding engine history also differs across caps.
The experiment cannot attribute every score increase solely to reduced censoring.

## HF and native fidelity

The teacher probe uses eight frozen AIME prompts and two metadata-ordered MATH prompts.
Four AIME prompts calibrate limits.
The remaining six problems test those limits.
Checkpoints are 0, 1, 4, 32, 128, and 1,024 cached steps.
Confirmation means held out from tolerance calibration, not a new benchmark dataset.

All 389 inspected loaded TP1 tensors match the pinned BF16 checkpoint bitwise.
Native and HF cache comparisons map physical-major and loop-major slot orders explicitly.
The state check compares selected written KV coordinates, not every cache byte.
Native pre-final-norm state is NOT_CAPTURED in the primary compiled path.

| Native layout | Max hidden relative L2 | Max sampled-KV relative L2 | Max confirmation JSD, nats |
|---|---:|---:|---:|
| TP1 | 14.965% | 4.451% | 0.00037815 |
| TP2 | 17.284% | 5.674% | 0.00056047 |
| TP4 | 16.833% | 4.714% | 0.00041751 |

The maxima above are rounded and are not confidence intervals.
The predeclared functional JSD limit is 0.005 nats.
Every confirmation distribution satisfies that limit.
The original strict state planning ceilings fail and remain unchanged.
An operator-path ablation reduces one late hidden mismatch but does not explain every checkpoint.
The report does not replace this partial result with exact HF equivalence.

Selected same-configuration numerical repeats match bitwise in all 126 inspected fields.
Those fields come from one calibration problem, not 126 independent requests.
The compared long P1 repeat also matches all four selected outputs within the same diagnostic session.
Cross-rank logits match in 120/120 stored groups.
Hidden states match bitwise in 101/120 groups, with the responsible difference NOT_CAPTURED.
The maximum hidden relative L2 across those groups is about 0.9790%.

Two longer MATH probes finish correctly under HF and native TP1/TP2/TP4.
Those eight outputs cover two independent problems.
Short capped AIME probes do not supply completed benchmark accuracy.
See [hf_vllm_fidelity.md](hf_vllm_fidelity.md) for the limits and operator ablation.

The bounded HF corroboration uses two previously declared AIME problems and both prompt arms.
Both arms finish problem60 correctly with EOS.
P0 produces 4,037 tokens, and P1 produces 2,228 tokens.
Both arms reach the 8,192-token cap on problem61 without a parsed final answer.
Every recorded KV slot has the expected prompt-plus-generated-prefix length.
These four outputs do not establish HF AIME30 accuracy or optimized serving performance.

The default-threshold selector audit covers one calibration teacher trajectory.
It selects R4 at all 1,025 inspected last-position forwards.
The six stored checkpoint distributions match the explicitly forced R4 output in that diagnostic.
This result does not identify early output selection as the cause of the quality gap.
It also does not prove the selector's behavior on every input.
All actual reference comparisons explicitly select recurrent step3, meaning R4.

The HF generation warning concerns mask inference when pad and EOS both equal ID2.
A CPU test of the installed helper returns all-ones masks for the four unpadded frozen inputs.
The full running model's internal attention mask remains NOT_CAPTURED.
The audit does not change the reference execution parameters.

The fresh default-core control repeats the same client request history as the first quality wave.
Its warm prefixes match all four requests.
Its long outputs still differ in all four requests at indices 180, 333, 1,926, and 39.
The two long waves within that fresh session match exactly.
Equal client history therefore does not establish equal internal execution or reproduce the original long wave.

Two fresh sessions with `VLLM_ENABLE_V1_MULTIPROCESSING=0` reproduce all four compared long outputs exactly.
Their within-session repeats also match.
Those outputs differ from the original primary run.
This is a bounded generic scheduling diagnostic, not a new quality baseline or a loop-specific method.
The primary scored configuration remains unchanged.

The fixed-logit sampler probe also repeats across fresh processes and B1/B4 in the tested conditions.
It does not execute model forwards or capture actual model-request RNG offsets.
The first differing model logits and exact physical cause remain UNRESOLVED.
Benchmark intervals do not account for unmeasured between-session variation.
No diagnostic result replaces a scored public benchmark output.
See [reproducibility_limits.md](reproducibility_limits.md) for the precise scope.

## Small hardware audit and unexecuted systems work

The frozen fallback executes two metadata-ordered MATH continuity problems at TP1/TP2/TP4.
Each problem has three technical repeats per layout.
The prompts contain 149 and 101 tokens respectively.
All 18 completions end with EOS and match their references.
The independent problem count remains two, not 18.
Selected-state fidelity hooks are absent from this timing lane.

| Layout | Output tokens, algebra / probability | Median clean API E2E, s | Median backend TTFT, ms | Median average cached TPOT, ms | Completed correct |
|---|---:|---:|---:|---:|---:|
| TP1 | 1,421 / 1,724 | 29.546 | 107.481 | 18.672 | 6/6 |
| TP2 | 1,420 / 1,664 | 26.028 | 122.307 | 16.795 | 6/6 |
| TP4 | 1,421 / 1,286 | 20.059 | 116.830 | 14.748 | 6/6 |

These medians describe six measured calls per layout, split equally across two problems.
The API wall includes native client processing, not an HTTP serving frontend.
Backend TTFT uses the runtime's queued and first-token timestamps.
Average cached TPOT is not a per-token ITL distribution.
The raw CSV preserves all per-request metrics and lengths.

Within each layout, all three technical repeats have identical token IDs for each problem.
Across TP degrees, both natural output sequences differ from TP1.
The first differences occur at token indices 81 and 184 respectively.
The probability completion also becomes substantially shorter at TP4.
The observed E2E differences therefore are not matched-shape hardware speedups.
They are not new-method improvements.

The algebra API E2E ranges are 26.508–26.833 s, 23.839–24.016 s, and 21.023–21.141 s at TP1/2/4.
The probability ranges are 32.259–32.311 s, 28.039–28.070 s, and 19.049–19.095 s respectively.
These are technical-repeat ranges, not population confidence intervals.
The two-problem audit does not establish a serving frontier for the full frozen cohort.
Idle assigned devices run project burners, as the requested operational policy requires.
Other GPUs remain outside the experiment's control.

Evidence: `results/exploratory_B1_summary.json` and `results/exploratory_B1_requests.csv`.
See `figures/exploratory_b1_natural.png` for the two-problem timing comparison.

The full 38-request AIME/MATH systems plan is NOT_RUN because the quality gate fails.
Concurrent TP1×DP4 and TP2×DP2 are NOT_RUN.
The C1/2/4/8 serving sweep is NOT_RUN.
Token-index ITL curves, measured residency limits, and a four-GPU Pareto frontier are NOT_RUN.
No sum of isolated TPS values supplies a concurrent-engine result.
No other PoC's absolute latency supplies a matched baseline.

No calibrated profiler lane establishes a removable non-overlapped critical-path cost here.
The report does not turn kernel service sums into an additive E2E breakdown.
TP collective critical-path time, rank waiting, and recurrent-specific removable cost remain UNRESOLVED.
New framework headroom is NOT_ESTABLISHED, not measured as zero.

## KV memory and novelty boundary

The exact aggregate BF16 payload is 1,572,864 bytes per token, or 1.5 MiB.
TP1/TP2/TP4 own 16/8/4 KV heads per slot respectively.
Loaded allocation records show 192 slots and actual GPU ownership.
Reserved KV pool, filled logical payload, graph workspace, and device-wide memory remain separate accounts.
The final small audit reports allocation without treating every reserved page as filled request KV.

| Layout | Local KV heads / slot | Unique weights / rank, GiB | Reserved KV pool / rank, GiB | Torch allocated / rank, GiB | Torch reserved / rank, GiB |
|---|---:|---:|---:|---:|---:|
| TP1 | 16 | 4.9695 | 60.6328 | 65.8721 | 65.9668 |
| TP2 | 8 | 2.4851 | 63.6680 | 66.4102 | 66.4238 |
| TP4 | 4 | 1.2429 | 64.9277 | 66.3575 | 66.7148 |

These rounded values describe the actual small-audit ownership records.
All ranks within each reported TP layout have the listed values at this precision.
The pool reservation does not shrink by four because memory utilization stays at 0.85.
Per-token KV head ownership still divides across TP ranks.
The CSV's filled-KV field is a structural payload estimate, not observed allocator occupancy.
No measured concurrency limit follows from these reservations alone.

See `figures/exploratory_rank_memory.png` for measured per-rank weight and pool storage.

The existence of 192 KV slots is not a new placement method.
Imperfect TP scaling also does not establish a recurrence-specific bottleneck.
Generic TP×DP, paged KV, and graph corrections remain existing runtime mechanisms.
The campaign implements no new scheduler, compression, speculative decode, or recurrent execution engine.
See [prior_overlap.md](prior_overlap.md) for Ouro, FlashLoop, CDB, vllm-rlt, LoopSpec, and WaveFront boundaries.

## Failures, preservation, and reproduction

The failed 4.54.1 cache constructor remains preserved.
The first native diagnostic fails on missing metrics and is not timing evidence.
The overlapping-GPU calibration attempt aborts without killing an unidentified job.
The corrected stable tie ordering changes derived ranking analysis, not raw states or benchmark grades.
The historical restart-property test still fails under different prior request histories.
It does not isolate a universal fresh-session sampling bug.

Fifteen implementation checks pass in the final guard-regression suite.
The separate historical scientific property remains a failed test.
The report does not claim that every test passes.
See `results/validation_execution_v3.json` for both exit codes and complete output.
Earlier validation files remain preserved.

The first fixed-history engine also fails because a restored burner starts late on GPU4.
The repair prevents duplicate restoration without changing any model setting.
The initialization-only retry uses a new output directory.
See [restore_race.md](restore_race.md) for the failing regression, ancestry, and scope.

The campaign does not reset or clean old worktrees, staged changes, earlier results, or failure logs.
Only this PoC directory belongs to the new commit.
Long output IDs and text enter a compressed evidence archive.
Selected state arrays and unassigned-process identities remain in hashed local evidence outside Git.
Original files remain unchanged.
The evidence manifest identifies each storage boundary.

An independent CPU replay extracts the public output archive into a new temporary directory.
It verifies all 122 wave-file hashes.
It reproduces every score, problem-bootstrap field, and byte of the grading CSV.
This check reproduces grading, not stochastic generation or strict reference-state equivalence.
See `results/published_replay_verification.json`.

Reproduction commands appear in [reproduce.md](reproduce.md).
Branch: `flashvep/ouro-reasoning-fidelity-multigpu-poc`.
The containing Git commit identifies this report and its evidence.
See `reproduce.md` for the exact revision lookup.
The final burner state appears in `results/final_burner_restore.json`.
Its verification is `RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs`.
The final scoped launcher leaves burners active on physical GPUs4–7.
No Ouro experiment process remains.

## Next action

Do not implement a new multi-GPU framework from this campaign.
First establish the public quality baseline under a separately frozen reproducibility protocol.
The bounded core-MP0 result is a candidate control, not a replacement benchmark result.
The author harness still differs from the public evaluator.
Long-output censoring and reference-state limits also require resolution before broad systems promotion.
No candidate runtime exists, so candidate correctness, benchmark quality, and measured method speedup are NOT_RUN.
