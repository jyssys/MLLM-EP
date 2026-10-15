# Recurrent depth × tensor-parallel scaling

Final verdict: `CONTROL_BLOCKED`. No loop-specific framework promotion.

We measured the full native scaling matrix. The broader tied-control gate remains incomplete.
The ordinary-graph natural follow-up passes, but an earlier retained-graph natural mismatch remains unexplained.
This conservative verdict does not classify the primary clean timing as invalid.

## Research question and scope

Does native recurrent execution add a physical TP penalty beyond the cost of the same mathematical operation tape?
The decisive comparison uses native Ouro and an explicit tape with identical shared weights, logical depth, and KV slots.
Two genuine Qwen checkpoints provide external AR context. They do not provide depth-matched causal controls.

The authoritative contract is [the frozen specification](authoritative_spec.md).
The parent commit is `f1002a2f4baa6d1524e7e9b9bc8065bdc5c89a69`.
No historical absolute latency enters the matched baseline.

## Environment and physical proof

The experiment uses physical H100 80GB GPUs 4, 5, 6, and 7.
The topology audit reports NV18 connectivity between those GPUs.
TP1 uses GPU4. TP2 uses GPUs4–5. TP4 uses GPUs4–7.
Logical CUDA rank 0/1/2/3 maps to physical GPU4/5/6/7 respectively.
Actual loaded ownership records check UUIDs, rank count, local heads, weights, and KV tensors.

The runtime is vLLM `0.26.0+cu129`, PyTorch `2.11.0+cu129`, and Transformers `5.16.1`.
The Torch CUDA build is 12.9. NCCL is 2.28.9.
All primary arms use native V2 execution, BF16, FlashAttention 3, compilation, and CUDA Graphs.
The graph capture sizes are 1, 2, 4, and 8 rows.
The runtime disables prefix caching, chunked prefill, and asynchronous scheduling.
The maximum context is 32,768. HBM utilization is 0.85.

These are dense TP models. There is no MoE EP Dispatch/Combine All-to-All.
Measured collectives include ordinary TP reductions and logits gathering.
The installed engine and checkpoint files remain unchanged.

See [environment pins](results/env_and_pins.json), [source audit](results/source_and_model_audit.json), and [physical ownership](results/physical_ownership.csv).

## Four mandatory families

| Family | Checkpoint revision | Physical / logical layers | Status |
|---|---|---|---|
| Native Ouro | f1edd81e7ac41355db670500ceaf204e0f73af68 | 48 / 48R | R1/2/4 × TP1/2/4 complete |
| Tied-Unrolled Ouro | Same exact checkpoint | Same 48 shared modules / 48R explicit calls | Full grid measured. Natural-output validation remains incomplete. |
| Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 28 / 28 | TP1/2/4 complete |
| Qwen2.5-3B | 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b | 36 / 36 | TP1/2/4 complete |

The four families use three independent pretrained checkpoints. Tied-Unrolled is an execution control, not another trained model.
The [architecture matrix](architecture_matrix.md) lists MHA/GQA, FFN width, vocabulary, Q/K normalization, and bias differences.
Qwen2.5 TP4 replicates each of two global KV heads across a rank pair.
All-layer K/V weight hashes and written KV samples check that replication.

## Workload and timing definitions

The primary cohort contains eight frozen requests: three coding, three mathematics, and two document requests.
Each request uses P=1,024 and P=16,384, with 64 identical teacher-forced output IDs within each model.
Each configuration uses three warmups and three measured technical repeats.
The analysis averages repeats before resampling eight requests with 5,000 paired bootstrap draws.

The 32K confirmation uses P=32,704 and three requests, one per domain. It is not an eight-request result.
The maximum total position count is 32,768. The experiment does not extrapolate RoPE.
The old Ouro input pool provides continuity. It is not a fresh held-out cohort.
Token counts match across models. Token IDs and semantic difficulty do not match across tokenizers.

Every latency cell below uses `prefill / cached TPOT / client E2E`, in milliseconds.
Prefill means scheduled-to-first-token wall time. It is not pure GPU prefill service.
TPOT uses 63 cached intervals. E2E includes prefill, all 64 outputs, native handling, and finalization.
The teacher hook retains native logits and argmax. It adds one small GPU copy per generated token to every primary arm.

See [timing contract](timing_contract.md), [all request results](results/all_config_per_request.csv), and [full headline table](results/four_family_headline.csv).
Those files include memory, ownership, phase-specific confidence intervals, and diagnostic validity fields.

## Native Ouro R×TP measurements

### P=1,024, n=8

| R | TP1: prefill / TPOT / E2E | TP2 | TP4 |
|---|---|---|---|
| 1 | 28.823 / 5.671 / 388.428 | 31.076 / 5.107 / 355.076 | 29.388 / 4.440 / 311.403 |
| 2 | 65.472 / 11.081 / 766.167 | 68.075 / 9.547 / 672.053 | 91.815 / 8.418 / 624.409 |
| 4 | 129.004 / 17.765 / 1250.447 | 115.271 / 16.504 / 1157.752 | 112.669 / 14.526 / 1030.307 |

### P=16,384, n=8

| R | TP1: prefill / TPOT / E2E | TP2 | TP4 |
|---|---|---|---|
| 1 | 242.416 / 8.939 / 811.852 | 159.440 / 6.455 / 571.874 | 105.684 / 5.479 / 456.664 |
| 2 | 485.327 / 15.388 / 1460.186 | 317.862 / 11.999 / 1080.236 | 208.065 / 9.958 / 841.429 |
| 4 | 981.057 / 26.414 / 2650.783 | 630.078 / 20.913 / 1953.682 | 407.972 / 16.982 / 1483.735 |

### P=32,704 confirmation, n=3

| R | TP1 E2E ms | TP2 E2E ms | TP4 E2E ms |
|---|---|---|---|
| 1 | 1357.272 | 845.698 | 655.941 |
| 2 | 2563.120 | 1671.521 | 1193.068 |
| 4 | 4898.273 | 3172.891 | 2202.883 |

The [native CSV](results/native_ouro_r_tp.csv) provides every phase and its request-bootstrap interval.
R1 and R2 change the effective network. They are not output-preserving accelerations of pretrained R4.

## Genuine AR scaling controls

### P=1,024, n=8

| Model | TP1: prefill / TPOT / E2E | TP2 | TP4 |
|---|---|---|---|
| Qwen3-1.7B | 16.139 / 3.716 / 252.457 | 15.791 / 3.184 / 218.665 | 17.144 / 2.869 / 200.111 |
| Qwen2.5-3B | 25.991 / 5.551 / 378.049 | 24.462 / 4.320 / 298.828 | 21.592 / 3.693 / 256.545 |

### P=16,384, n=8

| Model | TP1: prefill / TPOT / E2E | TP2 | TP4 |
|---|---|---|---|
| Qwen3-1.7B | 136.200 / 5.040 / 458.806 | 89.752 / 3.965 / 345.311 | 60.178 / 3.267 / 271.558 |
| Qwen2.5-3B | 220.272 / 6.404 / 629.297 | 136.409 / 5.111 / 463.900 | 86.635 / 4.311 / 363.749 |

At 16K, Qwen3 E2E speedup is 1.329× at TP2 and 1.690× at TP4.
The corresponding Qwen2.5 speedups are 1.357× and 1.730×.
Their TP4 cached-decode speedups are 1.543× and 1.486× respectively.
These are within-model ratios, not causal comparisons of recurrence.

The [real-AR CSV](results/real_ar_qwen_tp.csv) includes 32K confirmation and all phase-specific intervals.
Different depth, MHA/GQA, Q/K normalization, bias, FFN width, and vocabulary prevent direct recurrence attribution from raw E2E.
No optional Qwen3-8B measurement replaces either mandatory Qwen control.

## Does increasing R degrade TP scaling?

At 16K, the R1→R4 TP4 comparison gives:

| Phase | R1 speedup, 95% CI | R4 speedup, 95% CI | R4/R1 scaling interaction, 95% CI |
|---|---|---|---|
| Prefill | 2.294 [2.288, 2.300] | 2.405 [2.400, 2.409] | 1.048 [1.045, 1.051] |
| Cached decode | 1.632 [1.627, 1.636] | 1.556 [1.532, 1.579] | 0.953 [0.940, 0.966] |
| Client E2E | 1.778 [1.775, 1.781] | 1.787 [1.767, 1.805] | 1.005 [0.995, 1.014] |

Prefill scaling improves slightly. Cached-decode scaling declines modestly. E2E scaling remains similar.
The experiment must not combine these phases into a claim of a single recurrence penalty.
Short-prefill behavior is non-monotonic. At 1K, R2 TP4 prefill scales worse than R1, with interaction 0.777 [0.642, 0.926].
Its E2E interaction is 0.987 [0.950, 1.026]. R4 prefill scaling then improves relative to R1.
The [paired comparison CSV](results/paired_comparisons.csv) contains the full R1/R2/R4 and context analysis.

## Decisive tied control

`tied_tape.py` constructs 48R explicit ordered calls. Repeated calls retain the same physical modules and parameter storages.
The control preserves native residual resets, per-pass normalization, RoPE, masks, and independent loop KV allocations.
It does not clone the model R times.

All 126 selected same-TP/R field checks are bitwise equal across two prompt lengths and all ranks.
The fields are complete logit vectors, final-layer hidden vectors, and sampled KV values in every logical slot.
All 84 observed ordered model-kernel tapes also match between native and tied execution.
Those tapes cover three R values, seven rank instances per family, and four actual forwards at 16K.
The order comparison excludes generic memcpy and memset operations. Raw traces retain those operations.
Ranks, fields, and forwards do not increase the independent request sample count.

The primary observer records 64 nonempty mathematical forwards, 63 graph replays, and 64 sampler calls per request.
An extra empty `execute_model` API call does not execute another model pass.
The native model already compiles its constant recurrent loop into captured GPU work.
An explicit Python tape therefore need not produce a different GPU schedule.

The separate retained-graph lane compares 44 native/tied rank-and-shape pairs at R4 TP1/2/4 and R2 TP4.
All 44 pairs have equal node counts, edge counts, and node-type counts.
For example, R4 TP1 captures 2,306 nodes and 2,305 edges at each tested capture size.
These are direct count queries. They do not compare every kernel parameter or dependency-port attribute.
At 1K, all 44 additional ordered model-kernel tapes match after the same memory-operation exclusion.

The natural retained-graph lane does not fully pass output parity.
Of 24 technical native/tied comparisons, 22 match. The two failures repeat one math prompt at R4 TP4.
Both runs first differ at zero-based output index 114. Their prompts and first 114 output IDs match.
Within each arm, the two natural repetitions match. The mismatch cannot be dismissed as observed within-session randomness.
The ordinary-graph diagnostic separately tests that prompt and selected same-prefix logits. It does not relax the frozen tolerance.

That follow-up uses the ordinary graph policy in new native/tied TP4 sessions.
Both arms produce the same 256 natural token IDs on both technical repetitions.
Both also match the earlier native sequence. The previous tied retained-graph sequence remains different.
All 64 selected same-prefix field comparisons match bitwise across eight steps, two fields, and four ranks.
At the earlier divergence step, the ordinary native top-two logit margin is 0.125. Both ordinary arms select the same token.
The failure does not recur in this ordinary-graph pair. Its physical cause remains unresolved.
No tolerance changes or deletion of failed evidence occur.

The retained-graph observer has a separate failure ledger.
The first attempt did not retain the raw graph because the C++ constructor reset the flag.
A later legacy edge query returned CUDA error 402 when dependency metadata could not fit that API.
Another attempt omitted native metric reporting. The successful lane explicitly matches that primary runtime setting.
All failed attempts remain in `results/diagnostics`. None supplies headline timing.

The successful observer records node counts and edge counts directly.
Dependency-edge metadata remains unresolved where the legacy query returns error 402.
Matching counts do not establish formal graph isomorphism or equal edge-port metadata.

The 1K R4 TP1 scheduled-prefill ratio is an important exception to a simple equality claim.
Native/Tied is 1.241 [1.155, 1.322] for that phase.
However, its E2E ratio is 0.976 [0.967, 0.986], so tied execution does not improve that full request.
The exact cause of this phase discrepancy remains unresolved at this point.
Request-bootstrap intervals do not include configuration-session drift or every CPU/runtime confound.
The report retains this result. Matching kernel names do not prove equal wall time.

At 1K R2 TP4, Native/Tied is 1.446 [1.238, 1.734] for scheduled prefill.
Its E2E ratio is 1.072 [1.044, 1.113]. This exception requires an independent-session diagnostic.
That diagnostic repeats the frozen calibration cohort. It is not fresh held-out confirmation.
The campaign also schedules a reverse-order model-session recheck with the exact primary engine kwargs.
It keeps the ordinary raw-graph policy, original sampler hook, input IDs, warmup, and technical-repeat count.
The retained-graph and same-policy rechecks remain separate ledgers. Neither replaces the original measurements.

The reverse-order same-policy recheck preserves every primary engine kwarg in all four sessions.
It uses the same eight calibration requests and three technical repeats. It does not create fresh held-out evidence.

| 1K contrast | Primary Native/Tied E2E | Same-policy recheck E2E, 95% CI | Recheck tied reduction, 95% CI |
|---|---|---|---|
| R4 TP1 | 0.976 [0.967, 0.986] | 1.033 [1.027, 1.038] | 3.167% [2.670%, 3.670%] |
| R2 TP4 | 1.072 [1.044, 1.113] | 1.092 [1.067, 1.134] | 8.252% [6.249%, 11.443%] |

R4 TP1 changes direction across sessions. Its request-bootstrap interval does not measure that session uncertainty.
The R2 TP4 timing advantage survives this one same-policy recheck. Its cause remains unresolved.
This result must not disappear from the negative or blocked conclusion.
Equal kernel names and counts do not rule out different kernel parameters, readiness, or host effects.
No checked non-overlapped component yet explains the R2 difference as an intrinsic recurrence penalty.
R2 also changes pretrained R4 depth. It is not a quality-preserving acceleration of the released checkpoint.
At 16K and 32K, primary R2 TP4 E2E ratios are 0.999 [0.993, 1.006] and 1.001 [0.997, 1.007].
Thus the observed short-context contrast does not extend to those longer-context measurements.

See [same-policy comparisons](results/same_policy_recheck_comparisons.json) and [kwarg checks](results/same_policy_recheck_contracts.json).

All ten 1K heavy-profile strata exceed the 10% observer threshold.
Their signed inflation ranges from 18.771% to 54.693%. Fine timing attribution from that lane is `UNVALIDATED`.
Kernel counts remain descriptive. Those profiler walls do not explain the clean 1K latency differences.
The natural-trajectory teacher-hook check uses five technical pairs on one request per configuration.
Its mean signed differences range from −0.746% to +0.801%. This is not an independent-request observer confidence interval.

See [tied contract](tied_unroll_contract.md), [tied latency matrix](results/tied_unrolled_r_tp.csv), and [kernel tape checks](results/native_tied_kernel_comparison.json).

## Physical kernel evidence and attribution limits

The TP4 R4 trace executes 192 attention operations per forward.
Its ordinary reductions comprise two per logical layer plus the embedding reduction. Logits gathering adds one collective.
Prefill uses NCCL reductions. Cached decode uses the custom TP reduction path.
The FlashAttention `FwdCombine` kernel is an attention kernel, not EP Combine.

The following values are TP4 rank0 GPU kernel service at P=16K. They are not additive E2E shares.

| Phase and component | Ouro R1 service ms/forward | Ouro R4 service ms/forward |
|---|---|---|
| Prefill attention, including KV reads | 20.475 | 82.503 |
| Prefill combined QKV/FFN/output GEMMs | 26.626 | 106.658 |
| Prefill TP collective kernels | 31.936 | 126.497 |
| Cached attention, including KV reads | 1.185 | 4.732 |
| Cached combined GEMMs | 1.056 | 4.167 |
| Cached TP collective kernels | 0.544 | 2.424 |

Prefill attention, GEMMs, and TP service grow approximately with the fourfold executed depth.
Kernel service includes protocol and readiness effects. It is not pure data transport or removable waiting time.
QKV and FFN timing remain combined where kernel names cannot separate them reliably.
Pure non-overlapped communication and globally aligned rank waiting remain `UNRESOLVED`.

The [outer-event CSV](results/outer_events.csv) records actual occurrences rather than multiplying one replay by 63.
The [kernel-service CSV](results/forward_kernel_services.csv) normalizes by actual logical depth.
The [profile validity file](results/profile_validity.json) separates observer differences from clean timing.
Profiler wall time never becomes a headline E2E result.
Across 24 configurations, the single-profile signed wall difference ranges from −1.519% to +6.706%.
No primary kernel stratum exceeds the 10% observer threshold. This check does not establish an independent observer confidence interval.
The outer-observer signed difference ranges from −7.279% to +3.593%. Drift can affect those single observations.

At pretrained R4 and 16K, Native/Tied E2E ratios are:

| TP | Native/Tied E2E ratio, 95% request-bootstrap CI |
|---|---|
| 1 | 0.984 [0.974, 0.994] |
| 2 | 1.003 [1.002, 1.004] |
| 4 | 0.996 [0.994, 0.998] |

These small signed differences do not provide a meaningful TP4 acceleration from explicit unrolling.
They also do not justify a claim of statistical equality or universal zero overhead.

## Headroom and research decision

The strict final verdict is `CONTROL_BLOCKED`, not a loop-specific GO.
Primary selected probes pass, but the complete diagnostic history does not satisfy uniform numerical parity.
The ordinary-graph follow-up passes without a tolerance change. The retained-graph discrepancy remains unresolved.
This limits the broad causal conclusion. It does not invalidate the native measurements or the R2 short-context timing signal.

At 16K, increasing R1→R4 does not reduce native E2E TP4 scaling within the bounded interval.
The native R4 and tied R4 GPU schedules have matching observed operation counts and order.
No measured recurrence-specific removable critical-path component supports a new distributed framework.
The study also cannot prove that every recurrence effect is absent.
The R2 1K contrast needs a targeted runtime-cause investigation before any larger fresh confirmation.
It is not permission to build a new loop-aware engine.

The gap to ideal 4× TP scaling is not an achievable speedup estimate.
No measured recurrence-specific removable milliseconds can support a new framework yet.
The experiment does not implement a new scheduler, approximate attention, weight replication, pruning, or speculative decoding.
The [prior audit](prior_overlap.md) separates this study from sparse reuse, new trained architectures, and existing loop-speculative methods.

## Correctness, quality, and limits

The forced output sequence is not an accuracy benchmark.
Full logits and hidden vectors provide functional evidence beyond those forced IDs.
KV checks sample written positions in every logical slot. They do not compare every byte of the full KV pool.
Cross-TP BF16 differences have a separate [logit audit](results/cross_TP_logits.json).
The 20 cross-TP probes retain the same four greedy predictions. Their logits are not bitwise equal.
The largest observed absolute logit difference is 0.6875. This does not prove long natural outputs match across TP degrees.

Numerical status is `PRIMARY_SELECTED_PASS / ORDINARY_NATURAL_FOLLOWUP_PASS / RETAINED_DIAGNOSTIC_UNRESOLVED`.
This is not an unconditional accuracy or universal-parity PASS.

The retained-graph natural lane contains 60 capped generations across ten configurations and three prompts.
Its summary has 30 prompt/configuration conditions, each with two technical repetitions.
All within-condition repetitions match. Native/tied equality still fails in the R4 TP4 math condition.
Eight conditions reach EOS. The other 22 conditions reach the 256-token cap.
Output length ranges from 80 to 256 tokens. Truncated reasoning is not completed benchmark evidence.
GSM8K exact match, HumanEval pass@1, and accuracy across changed R values are `NOT_RUN`.
No generated program executes in this campaign.

The main cohort has eight calibration requests. The 32K confirmation has three requests.
Neither result establishes a universal claim across looped architectures or hardware.
Only one looped pretrained architecture appears in the main target.
Both native and tied arms share weights, so this control isolates execution organization, not all effects of parameter sharing.
The genuine AR controls differ in architecture and cannot remove every cross-model confound.

Other users can use GPUs0–3 on this host. Their CPU or fabric activity remains uncontrolled.
The campaign does not lock clocks. Small configuration differences can therefore include drift.
Some teardown logs contain process-group warnings after successful result flush. Those warnings occur outside request timing and remain preserved.

## Evidence, reproduction, and next action

The [reproduction guide](reproduce.md) lists guarded commands and exact interpreter paths.
Full profiler events and parity arrays remain outside Git with hashes in the heavy-evidence manifest.
Compact raw request results, ownership, numerical checks, CSVs, figures, and code remain in this directory.
The original staged and dirty work must remain unchanged.

All 24 primary configurations produced their full clean collection.
There are 1,368 measured request repetitions and 456 request/configuration/length means.
Those counts do not create 1,368 independent prompts. Each primary comparison has eight requests, or three in the 32K confirmation.
Ten retained-graph diagnostics and four exact-primary-policy reverse-order sessions also completed.
Two ordinary-graph sessions completed the selected natural-output follow-up.
No new optimizer, full scheduler, optional 8B model, or fresh 24-request confirmation ran.

The next smallest action is a runtime-only investigation of the retained-graph discrepancy and the R2 1K phase difference.
Freeze one such cause before any larger cohort. Do not treat ordinary TP service or ideal scaling as a new loop-specific contribution.
See [machine-readable verdict](results/verdict.json), [natural audit](results/natural_discrepancy_audit.json), and [final audit](results/final_audit.json).

### Figures

- [Ouro 1K matrix](figures/ouro_matrix_P1024.png)
- [Ouro 16K matrix](figures/ouro_matrix_P16384.png)
- [32K confirmation](figures/ouro_matrix_P32704.png)
- [Native versus tied](figures/native_vs_tied.png)
- [Within-model 16K TP scaling](figures/within_model_scaling_P16384.png)
- [KV ownership](figures/kv_ownership.png)
- [Layer-normalized service](figures/layer_normalized_service.png)
- [Partial layer diagnostics](figures/layer_attention_diagnostic.png)
- [Partial outer timeline](figures/outer_timeline_partial.png)
- [Headroom and confounds](figures/headroom_and_confounds.png)
- [Actual schedule counts](figures/control_schedule_counts.png)
- [Same-policy short-prefill rechecks](figures/short_prefill_same_policy_rechecks.png)
