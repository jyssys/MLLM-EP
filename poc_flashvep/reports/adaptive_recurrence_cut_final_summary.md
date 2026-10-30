# Dynamic A PoC: Final Evidence and Decision

Overall verdict: **CONDITIONAL**.
Execution status: **GATE-STOPPED finite mechanism PoC**.
This is not a completed online Dynamic Cut or nine-arm serving evaluation.

The new multi-chunk Loop migration works in the tested configurations.
Ouro shows load-dependent fixed-cut trade-offs.
Huginn does not show a reliable benefit from the frozen regime-dependent selector.
Neither model has a measured Dynamic SLO-goodput gain in this study.
The evidence does not support a GO for a new Loop-specific serving method.

## 1. Code and inherited results

Code and raw-evidence release: `71453683a69d600a7cabc667386df4dd1b622f7a`.
Branch: `flashvep/adaptive-recurrence-cut-poc`.
The final publication adds reports and Git-index scope metadata.
It does not change the model execution paths from that evidence release.

| Inherited component | Revision |
|---|---|
| Numerical correction | `3b7a722898dab5f6dd797b474c1ff461659d8fa1` |
| RCC Stage 1 base | `b53a8fed36a1706ac74754b53753f20edb8b913e` |
| Original A/B/C results | `b82eabd20a4ec168a5ad9a119c8401f97a565e78` |
| B Serving results | `fea20edef840ed357ee0493c2f0dd69668c66c1f` |
| Private RLT HEAD | `6b164d3844d35ab8e753df1864f0c87dc180401a` |

The preserved RLT diff SHA-256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
All 1,637 locked prior source files remain unchanged.
All archived map runs use the same retained-state and cut-connector source hashes.
Each run preserves its actual harness version separately.
The prior RCC and A/B/C conclusions remain unchanged.

The experiment uses Torch 2.11.0+cu129, FA4, NIXL/UCX, and the existing pinned environment.
It does not replace the engine with LoopServing or production vLLM.

## 2. Models and computation

| Model | Checkpoint revision | Depth | Numerical correction |
|---|---|---:|---|
| Ouro Base 1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | R4 | `down_norm` |
| Huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | R32 | `down_norm_adapter` |

All requests retain BF16 weights and KV.
Every generated token retains the full prescribed recurrent depth.
No early exit, approximation, training, quantization, or weight change is used.
Native Mixed and Prefill/Decode shared-GEMM co-batching are off.
The execution guard rejects such a combined recurrent batch.

## 3. Implementation and correctness

The old A connector accepts only a whole-prompt single chunk.
The new connector retains absolute prompt-state rows across multiple chunks.
It blocks P execution above the committed request cut.
It transfers valid KV and the intermediate state for every prompt position.
Huginn's carrier retains both recurrent and injection state.
The source buffers remain leased until transfer acknowledgement.

A Cut0 target performs full Prefill and Decode on D while P remains resident.
Full Cut R performs all Prefill recurrence on P.
Intermediate cuts execute the remaining recurrence on D.
All subsequent Decode tokens execute all R loops on D.
The adapter supports up to 4,096 retained prompt rows.
That adapter limit is not Ouro's model context limit.

Both models passed finite output, Prompt KV, hidden-state, and recurrent-work checks.
The tests include every initial cut at length 128.
They include midpoint migration at lengths 512 and 1,536.
The latter Ouro test exceeds the original four-by-256 staging ring.
The native streaming PD_G1 path also passed a separate two-request multi-chunk check.

| Evidence | Actual result |
|---|---|
| Completed GPU run namespaces | 36 |
| Preserved failed development namespaces | 4 |
| Measured requests across all completed scopes | 1,428 |
| Fixed-cut/control map cells | 151 |
| Measured map requests | 1,208 |
| Full-depth map work checks | 151/151 |
| Non-reference target output comparisons | 828/828 exact |
| Native streaming versus snapshot output comparisons | 48/48 exact |
| Snapshot payload/counter reconciliation | 145/145 cells |
| Profiler rank traces | 8 |

These denominators count repeated mechanism experiments, not distinct benchmark problems.
The same small diagnostic prompt sets recur across starts and policies.
The results are not a new 200/400-request universal migration qualification.
They do not establish original-HF losslessness or GSM8K/MATH accuracy.
The inherited original-HF fidelity limits remain separate.

## 4. GPU isolation

Correctness probes use independent pairs across GPU0–7.
Clean timing runs use one 1P+1D pair at a time.
The map crosses GPU0/1, GPU4/5, and GPU6/7.
Every worker map records the physical ID and UUID.
The topology reports NV18 connectivity.
The resource manager checked ownership before it stopped burner sessions.
Other users' processes and work files were not terminated or removed.

The study does not claim complete CPU-host exclusivity.
It avoids concurrent GPU experiments and owned burners during clean measurements.
Burners on physical GPU0–7 were restored after the final GPU run.
Four-GPU and eight-GPU pooled serving are NOT_RUN.

## 5. Static cut findings

The decisive map uses length 512, chunk 128, and token budget 128.
It has eight measured requests per cell and at most four compute slots per worker.
Three independent starts cover every initial cut and load profile.
Candidate order changes across starts. Full-Cut parity anchors run first.
Thus the control order is not completely counterbalanced.

Ouro's lowest mean E2E point changes with the profile.
Cut2 is best in the short-output Prefill-pressure profile.
Cut1 is best in the balanced profile.
Full Cut4 is best when two background Decode requests are already active.
The complete cut map includes confidence intervals and every measured cut.

Huginn Cut28 has a slightly better P-pressure point mean than Full Cut32.
That difference is small relative to variation.
Full Cut32 has the lowest mean in the other two profiles.

The single-start L1536 subset reinforces the trade-off for Ouro.
Under P-pressure, Ouro Cut2 versus Cut4 mean E2E is 3,713 versus 4,619 milliseconds.
Under loaded D, Cut4 versus Cut2 mean E2E is 4,199 versus 5,150 milliseconds.
For Huginn, Full Cut32 is best among the three tested long-context cuts in both profiles.
These long-subset results have no qualifying confidence interval.
Their chunk size is 256, so they are not a pure input-length ablation.

## 6. Dynamic headroom is not established

A start-0 calibration freezes one static target cut and one cut per known profile.
The analysis evaluates those frozen choices on independent starts 1, 2, and 3.
It composes complete homogeneous cohort measurements.
It does not choose independent per-request latency minima.
It also does not execute live Dynamic Cut.

| Model | Frozen static cut | Held-out composed mean E2E reduction | Paired-start bootstrap 95% CI |
|---|---:|---:|---|
| Ouro R4 | 1 | 4.98% | [0.90%, 8.24%] |
| Huginn R32 | 32 | -3.62% | [-4.49%, -2.81%] |

Evidence class: `OFFLINE_FIXED_REGIME_COMPOSITION`.
Negative reduction means worse latency.
The prompt IDs remain fixed. These are not held-out domain or benchmark-accuracy results.

Ouro has modest restricted headroom worth preserving as evidence.
Huginn's frozen selector is worse than its static comparator.
Neither result measures sustainable SLO-goodput.
The queue simulator and its out-of-sample prediction tests are NOT_RUN.
The Dynamic Oracle is also NOT_RUN.
The saved 10% goodput expansion gate therefore did not qualify.
This is not proof that every future Dynamic Cut policy must fail.

## 7. Transfer policy is an important generic control

The main cut map uses a completed-prompt snapshot at Full Cut R.
The existing native PD_G1 path streams completed full-depth token chunks.
These are distinct handoff policies with the same model computation.
The study measures native streaming separately instead of hiding that difference.

Snapshot speedups relative to native are 1.270x for Ouro and 1.222x for Huginn in the single-start P-pressure control.
Native streaming is faster in the loaded-D control of that start.
The experiment has one paired start per model/profile and no qualifying confidence interval.
The result does not establish a generally better transfer policy.
It does show why handoff changes must not be called recurrent-partitioning gains.

The exact NIXL payloads and completion counts are retained.
Control bytes and pure KV DMA time are NOT_MEASURED.
Host cumulative transfer observations are not exclusive GPU DMA durations.

## 8. GPU work and E2E percentages

The profiler-only midpoint runs move kernel work from P to D.
They do not remove full-depth computation.
Their summed per-device kernel busy times remain similar to the Full-Cut profiles.
Those sums are not the multi-GPU E2E critical path.
The two-request profiles capture 0.00 ms of simultaneous P/D kernel execution.
That result does not describe clean eight-request pipeline overlap.

The profile records actual GPU kernel counts and CPU matrix shapes.
Direct External-ID mapping separates some FFN, Attention-projection, and adapter GEMMs.
Graph replay leaves some module mappings incomplete.
The detailed profiling report states those limits.

Both E2E percentage files cover all measured map requests, including background requests.
They separate arrival delivery, reservation, P observation, handoff, D continuation, and generation observations.
Their per-request intervals are exclusive and cover the E2E denominator.
They are not exclusive dispatch/GEMM/DMA GPU percentages.
No overlapping device durations are added to create a speedup claim.

## 9. Token versus Loop versus Layer

Actual Token migration and intra-loop logical-layer migration are NOT_IMPLEMENTED.
The six-arm static/dynamic partition-axis comparison is NOT_RUN.
The current study cannot claim that Loop-Dynamic beats Token-Dynamic or Layer-Dynamic.

A generic same-model logical-layer action space contains the matching recurrent boundaries.
At the same boundary, it can use the same physical weights and state-transfer protocol.
Ouro's matching boundary includes its recurrent norm and required transition operations.
Huginn's matching boundary preserves its injection carrier and valid Prelude/Coda state.
Those matching cases are `EQUIVALENT_MECHANISM` or `EQUIVALENT_LAYER`.

Each current worker retains the complete physical weights.
The measured storage is 2,869,305,346 bytes for Ouro and 7,129,953,600 bytes for Huginn.
A generic same-model implementation can retain the same weight aliases.
It must not be forced to allocate separate weights for every unfolded logical layer.
No distinct weight-residency saving is demonstrated here.

The partition-axis-alone novelty claim is `NO_GO_LOOP_SPECIFIC_NOVELTY`.
This is not a proof that practical Loop-aware control or serialization advantages cannot exist.
Those advantages remain unmeasured against a complete generic implementation.

## 10. Main serving work that did not run

The primary metric remains maximum measured stable TTFT-plus-average-TPOT SLO-goodput.
The required request attainment is at least 90% on a declared finite rate grid.
The study did not calibrate or measure that main W1 metric.
It did not perform a Poisson sustainable-rate sweep or claim capacity from finite throughput.

The W1 calibration and test manifests each contain 200 complete ShareGPT prompts.
They have disjoint prompt hashes and common-support lengths valid for Huginn.
Their controlled output targets average 448 tokens.
That target is not an observed generation-length mean.
No Azure prompt text or production Azure arrival trace is claimed.

W1-LG, W1-Natural, W2, W3, W4, the full nine-arm comparison, and 4/8GPU serving are NOT_RUN.
The 270-row main status matrix records every model/method/budget/workload cell explicitly.
Its missing values are NOT_MEASURED, not estimates or zeros.

## 11. Stage decisions

| Stage | Ouro | Huginn |
|---|---|---|
| 0: migration correctness | Finite GPU qualification PASS | Finite GPU qualification PASS |
| 1: static map | Complete at declared pilot scope | Complete at declared pilot scope |
| 2: Dynamic headroom | CONDITIONAL, no qualified goodput gate | Frozen selector NO-GO, general oracle NOT_RUN |
| 3: granularity | Matching boundary EQUIVALENT_LAYER, full comparison NOT_RUN | Matching boundary EQUIVALENT_LAYER, full comparison NOT_RUN |
| 4: online policy | NOT_RUN | NOT_RUN |

No online algorithm, admission recheck, or safe cut-renegotiation fallback is claimed.
The current source commits the cut before P starts.
The online-policy report specifies the missing safety protocol.

## 12. Minimum next evidence before expansion

1. Tune strong native and snapshot PD controls with equal chunk and pure-Prefill token-budget budgets.
2. Obtain clean service and queue traces for a contention-aware model.
3. Measure out-of-sample fixed-cut prediction error before a Dynamic Oracle claim.
4. Implement and qualify Token and internal Layer migration before an axis-superiority claim.
5. Calibrate main SLOs on separate requests before an online goodput evaluation.

The current no-Mixed map does not represent optimized production Mixed serving.
A smaller token chunk need not imply a smaller GEMM when multiple requests can share Prefill execution.
These controls are necessary before further architecture claims.

## 13. Reports and raw evidence

- [Static map](adaptive_recurrence_cut_static_map.md)
- [Oracle and headroom limits](adaptive_recurrence_cut_oracle_headroom.md)
- [Token/Loop/Layer analysis](adaptive_recurrence_cut_token_loop_layer_comparison.md)
- [Online-policy status](adaptive_recurrence_cut_online_policy_results.md)
- [Correctness contract](adaptive_recurrence_cut/correctness_and_execution_contract.md)
- [GPU profiling and accounting limits](adaptive_recurrence_cut/profiling_and_limits.md)
- [Reproduction commands](adaptive_recurrence_cut/reproduce.md)

Raw data reside in `adaptive_recurrence_cut/raw/`.
The command ledger is `adaptive_recurrence_cut/commands.jsonl`.
The experiment matrix preserves completed, failed, and unexecuted configurations.
The final validator checks 36 completed namespaces and preserves all four failed namespaces.
The Git index guard preserves every unrelated staged entry.
