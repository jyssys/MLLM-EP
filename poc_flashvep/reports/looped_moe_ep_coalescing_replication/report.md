# Looped MoE EP discovery report

## Executive conclusion

The tested cross-loop coalescing and loop-conditioned replication policies do not justify a new architecture prototype.
TRUE EP4 and EP8 work on all three required checkpoints.
The distributed baseline preserves the native logits and observed greedy outputs.
The best-window coalescing model gives 1.000× versus its best generic control.
Loop-conditioned replication adds less than 1% over matched generic replication in every tested cell.
No model clears both the 1.15× realistic headroom gate and the 5% recurrence-specific increment.

This is a bounded negative result, not a proof against all recurrent EP designs.
The coalescing scheduler is a finite policy search, not a global optimum.
The replica model is optimistic and not an actual replica-aware dispatcher.
The production asynchronous replay gate remains CONDITIONAL.

## 1. Models, loaders, and actual hardware

The study uses eight NVIDIA H100 80GB HBM3 GPUs.
All GPU pairs report NV18 connectivity.
The recorded CPU affinity is 0–191 on NUMA node 0.
The experiment does not infer achieved bandwidth from that topology label.
`preflight.json` records physical IDs, UUIDs, PCI addresses, and initial process ownership.

EP4 uses physical GPUs 0–3 or 4–7.
EP8 uses physical GPUs 0–7.
Each process exposes one physical GPU as CUDA device 0.
Latency runs use one active EP group at a time.
Owned burners remain on unused devices, as requested.
Therefore the quiet lane excludes concurrent experiment groups, not all board or CPU power effects.

| Model | Learned parameters | Physical routed sites | Native logical calls | Recurrence-eligible sites |
| --- | --- | --- | --- | --- |
| IFM Ouro | 8,780,889,600 | 28 | 28 × 4 = 112 | 28 |
| IFM Huginn | 8,783,251,968 | 28 | 8 + 12 × 8 + 8 = 112 | 12 |
| Foil-1 | 553,730,048 | 3 | 1 + 16 + 1 = 18 | 1 |

IFM uses the official manifest-verified native loader and supported FA4 attention.
FA3 is unavailable in this environment.
Both sides of each EP fidelity comparison use identical FA4 kernels.
The prior FA4-versus-explicit-attention limitation remains unchanged.
Foil uses its official remote code and its Foil-1 checkpoint subfolder.
The experiment retains Foil's original FP32 RoPE buffers.

The specification's Foil count needs an accounting qualification.
The loaded checkpoint has 553,730,048 learned parameters.
It also has 9,437,184 registered FP32 buffer elements.
Their sum is 563,167,232 persistent elements, or approximately 563.2M.
The experiment does not invent parameters to reconcile the specification.

## 2. TRUE EP implementation

`runtime.py` replaces only the native routed-expert expression.
Attention, router values, normalization, residuals, recurrence, and output heads remain native.
Experts have stable owners: `expert_id % EP_degree`.
IFM's 25 experts use ragged owner counts, not padded logical experts.
Foil's untied per-pass attention never enters expert coalescing.
Huginn prelude and coda remain one-off model work.

Every MoE invocation performs actual NCCL communication:

1. Partition source rows by rank.
2. All-Gather exact routing counts.
3. All-to-All BF16 expert inputs.
4. Execute owner-local native expert GEMMs.
5. All-to-All expert outputs.
6. Apply original top-2 scores and restore source rows with All-Reduce.

Dense parameters remain replicated.
Expert parameters become owner-only storage before measurements.
Every load checks exact native weight subsets.
Every completed worker checks retained expert storage identities.
This is TRUE EP, but not optimized DeepEP production serving.
Its count synchronization and source restoration can dominate small work units.
Those generic bridge costs are not a new loop-specific contribution.

## 3. Measured latency

The following complete-forward values use prefix 128.
Each B1/B2/B4 cell has three independent process starts.
Each start uses ten warmups and twenty timed repetitions.
The 95% bootstrap interval resamples process-start means, not individual kernels.
Starts 0/1/2 use public prompt-cohort offsets 0/4/8.
The interval therefore combines restart and prompt-cohort variation.
It does not isolate GPU-group variance.
EP4 and EP8 use the same prompt IDs at each corresponding start.
Three starts give limited precision for hardware variability.
Prefixes 1024 and 2048 have one start and exploratory repetition intervals.

| Model | EP | B1 ms [95% CI] | B2 ms | B4 ms |
| --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 339.95 [325.55, 355.86] | 343.39 | 346.58 |
| IFM Ouro | 8 | 329.00 [323.73, 334.89] | 330.81 | 335.45 |
| IFM Huginn | 4 | 323.12 [320.58, 327.09] | 324.31 | 331.81 |
| IFM Huginn | 8 | 331.97 [330.19, 334.83] | 335.54 | 338.93 |
| Foil-1 | 4 | 50.23 [49.26, 50.86] | 57.83 | 59.08 |
| Foil-1 | 8 | 50.80 [49.71, 51.85] | 51.14 | 51.64 |

Units are milliseconds.
`summary/baseline_forward.csv` contains all 54 model/degree/batch/prefix cells.
The duration includes attention, experts, metadata, restoration, and completion synchronization.
It is not a sum of kernel service times.

### Warmed generation control

The following B1 measurements use prefix 128 and 128 actual greedy output steps.
EOS is disabled for this hardware workload.
The path recomputes the full prefix on every step.
These values are not cached-decode TPOT or natural-completion benchmark accuracy.
Each cell uses three warmed full generations after one full shape-preparation generation.
The interval resamples those three serial generations from one process start.

| Model | EP | B1 E2E s [95% CI] | TTFT ms | TPOT ms | Output tokens/s |
| --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 41.640 [41.311, 42.240] | 365.24 | 325.00 | 3.07 |
| IFM Ouro | 8 | 42.198 [42.161, 42.269] | 368.14 | 329.37 | 3.03 |
| IFM Huginn | 4 | 41.990 [41.612, 42.290] | 349.59 | 327.88 | 3.05 |
| IFM Huginn | 8 | 42.759 [42.588, 42.941] | 377.31 | 333.71 | 2.99 |
| Foil-1 | 4 | 6.467 [6.457, 6.478] | 92.14 | 50.19 | 19.79 |
| Foil-1 | 8 | 6.600 [6.600, 6.602] | 82.12 | 51.33 | 19.39 |

The first large matrix also covers B1/B2/B4 at output caps 32 and 128.
Those values remain in `summary/generation.csv`.
Their first-occurrence shapes were not fully warmed.
They are exploratory measurements, not the steady-state headline.

### Component evidence

The following separate observer uses B1 and prefix 128.
All components come from one cumulative-critical rank per forward.
Columns contain CUDA-event API spans, including host launch gaps.
The selected recurrent capture site also clones diagnostic expert inputs.
Its expert span therefore includes that observer copy.
They are not kernel-only service times or a clean E2E percentage partition.

| Model | EP | Clean ms | Observer ms | Pack | Dispatch | Group | Expert | Combine | Restore | Residual ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 325.55 | 386.00 | 79.55 | 15.89 | 12.60 | 46.37 | 30.90 | 36.42 | 164.27 |
| IFM Ouro | 8 | 334.89 | 390.45 | 84.83 | 18.59 | 12.51 | 43.09 | 35.03 | 35.20 | 161.21 |
| IFM Huginn | 4 | 321.68 | 386.19 | 72.18 | 16.45 | 13.96 | 47.79 | 31.40 | 33.19 | 171.21 |
| IFM Huginn | 8 | 330.19 | 395.76 | 77.09 | 16.90 | 14.07 | 45.93 | 31.17 | 37.50 | 173.10 |
| Foil-1 | 4 | 50.58 | 59.35 | 12.33 | 3.24 | 1.83 | 3.41 | 3.61 | 3.75 | 31.19 |
| Foil-1 | 8 | 50.83 | 59.51 | 10.80 | 1.90 | 1.93 | 3.31 | 3.15 | 5.17 | 33.26 |

All values are milliseconds.
Residual includes non-expert execution, observer overhead, and unresolved gaps.
Rank selection and host overlap prevent an exact clean critical-path attribution.
CPU synchronization and rank waiting remain PARTIAL.
`summary/communication_accounting.csv` separates logical tensor bytes from unmeasured wire bytes.

## 4. Fidelity and quality

| Model | EP | Checks passed | Max logit relative L2 | Max |ΔNLL| | Top-1 equal / positions | Native greedy |
| --- | --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| IFM Ouro | 8 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| IFM Huginn | 4 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| IFM Huginn | 8 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| Foil-1 | 4 | 105/105 | 0.0 | 0.0 | 30733/30733 | 8/8 exact |
| Foil-1 | 8 | 105/105 | 0.0 | 0.0 | 30733/30733 | 8/8 exact |

Each cell has 32 language examples, 32 GSM8K examples, 32 MATH-500 examples, and nine hardware-shape probes.
The teacher-forced examples use a 128-token input cap.
The shape probes cover B1/B2/B4 and 128/1024/2048 tokens.
They are not 105 free-running mathematical answers.
The table does not report downstream answer accuracy.
No candidate A/B implementation exists, so candidate answer accuracy is NOT_RUN.

Huginn's official initializer depends on the batch index.
Native B4 therefore differs from four independent native B1 initializations.
Every native-versus-EP check preserves batch geometry.
The independent-request observer compares against sequential B1 references instead.
The study does not use B4-versus-four-B1 equality as a quality claim.

## 5. Track A: ready coalescing

The diagnostic observer runs independent requests through unchanged native attention and recurrence.
Only one coordinator issues distributed collectives.
CUDA events protect input readiness and output lifetime.
All ranks vote on device readiness before a recorded candidate becomes eligible.
This avoids comparing unsynchronized absolute CUDA clocks across GPUs.
The observer never coalesces expert work.

| Model | EP | B1 eligible calls | B2 eligible calls | B4 eligible / decisions | B4 exact output |
| --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 0 | 0 | 0/7168 | 2/2 arrival cases exact |
| IFM Ouro | 8 | 0 | 0 | 0/7168 | 2/2 arrival cases exact |
| IFM Huginn | 4 | 0 | 0 | 0/7168 | 2/2 arrival cases exact |
| IFM Huginn | 8 | 0 | 0 | 0/7168 | 2/2 arrival cases exact |
| Foil-1 | 4 | 0 | 0 | 227/1152 | 2/2 arrival cases exact |
| Foil-1 | 8 | 0 | 0 | 51/1152 | 2/2 arrival cases exact |

Each entry combines synchronized and staggered arrivals over eight generated steps.
The denominator counts observed MoE scheduling decisions, not tokens or unique experts.
An eligible call shares at least one physical expert with another ready request at a different pass.
Observed eligibility is not the fraction of work that can be removed.
The control broadcasts and readiness votes perturb queue timing.
These snapshots are sampled evidence, not an exhaustive opportunity ceiling.

B1 has no independent cross-request frontier.
The tested B2 observer also finds no cross-pass eligible calls.
Foil B4 shows actual cross-pass opportunities.
IFM's longer repeated layer chains produce little aligned physical-site readiness in this scheduler.
The absence of observed pairs is specific to this workload and coordination policy.

The O1 model preserves each request's causal layer and recurrent dependencies.
It charges measured complete expert-service costs and explicit waiting windows.
It compares generic same-pass and cross-pass policies within one executor.
The tested windows are 0/50/100/250/500 microseconds and 1/2/5/10 milliseconds.

The same-window comparison can show apparent improvements up to 1.359×.
However, the best generic window is zero in the tested cases.
Every matched best-cross-versus-best-generic ratio is 1.000×.
Waiting creates some opportunities but costs more than their measured batching benefit.

The 32-step O2-style replay also gives 1.000× against its best generic window.
That replay clones a representative request template.
It uses separately measured cohort scaling and a causal decision-cost lookup.
Huginn's batch-indexed initializer further limits that cohort-scaling equivalence.
The replay is not a tight bound for arbitrary independent-request trajectories.
It does not constitute independently validated asynchronous serving.
`oracle/replay_validation.csv` records held-out start drift and in-sample fit separately.
`oracle/generation_validation.csv` records the remaining generation-model mismatch.
No modeled ratio receives an M-level measured speedup label.

## 6. Track B: replicated experts

The study calibrates static replica sets on eight examples per domain.
The remaining examples and hardware cohorts do not tune those sets.
Generic hot, generic cost-aware, and loop-conditioned searches receive the same observations and copy budgets.
All searches use the same bounded placement proxy.
Runtime assignment uses exact current demand and only resident copies.
No expert weight moves between passes.

| Model | EP | A vs best generic | B vs static, max | B vs generic, max | B load-only ideal |
| --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 1.000× | 1.068× | 1.002× | 1.083× |
| IFM Ouro | 8 | 1.000× | 1.094× | 1.002× | 1.112× |
| IFM Huginn | 4 | 1.000× | 1.055× | 1.004× | 1.074× |
| IFM Huginn | 8 | 1.000× | 1.067× | 1.007× | 1.092× |
| Foil-1 | 4 | 1.000× | 1.006× | 1.001× | 1.016× |
| Foil-1 | 8 | 1.000× | 1.009× | 1.008× | 1.022× |

These are optimistic whole-forward projections, not actual optimized E2E timings.
The projection maps measured expert spans linearly to maximum rank-row load.
Those API spans can include observer copies and launch gaps.
The projection is not a precise expert-kernel critical-path model.
Communication remains unchanged.
The conservative sensitivity halves the computed saving and charges selector CPU time.
The omniscient column ignores replica residency and is not implementable.
It is a load-only diagnostic, not an absolute bound on all communication optimizations.

The loop-conditioned increment over generic replication remains below 1%.
It falls below the 5% recurrence-specific threshold.
With the same resident set and exact current routing, pass identity adds no information to this selector.
The same-resident-set control therefore yields exactly the same integer assignment.
This is a property of the implemented selector, not proof about all future selectors.

Replica memory is not free.
The study allocates and hashes real native BF16 copies at one and EP-degree extra copies per recurrent site.
Those controls measure initialization time and retained per-rank memory.
They do not execute replica-aware routing.
`summary/replica_memory.csv` and `summary/hbm.csv` contain actual bytes.

## 7. C0: overlap and routing lookahead

The operator control overlaps one independent ready dispatch with one native-weight expert GEMM.
Sequential and two-stream runs use identical tensors.
The final quality controls verify exact expert outputs.
The earlier operator controls did not record that output check.
It measures twenty repetitions after ten warmups.
CUPTI timelines distinguish NCCL kernels from other kernels.
`summary/kernel_overlap.csv` reports union and intersection durations.
`summary/overlap_operator.csv` reports complete operator wall-time ratios.
Concurrent kernels alone do not establish request-level acceleration.
Request critical-path saving is NOT_MEASURED.

Previous-pass routing recall uses up to 32 sampled positions per call.
Precision equals recall because both selections contain top-2 experts.
The diagnostic never predispatches an unknown next-pass activation.
False positives can waste replica memory even when expert sets overlap.
No learned gate or predictor is introduced.

## 8. Research decision and limitations

P0 passes all six native/TRUE EP cells.
P1 passes bounded characterization, with sampled-ready and critical-path limitations.
P2 provides calibrated policy estimates, but production O2 validation remains CONDITIONAL.
P3 optimized dispatch is NOT_RUN because neither track clears the research gate.
The tested designs receive NO_GO_A and NO_GO_B.
The study makes no universal impossibility claim about recurrent EP.

The strongest practical next step is generic EP backend optimization, not a named loop architecture.
A future recurrent proposal must first show a ready independent frontier and a gain beyond that optimized generic executor.
It must also validate natural answer accuracy and full generated-request timing.
No skipping, output reuse, gate change, quantization, or loop pipeline was implemented.
