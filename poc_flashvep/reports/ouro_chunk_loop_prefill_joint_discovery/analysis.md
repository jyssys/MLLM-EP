# Ouro chunk–loop prefill discovery

## Decision

**CONDITIONAL_FIXED_QUANTUM_ONLY. Do not promote this result to an Adaptive Joint scheduler or a multi-GPU method.**

G2 is correct in isolated numerical probes. It changes the frequency and duration of prefill interruptions.
It does not pass the registered TTFT/ITL-p95 continuation gate against tuned Atomic C1024.
It retains a useful, narrower observation about rare stalls.
This observation does not establish a new serving architecture or a paper-level joint-policy result.

The paused native-vLLM Loop PP program remained paused.
This study changed no PP runtime, TP runtime, attention kernel, pretrained weight, or model exit policy.

## What actually ran

All inference ran on physical GPU4, an H100 80GB HBM3.
The model was base `ByteDance/Ouro-1.4B`, BF16, at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Every prompt token and generated-token forward used Static R4.
There are 24 shared physical layers and 96 independent depth/layer KV namespaces.

The runtime was the separate RLT checkout at `4a350344027960a48384626aa646f1765fda6d4f`.
This is a standalone recurrent engine. It is not native vLLM with an added plugin.
The patch changed three files, with 23 additions and nine removals.
It reused the existing `ExecutionBatch.loop_count` primitive and preserved native KV metadata and model operations.
See [implementation_scope.md](implementation_scope.md) for pinned source links and the exact dependency contract.

The completed evidence contains:

- A stock calibration sweep with 12 serving cells.
- Two discovery grids with 64 cells each. One used Triton. One used FlashAttention4.
- A corrected FA4 confirmation with 100 measured cells and 30 calibration warm-up cells.
- Ten paired held-out cohorts per shortlisted configuration, across three fresh processes.
- Prefill-only, decode-only, observer, graph-off, and staged-G4 controls.
- 43 pretrained numerical probes, including an 8K Atomic/G2 comparison.
- A physical Nsight timeline, with 741,806 kernel records and 1,033 matched scheduler submissions.

`results_raw.csv` contains 326 completed serving cells, including warm-up and rejected cold-order measurements.
Its denominator is not 326 independent held-out cohorts.
`experiment_matrix.csv` records completed, conditional, and excluded work separately.

## Frozen workload and clocks

The token-chunk menu was 128, 256, 512, and 1024.
The recurrent-group menu was 1, 2, and 4.
The discovery grid used prompt bins 2K/8K and request concurrency 4/8.
The actual long prompt lengths were 2,033 and 8,177 tokens.
`C` means token-chunk capacity. `N` or `concurrency` means simultaneous requests.

The manifest froze natural mathematical problem text, input IDs, request IDs, sampling, and thresholds before the candidate grid.
The long prompts concatenate public problem text. They are not original conversational traffic.
Background requests reached decode before the client scheduled the long request 20 ms later.
Every measured interference case retained live background decode at the first long-request GPU submission.
This is a controlled causal arrival rule, not a pre-existing external arrival trace.

Generation was free-running and greedy. EOS was permitted. The output cap was 128 tokens.
No output IDs were forced. No hardware replay supplied headline results.
All 600 corrected-confirmation request executions reached the cap. None ended with EOS.
These are bounded natural-generation serving measurements, not complete reasoning or benchmark accuracy results.

Client times use the in-process API clock, from scheduled arrival to observed token or completion.
They include ordinary queueing, graph activity, sampling, and output handling.
They do not include a network frontend.
The separate `interference_span_s` excludes priming. It is not full cohort makespan.

The frozen SLO required TTFT <= 5 s.
It also required maximum observed request ITL <= 100 ms.
The reported SLO-qualified completion rate divides qualifying completions by drained cohort duration.
It is not steady-state, open-loop SLO goodput.
No threshold changed after candidate measurements.

## Corrected confirmation results

The table reports means of per-cohort statistics, not pooled-token quantiles.
Each row contains ten held-out cohorts.
All rows use C1024 and the same backend, token budget, output cap, and KV pool.

| Prompt / concurrency | Policy | G | Long TTFT ms | ITL p95 ms | ITL p99 ms | SLO-qualified req/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2K / 4 | Atomic | 4 | 235.083 | 26.753 | 79.200 | 0.251 |
| 2K / 4 | Split | 1 | 290.587 | 27.512 | 108.761 | 0.269 |
| 2K / 4 | Mixed | 1 | 337.428 | 25.724 | 121.210 | 0.275 |
| 2K / 4 | Grouped | 2 | 258.736 | 26.174 | 138.205 | 0.275 |
| 8K / 8 | Atomic | 4 | 924.384 | 38.510 | 364.162 | 0.169 |
| 8K / 8 | Split | 1 | 1452.585 | 137.166 | 173.306 | 0.148 |
| 8K / 8 | Mixed | 1 | 1648.820 | 122.176 | 128.611 | 0.123 |
| 8K / 8 | Grouped | 2 | 1116.074 | 139.216 | 205.288 | 0.159 |

The full table includes Mixed C128, E2E, TPOT, throughput, and allocated memory.
See [benchmark_summary.md](benchmark_summary.md) and `warm_confirmation_summary.csv`.

The next table compares G2 with Atomic C1024.
Positive improvement means lower latency. A speedup below 1.000× means a slowdown.
The intervals use 5,000 paired bootstrap draws, with process-start blocks and cohort resampling.
No token or kernel is an independent bootstrap sample.

| Shape | Metric | G2 speedup | Improvement % [95% CI] |
| --- | --- | ---: | --- |
| 2K / 4 | Long TTFT | 0.909× | -10.06 [-18.56, -1.07] |
| 2K / 4 | ITL p95 | 1.022× | 2.16 [-0.55, 6.25] |
| 8K / 8 | Long TTFT | 0.828× | -20.74 [-36.74, -10.12] |
| 8K / 8 | ITL p95 | 0.277× | -261.51 [-288.88, -243.74] |
| 8K / 8 | ITL p99 | 1.774× | 43.63 [34.61, 49.12] |
| 8K / 8 | Maximum ITL | 2.155× | 53.60 [45.65, 59.38] |

G2 is not uniformly dominated on every possible tail-latency axis.
At 8K/concurrency8, it reduces p99 and maximum ITL relative to Atomic.
However, it worsens TTFT and p95. Mixed C1024 has an even lower p99 of 128.611 ms, but worse TTFT.
The result is a trade-off between existing endpoint quanta, not a confirmed joint-policy gain.

The registered TTFT regression allowance was 5%.
The observed TTFT regressions exceed that allowance.
The p95 improvement at 2K is small, and its interval includes no improvement.
The 8K p95 result is substantially worse.
Therefore, the practical Phase 2 continuation gate did not pass.

## Why grouping changes the result

Atomic processes four recurrent loops before returning control.
G2 introduces a boundary after two loops.
In the matched 2K heavy trace, Atomic used two long-prompt compute quanta. G2 used four.
Their median measured CUDA-event intervals were 113.885 ms and 54.075 ms, respectively.
Those intervals include staging and waits. They are not pure kernel time.

Shorter quanta can reduce an isolated large decode stall.
More frequent stalls can also move many intervals into the p95 tail.
Thus a policy can improve maximum ITL while worsening p95.
The raw interval counts remain in `ITL_stall_counts.csv`.
For 8K/concurrency8, Atomic recorded 253 intervals above 100 ms among 8,890 observed background intervals.
G2 recorded 483 among 8,866 intervals.
Intervals above 200 ms fell from 194 to 104.
These are descriptive counts, not independent statistical samples.
Percentile changes alone do not prove more successful SLO completions.

G2 also requires staged hidden state and more scheduler submissions than Atomic.
The source and traces support those costs. They do not isolate one causal percentage for host scheduling.
Mixed further changes batch packing and uses native eager fallbacks for incompatible graph shapes.
Its results cannot be attributed to G alone.
The staged-G4 control separates part of the prelude/staging difference, but contains only one diagnostic cohort.

## Correctness and memory

All 31 non-reference pretrained numerical comparisons passed exact token, depth, hidden, logit, and prompt-KV checks.
The probes covered both backends and all 96 KV namespaces.
There were zero failures among 1,551 issued prompt-work checks.
Each check matched four times the actual prompt length.
These scheduler counts are corroborated by separate GPU numerical probes. They are not independent kernel counters.

The held-out batched serving lane is more limited.
Grouped and Split matched complete capped token sequences in 112/120 paired request executions.
Depth sequences matched in 120/120. All outputs had the same capped length.
We did not isolate every batched BF16 difference from a possible implementation issue.
Full GSM8K/MATH accuracy and quality noninferiority are **NOT_RUN / NOT_ESTABLISHED**.
See [correctness.md](correctness.md).

The common KV reserve was 12 GiB.
The staged activation buffer was 16 MiB at concurrency4 and 32 MiB at concurrency8 for G2 C1024.
Atomic had no separate staged-prefill activation buffer.
Mean peak allocated HBM was 14.898/15.293 GiB for Atomic/G2 at 2K/concurrency4.
It was 15.368/15.653 GiB at 8K/concurrency8.
These total peaks include allocator and graph state. Their difference is not pure activation overhead.
Graph-private memory and exact padded graph rows are not separately resolved.

## Measurement limitations and rejected evidence

An initial confirmation gave Atomic a one-sided first-use FA4 JIT cost.
We stopped that owned process, preserved its data, and repeated with equal calibration warm-up per configuration.
The corrected contract did not change held-out inputs, SLO thresholds, or decision margins.

New held-out shapes can still trigger JIT or graph work during requests.
Policies ran in a fixed order. Atomic was first within a process.
Full-cohort throughput and maximum-stall comparisons can retain this order effect.
We do not credit the apparent 2K throughput increase as architecture speedup.
The Nsight capture was a heavy, cold diagnostic. Its wall time is not headline serving latency.
The observer control used one cohort and does not prove negligible observer overhead.

The discovery matrix used all four shapes. The repeated shortlist used two shapes only.
The confidence intervals remain exploratory because there are only three process-start blocks.
We did not run steady open-loop Poisson traffic, a full quality anchor, or an adaptive baseline.
These limits prevent a joint-policy GO claim even where a tail statistic improves.

## Direct answers and next action

1. **Is G2 useful beyond G1/G4?** It supplies a distinct fixed-quantum tail trade-off. It does not pass the registered joint-scheduling gate.
2. **Is this a new recurrent mechanism?** No. The small adapter fills an interior quantum between existing RLT endpoints.
3. **Does Adaptive Joint beat Adaptive C-only/G-only or Mixed?** NOT_RUN. No such claim is supported.
4. **What memory does it add?** The measured staged buffer adds 16/32 MiB in the shortlisted concurrency4/8 cells. KV reserve remains unchanged.
5. **Is multi-GPU portability measured?** No. Phase 4 is NOT_RUN because the single-GPU gate did not pass.
6. **Is there enough evidence for a new research method?** No. Keep the exact adapter as an ablation, not a new serving architecture.

The smallest optional follow-up is a counterbalanced, fully warmed tail-SLO test at 8K/concurrency8.
It should use complete outputs and compare C/G choices with matched uninterrupted service duration.
It should first test whether the rare-stall benefit survives tuned smaller-C Atomic and stock Mixed.
Do not implement an Adaptive Joint controller before that result.

The implementation, raw measurements, failed lane, manifests, source pins, and reproduction commands remain available.
GPU4 returned to its burner after the final probe. GPUs5–7 retained their existing burners.
