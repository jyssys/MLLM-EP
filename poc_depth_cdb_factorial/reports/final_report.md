# CDB Refill × Cross-GPU Hindsight: final report

## Verdict

The final verdict is **`QUALITY_CONFOUNDED`**.

- The placement and refill switches are orthogonal in code. Their measured performance effects are not reliably additive.
- CDB is LOCAL_ONLY. The outer controller assigns requests across four independent replicas before prefill.
- At C64, hindsight gives 0.997× without refill and 1.038× with refill. Both five-repeat 95% intervals include 1.000×.
- The selected synchronous last_exited refill path fragments active core batches. It is not the paper's best shared-KV asynchronous configuration.
- The bounded speed repeats preserve all 2480 paired token and depth traces within each configuration. Full-answer parity between configurations does not hold.
- A1 and A4 each score GSM8K 97/128 versus A0 101/128. Each paired difference is −3.125 pp, with 95% CI [−6.25, −0.78125] pp.
- Natural-answer diagnostics retain conditional timing headroom. Ratios of four summed C64 serving intervals are 1.288× without refill and 1.571× with refill.
- Those diagnostics change generated work and answers. They do not isolate depth information from future length or establish matched quality.
- The current evidence does not justify RL training. It also does not prove that all natural-workload placement headroom is absent.

All 125 primary speed cohorts and all 1280 quality generations ran on the specified physical GPUs.
No weak or negative gain stopped a required configuration.
No RL policy, live KV migration, gate change, or new serving engine is part of this study.

## Five-row main table, C64

| Method | GPUs | GSM8K accuracy, correct/N | MATH-500 accuracy, correct/N | Makespan, s | Speedup vs A0 [95% CI] | Peak allocated/GPU, GiB |
|---|---|---|---|---|---|---|
| Naive Ouro (w/ dynamic depth) GPU1-DP1 | 1 | 78.906% (101/128) | 61.719% (79/128) | 27.266 | 1.000× [1.000, 1.000] | 26.836 |
| Naive DP4 Ouro (w/ dynamic depth) | 4 | 75.781% (97/128) | 62.500% (80/128) | 7.304 | 3.733× [3.456, 4.059] | 26.947 |
| Naive DP4 Ouro (w/ dynamic depth) + Hindsight | 4 | 77.344% (99/128) | 57.812% (74/128) | 7.323 | 3.724× [3.493, 4.075] | 26.845 |
| DP4 Ouro + CDB | 4 | 78.125% (100/128) | 63.281% (81/128) | 39.914 | 0.683× [0.640, 0.723] | 26.947 |
| DP4 Ouro + CDB + Hindsight | 4 | 75.781% (97/128) | 67.188% (86/128) | 38.451 | 0.709× [0.688, 0.726] | 26.845 |

The speed column uses the identical C64 bounded-progress request cohort and the same adaptive gate and KV policy.
It divides the 1GPU A0 cohort makespan by the method cohort makespan.
It is multi-request makespan scaling, not single-request latency speedup.

Accuracy uses a separate identical full-answer cohort across all five configurations, with an 8192-token cap.
Speed uses a 256-token natural-EOS cap. The two workloads do not share a latency denominator.
Memory means PyTorch peak allocated memory, not the NVML maximum for the whole device.

## Scaling

| Method | C8 speedup | C16 speedup | C32 speedup | C64 speedup |
|---|---|---|---|---|
| Naive Ouro (w/ dynamic depth) GPU1-DP1 | 1.000× | 1.000× | 1.000× | 1.000× |
| Naive DP4 Ouro (w/ dynamic depth) | 1.260× | 1.237× | 2.079× | 3.733× |
| Naive DP4 Ouro (w/ dynamic depth) + Hindsight | 1.281× | 1.273× | 2.073× | 3.724× |
| DP4 Ouro + CDB | 0.746× | 0.519× | 0.496× | 0.683× |
| DP4 Ouro + CDB + Hindsight | 0.727× | 0.527× | 0.574× | 0.709× |

Each C uses its own matching A0 baseline.
The latency report includes raw seconds, confidence intervals, TTFT, TPOT, and P95 E2E.

## Factorial contributions

| C | Hindsight without refill | Hindsight with refill | Refill without hindsight | Refill with hindsight |
|---|---|---|---|---|
| 4 | 1.046× [1.000, 1.117] | 0.968× [0.955, 1.008] | 1.041× [0.989, 1.100] | 0.963× [0.944, 0.997] |
| 8 | 1.017× [0.977, 1.045] | 0.974× [0.898, 1.023] | 0.592× [0.565, 0.651] | 0.568× [0.565, 0.613] |
| 16 | 1.029× [1.000, 1.052] | 1.015× [1.010, 1.097] | 0.420× [0.381, 0.420] | 0.414× [0.385, 0.420] |
| 32 | 0.997× [0.985, 1.050] | 1.158× [1.106, 1.165] | 0.239× [0.230, 0.245] | 0.277× [0.262, 0.277] |
| 64 | 0.997× [0.997, 1.038] | 1.038× [0.986, 1.098] | 0.183× [0.177, 0.185] | 0.190× [0.173, 0.197] |

The matched 4GPU ratios isolate the measured incremental effects.
They must not be replaced by the much larger A0-normalized four-GPU scaling value.
The interaction report retains paired log-interaction intervals and the historical matched-placement sensitivity.

## Direct answers

1. Local refill utility: at C64, T(A1)/T(A3) is 0.183× and T(A2)/T(A4) is 0.190×. Thus refill is slower in this selected configuration. Median active core batches are 12.465 for A1 and 1.915 for A3. In the separate C64 trace, completion-critical core intervals are 5854.51 ms for A1 and 34146.20 ms for A3. Exit-KV copy intervals are 487.07 ms and 2337.85 ms. These are stream brackets from one trace, not exclusive kernel shares or a paper-performance reproduction.

2. Hindsight without refill: C64 gives 0.997× [0.997, 1.038] over A1. A single shuffled-order control gives 0.994× [0.991, 0.996]. The bounded primary does not establish a large incremental gain over the faster local mode. C4 already shows 1.046× from remapping one request per rank. That low-load effect is not evidence of queue-work balancing, and its causes are not isolated.

3. Hindsight with refill: C32 gives 1.158× [1.106, 1.165], while C64 gives 1.038× [0.986, 1.098]. With the fixed shuffled order, C32 reverses to 0.962× and C64 gives 1.056×. Thus there is no universal additive gain or monotonic saturation trend. The natural-answer aggregate gives 1.571× [1.178, 2.006], but that diagnostic lacks matched output quality and a strong length-aware natural-workload control.

4. Global scope: the authors' CDB scheduler uses one local cache and local queues. It does not inspect another replica's queue or move its requests. The outer run.py controller performs initial DP placement. No request changes GPU after prefill. Any change in rank finish balance from refill is an indirect local-service effect, not CDB global coordination.

5. Work balance versus time: C64 actual depth-work CV decreases from 0.04634 in A1 to 0.02133 in A2. Rank-finish CV decreases from 0.04223 to 0.03536. Yet median makespan changes from 7.304 s to 7.323 s. Under refill, work CV decreases from 0.06021 to 0.01825, but the timing gain is only 1.038×. Batched execution, depth-slot fragmentation, boundary stages, KV copies, and changed trajectories prevent a depth-work sum from being a complete physical cost model.

6. C32 versus C64: the original request order cycles domains and therefore segregates them under round-robin placement. At C32, the rank-summed no-local-request tail decreases from 21.828 to 7.893 rank-seconds under hindsight. At C64 it decreases from 23.645 to 14.392 rank-seconds. These sums are not removable serial wall time. The shuffled-order C32 reversal confirms order dependence.

    CDB queue ordering and realized token/depth changes also differ. These data do not establish a single isolated causal explanation.

7. Historical v3: the preserved 4.410% result is a reduction in mean P95 latency, equivalent to 1.046×. It is not the new cohort-makespan statistic.

    With both historical placements reused, refill gives 1.061× mean-P95 speedup but 1.032× makespan speedup. Without refill, the same historical plan gives 1.085× mean-P95 speedup but only 0.616× makespan speedup. The old search optimized P95, not cohort completion. The 12.77% model also conditions on frozen traces that can change after new batching. This experiment does not identify the exact causal contribution of every mismatch.

8. Quality and interaction: A3 versus A4 has exact token outputs for 134/256 requests and exact depth traces for 46/256. A4 minus A3 is −2.344 pp on GSM8K and +3.906 pp on MATH-500. Their paired 95% intervals are [−5.469, 0.781] pp and [−2.344, 9.375] pp. Neither interval establishes equivalence.

    The selected settings show local fragmentation rather than universal absorption of global scheduling headroom. The no-refill natural-answer diagnostic is 1.288× [0.922, 1.505]. These exploratory natural-workload signals remain conditional, not evidence for an implemented or quality-preserving depth-aware RL policy.

## Validity and limits

The five-arm short probe passes token/depth parity and 6408 exact terminal-KV copy checks.
The earlier independent official-reference comparison remains `REFERENCE_MATCH_PARTIAL`.

Full-answer parity and paired accuracy differences appear in `quality_five_arms.md`.
Changed batch composition can change future outputs and actual depth work.
The study reports those changes instead of assuming fixed execution traces across placements.

The source audit proves local versus global scheduling scope.
It does not prove that every bad refill result comes from one kernel or numerical cause.
The selected synchronous, slot-homogeneous configuration is not every possible CDB implementation.

The LPT hindsight signal combines future output length and depth. It is not a pure depth-only control.
The burst speed cohort is historical and heavily capped. It is not a fresh full-answer serving benchmark.

The five repeats provide limited timing uncertainty estimates.
The question bootstrap does not include all benchmark-selection or parser uncertainty.
Exact whole-GPU idle attribution remains unresolved where other CUDA streams can execute.
Optional C128/C256, new RL training, and deployment-quality validation are not part of this campaign.

## Evidence and next action

Do not train an RL placement policy yet. First run a small teacher-aligned batch-shape fidelity probe on the same token prefixes and native depth-indexed KV states. Compare B1/B3/B16, no-refill/refill, gate logits, greedy logits, and the first differing token. This probe must separate numerical batch effects from a cache or graph-state defect. If fidelity is credible, the next scheduling test needs a strong causal length-aware control on fresh natural-EOS requests.

See `reproduce.md`, `timing_contract.md`, `source_scope_audit.md`, and the raw artifact manifests.
The final ownership audit checks preserved dirty work and restored burners on physical GPUs 4/5/6/7.
Branch: `flashvep/depth-cdb-factorial-hindsight-poc`.
The final user response records the published commit SHA.

## Figures

- [Five-arm scaling](../figures/five_arm_scaling.png).
- [Incremental hindsight and uncertainty](../figures/incremental_hindsight.png).
- [C32 rank timeline](../figures/rank_timeline_C32.png).
- [C64 rank timeline](../figures/rank_timeline_C64.png).
- [Depth work versus measured core intervals](../figures/depth_work_vs_gpu_time.png).
- [Active core batch distribution](../figures/no_refill_refill_batch_histogram.png).
- [Five-arm accuracy](../figures/five_arm_quality.png).
