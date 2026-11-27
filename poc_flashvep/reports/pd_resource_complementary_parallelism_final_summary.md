# Resource-complementary PD on H200

## Decision

No proposed direction advanced to serving integration under the frozen physical and incremental gates. The strongest useful observation is that P has a substantial backlog while D spends much of the concurrent window without recorded kernels. Applying that opportunity as simultaneous resource sharing failed the tested decode protection gate. The experiment establishes a benefit for some isolated temporal placements and measures the online generic controls separately.

| Direction | Verdict | Limiting measurement |
|---|---|---|
| A. Operator-level PD | `NO_GO_INTERFERENCE` | 0/24 simultaneous-release and 0/6 retiled P→D→P arms pass both the 10% P improvement and the 5% D protection thresholds. Best retiled D p95 harm is +63.87%. |
| B. Destination-aware compute | `NO_GO_COMMUNICATION` | For one Qwen3-8B layer, hidden-forwarding plus reprojection is 3.72–4.74× the direct K/V path. A contiguous suffix can amortize hidden bytes and has prior art. |
| C. True MoE P/D sharing | `NO_GO_INCREMENTAL` | Physical EP2 yielded 7/48 useful NCCL pairings and 0/48 DeepEP HT pairings. The seven use D expert compute and a small P dispatch, an existing DBO pattern. |
| D. Heterogeneous prefill | `NO_GO_INCREMENTAL` | Best four-layer eager improvement over the fastest uniform layout is 1.92%. At 16K with CUDA Graph, the hybrid is 2.51% slower than balanced uniform CP2. |

[Selection and gates](pd_resource_complementary_candidate_selection.md) · [Verdict matrix](../pd_resource_complementary_parallelism/figures/candidate_verdict_matrix.png)

## H200 native behavior

Qwen3-8B BF16, FlashAttention 3, CUDA Graph, paged KV and physical NIXL run as separate 1P and 1D processes. The primary shared-nothing prompt traces disable APC. The performance inputs come from real LongBench records and are truncated or extended to freeze exact token geometry. P-only calibration uses a separate reference process. Both the overloaded drains and the QPS sweep use a finite open-loop cohort.

An observer-qualified 16K/16-output, 2.6 offered req/s N24 trace recorded P kernel busy of 81.29%, D of 14.13%, and joint kernel activity of 12.54% over a common central arrival window. P was active while D had no recorded kernel during 68.75% of the window. This lightweight CUPTI trace retains 24/24 exact token outputs against both clean arms, with a 2.87% maximum primary timing shift. Kernel-only coverage gives an upper bound on temporal investigation because it leaves DMA, scheduler handoff and future deadlines outside the timing coverage.

The N64 16K/16-output confirmation at 2.6 req/s completes 30,500 P prompt tok/s in the central arrival interval, 94.24% of the strongest finite P-only witness across the tested batch budgets. P has about 6.66 waiting requests on average, all in scheduler capacity waiting. D has zero sampled capacity waiting. Sampled KV occupancy peaks at 11.72% on P and 4.69% on D, with no preemptions and 64/64 requests within the absolute decode SLO. This supports a prefill-side backlog over D memory pressure in this measured region. The simultaneous N64 CUDA observer fails the 5% timing guard, so its 92.62% P and 16.30% D kernel busy measurements are diagnostic. The admitted N24 simultaneous measurement reaches 84.79% of the tuned P witness and captures a different workload window. The strict joint motivation gate requires its conditions in one valid observation.

A further independent N64 start profiled P alone. Its P union was 92.52% and its greedy outputs agreed on 64/64 in both clean comparisons. TPOT p95 shifted +5.81% and +19.98% and the clean-arm ITL p99 shifted −26.20%. The P-only trace is diagnostic under the same 5% guard. This observation recorded only P kernel intervals. Clean measurements and the lean N24 trace establish the prefill pressure and temporal opportunity in the tested region, with strict same-window qualification still inconclusive.

The initial clean N64 point at 16K/16 and 1.9 req/s completed 64/64. Mean sampled waiting requests were 2.25 on P and 0.04 on D. P/D mean sampled SM activity was 86.82/10.36%, Tensor activity 73.18/0.74% and DRAM bandwidth activity 15.27/10.26%. TTFT p95 was 3.86 s, TPOT p95 5.41 ms, ITL p99 24.96 ms and goodput 1.546 req/s (56/64). At 2.6 req/s N64 the main queue raised TTFT p95 to 9.80 s. The short-context O256 control drove average D DRAM bandwidth activity to 61.38%, a materially different regime. These aggregate counters give utilization of each reported resource over the chosen window. They leave available simultaneous kernel resources and deadline-safe slack unmeasured.

The 16K N24 activity trace splits P kernel active time into attention 28.94% and linear 65.38%, and D into attention 17.97% and linear 74.42%. Fraction denominators are each GPU's interval union. Classification uses kernel-name heuristics, with 0.98% of P and 2.66% of D activity left unresolved. Profiler A/A failure marks the first Kineto run and the independent shape observer as diagnostic. The unprofiled default D recorded exactly one engine token in each of 1,024 iterations at all three matched N64 arrival points, corroborating the B1 control.

[Native motivation](pd_prefill_heavy_resource_motivation.md) · [Resource profiler](pd_operator_resource_profiling.md) · [Raw measurements](../pd_resource_complementary_parallelism/results/raw_native.tar.gz)

## Answers on design and generality

- **A, active D.** Qwen3-8B FA3 at D B4/KV16K shows an isolated Tensor/DRAM mix of 19.60/54.73%, while P4096 MLP reaches 61.92/26.73%. The resource pattern survives original models and native kernels. In simultaneous streams it produces kernel interference. With real P→D and D→P transfer, a 50% split at P8192/D B1/KV16K saves 20.87% of the source block time and raises D p95 from 0.124 to 1.16 ms. Retiling mitigates the conflict and still violates the 5% bound. A 112D/16P SM split protects some D B1 isolated comparisons and makes P MLP 6.44–6.82× slower. The current pairing has no demonstrated advantage over request, token or layer granularity. [Operator study](pd_operator_level_parallelism_discovery.md)
- **A, temporal D.** With D released during the source front, 15/24 cases pass the numerical P/D limits and have zero median measured overlap. A half share at the matched shape reduces the source block p50 from 7.20 to 5.66 ms while raising D p95 by 0.27%. That is feasible block-level useful work. An engine would need an online admission contract and evidence that filling those opportunities at operator boundaries beats an already optimized whole-request/token fallback. The tested activation round trip adds 64 MiB to one block in this case. [Release-time tradeoff](../pd_resource_complementary_parallelism/figures/operator_release_tradeoff.png)
- **B.** Desired K/V is 4,096 B per token per layer, from an 8,192 B hidden row in BF16 Qwen3. The attention worker needs its own original K/V; D repeats computation when given the same hidden row. In the 8,192-row probe, the exact direct copy is 0.1285 ms and the destination path 0.6085 ms. Qwen2.5-7B-Instruct also disfavors hidden forwarding, with 7,168 B vs 2,048 B and a 3.96× time ratio at 4,096 rows. There is no evidenced independent A+B gain because A's measured MLP move starts after the layer's K/V was already used. [Transfer study](pd_destination_aware_computation_discovery.md)
- **C.** P-EP2/D-EP2 has real expert owners and all-to-all. The largest positive concurrent effects occur in a standard small-dispatch/large-expert combination. The active D cases with P expert GEMM fail protection. The four-GPU native 30B pilot completed 24/24 in each of its two cohorts and verified the active backend and physical NIXL path. Numerical comparisons are from the separate layer-0 EP replay. [MoE study](pd_moe_resource_sharing_discovery.md)
- **D.** Changing the layout can reduce collectives in isolation. At 32K/4 layers the TP-attention/row-MLP hybrid cuts NCCL operation time from 7.59 to 4.84 ms versus TP2 but adds 2.11 ms of MLP time. Balanced uniform CP2 reaches 79.12 ms against the 79.00 ms hybrid, a 0.16% gap. It uses a lower logical peak rank TX payload. Handoff-specific benefit and D-tail protection for this design remain unexecuted. [Prefill layout study](pd_heterogeneous_prefill_parallelism_discovery.md)

## Serving and research significance

The dense raw archive covers 67 complete cohorts, including P-only service references and observer controls. Real end-to-end dense serving arms complete 3,032 of 3,032 requests across the repeated cohorts. Separate true EP native pilots complete 48 of 48 requests on four H200s.

Candidate TTFT, TPOT, ITL, E2E and goodput gain, task-quality noninferiority and candidate performance under APC or an alternative model remain `NOT_RUN`. Real native PD and generic routing comparisons are the online evidence in this campaign. All primary native geometry studies and both dense-model physical screens ran with the original BF16 weights. Llama 3.1 access returned HTTP 401 without approved gated credentials, so the physical secondary is the already available Qwen2.5-7B-Instruct checkpoint. Optional 32B Dense serving confirmation was outside this screen.

At 16K/O16 and 2.6 req/s, the generic two-GPU whole-request deflector raises finite goodput from 0.757 to 1.964 req/s and cuts TTFT p95 from 9.96 to 4.50 s against its matched PD control. It also increases TPOT p95 from 4.59 to 13.64 ms and ITL p99 from 8.17 to 25.56 ms, exceeding both 5% guardrails. Among equal three-GPU allocations at that point, two independent P replicas and one D reach 2.344 req/s, versus 0.758 for one P and two D and 0.844 for one TP2 P and one D. The corresponding ITL p99 values are 23.29, 6.78 and 262.28 ms. Placement is a measured trade-off across offered load, queueing, handoff and tail protection. These are one-start finite controls. [Serving baseline detail](pd_resource_complementary_online_serving.md)

The online results establish the compare-against bar for a future mechanism. The narrow A idea would still need end-to-end proof of deadline-safe D placement under outstanding work, beyond ConfluxServe, spatial multiplexing and existing elastic PD. B's contiguous layer alternative has nearby precedent in ReKV. C's surviving pair belongs alongside DBO and ExpertPlex. D's main measured advantage is already available through generic TP/CP geometry choices. There is no evidence for an independent systems-paper contribution from the tested new combinations.

Maintain the balanced generic baseline and optimize measured scheduler/connector bottlenecks first. A follow-on operator experiment should start with an observed active-decode geometry and a qualified execution quantum that actually protects D while finishing useful, transfer-complete work. There is no basis to optimize the nominal overlap percentage alone.

## Gate disposition

| Gate | Status |
|---|---|
| 0. Native baseline | Pass. Optimized BF16 native P/D, colocated and static topology controls measured. |
| 1. Actual motivation | P-side backlog and D temporal opportunities observed. Strict same-window qualification inconclusive. |
| 2–3. Physical and incremental | A and B stop on protected concurrent execution or communication cost. C and D stop on incremental value. |
| 4–5. Candidate online and confirmation | `NOT_RUN`. The online measurements are native and generic controls. |

The initial full profiler, high-load simultaneous observer, independent scheduler shape observer and P-only high-load observer each exceeded the frozen observer tolerance. Their discovered timing data remain available as diagnostics. The two primary GPU experiment devices were returned to their original burner operation, and all eight `train_lm.py` jobs were verified at restoration. [Burner receipt](../pd_resource_complementary_parallelism/results/burner_restore.json)

[Serving measurements and scope](pd_resource_complementary_online_serving.md) · [Prior work](pd_resource_complementary_prior_work.md) · [Reproduction](../pd_resource_complementary_parallelism/README.md)
