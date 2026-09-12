# LLaDA2.0-Flash 100B EP4 logical/physical mismatch discovery

## Executive decision

**Final label: `CHARACTERIZATION-SIGNAL`.**

The discovery sprint found a robust physical fact: expert-row shape materially improves expert-kernel latency prediction beyond token/expert-pair count. After `(dataset, layer, rank)` p99 trimming, adding active-support and rows-per-expert statistics reduces leave-dataset-out RMSE from 0.05347 to 0.02929 ms (**45.2%**), while fanout/load/remote features add only 0.47% more. However, a zero-cost perfect shape/fragmentation optimizer is worth only **3.311% GSM8K / 2.852% HumanEval** of clean request E2E. After hypothetical Epoch-like live-row compaction, the novel residual is still only **2.119% / 1.134%**. No method meets even the 5% weak gate.

Therefore no custom kernel, coalescer, online scheduler or TP4 control was implemented. This is a scientific NO-GO for a paper-level method from the measured signal, not an environment failure.

## Substrate and evidence

- Model: local `inclusionAI/LLaDA2.0-flash` revision `LLaDA2.0-flash-744c3f8`, BF16, 32 layers, hidden 4096, 256 routed experts/top-8 plus one shared expert.
- Runtime: instrumented dInfer commit `9132ce9a` on base `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`; SGLang 0.5.3.post1; DeepEP `1.2.1+73b6ea4`; PyTorch 2.8.0+cu128/NCCL 2.27.3.
- GPUs: physical H100 0--3 only, all-to-all NV18 connectivity.
- Topology: dense TP4 + routed EP4, 64 complete experts/rank, DeepEP normal dispatch → owner-local fused expert → reverse combine, DP1.
- Baseline: submitted batch 32, mini32, gen32/block32, the prior sweep's strongest static configuration.
- Clean evidence: three independent restarts per dataset. GSM8K median 6.075 s/NFE66; HumanEval 7.343 s/NFE86.
- Diagnostic evidence: 4,650 refinement layer-waves and 18,600 rank-layer rows across 64 requests. Observer-heavy wall time is excluded from performance claims.

The three clean restarts produce identical answers within each dataset and the same bounded quality score (GSM8K 5/32, HumanEval 6/32). These short-generation scores are a deterministic substrate anchor, not a quality claim for a new method.

Detailed environment and execution proof: [00_environment.md](../../poc_dllm_ep_discovery/reports/00_environment.md). Clean/trace coverage: [01_trace_summary.md](../../poc_dllm_ep_discovery/reports/01_trace_summary.md).

## What changed logically and physically

| dataset | decision-live early→late | active experts | p50 rows/expert | <=4-row experts | expert time | whole measured MoE |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 0.904→0.181 (-80.0%) | 186→160 | 15.75→5 | 25.5%→46.2% | -23.4% | -21.2% |
| HumanEval | 0.783→0.172 (-78.1%) | 200→116 | 15→3 | 26.4%→64.0% | -42.5% | -24.4% |

Liveness collapses much faster than physical cost, but this is the known dead-work problem now directly targeted by [Epoch](https://arxiv.org/abs/2609.09748). It also narrowly misses the predeclared `latency <=20%` strong example on GSM8K and misses clearly on HumanEval.

The more novel proposed paradox does not survive control. At fixed M=1024, the lowest-live wave does activate more experts (+14.5% GSM8K, +7.0% HumanEval), but tiny-expert fraction falls rather than rises; expert time changes only +6.2%/+1.7%, and whole-MoE time is effectively flat/slightly lower. Natural late fragmentation is mainly a shrinking-ready-pool effect.

Full reproduction/control analysis: [02_known_signal_reproduction.md](../../poc_dllm_ep_discovery/reports/02_known_signal_reproduction.md) and [03_logical_physical_mismatches.md](../../poc_dllm_ep_discovery/reports/03_logical_physical_mismatches.md).

## Surprising but non-actionable observations

### Shape beats count

The p99-robust held-out predictor improves expert-time RMSE by 45.2% when active support and rows-per-expert statistics are added. Count-matched adjacent pairs also show a median 1.153× expert-time ratio, although the largest cases are consistent with known runtime tails and are not called fragmentation. Shape features do not similarly explain dispatch latency, indicating separate communication/runtime state.

Detailed model evaluation: [04_latency_predictor_analysis.md](../../poc_dllm_ep_discovery/reports/04_latency_predictor_analysis.md).

### Coarse geometry is stable; exact compute is not

Rank-load cosine is roughly 0.999, and late destination-rank-set identity is about 74%. Yet exact top-k set identity is only 7--18%. No representative layer/phase has any matched MoE output with relative L2 below `1e-4`; middle-layer output rel-L2 is roughly 0.61--0.82. Stable rank geometry therefore does not authorize exact expert-result reuse.

### Confidence/router paradox is absent

Late GSM8K confidence rises 24.7%, router entropy falls 2.5%, and top-k mass rises 21.4%. HumanEval has the same routing direction. The hypothesized high-confidence/diffuse-routing region is not present.

## Post-compaction sensitivity

Keeping only captured decision-live rows makes late work highly fragmented: compact p50 rows/expert is 2 and <=4-row fraction reaches 68.5% GSM8K / 79.4% HumanEval. But economics kill the successor:

| oracle | GSM8K E2E | HumanEval E2E |
|---|---:|---:|
| hypothetical live-row compaction, expert-only context | 7.108% | 6.481% |
| perfect new fragmentation residual, p99 robust | **2.119%** | **1.134%** |
| 50%-capture feasible proxy | **1.060%** | **0.567%** |

The compaction column is modeled sensitivity and prior-art context, not measured Epoch. The residual comparison is the new contribution test. Details: [05_post_compaction_sensitivity.md](../../poc_dllm_ep_discovery/reports/05_post_compaction_sensitivity.md).

## Oracle tournament and methods

At least three method principles were generated from the only strong signal:

1. cross-wave expert-major live-work coalescing;
2. fragmentation-aware request/wave composition or bounded expert queues;
3. exact tiny-expert specialized execution;
4. an explanatory shape-aware cost model;
5. route-plan delta reuse as a separate temporal candidate.

All execution candidates fail. The strongest zero-cost ceiling is below 3.4%; waiting, map construction, packing and a new kernel can only lower it. The earlier ready-set-feasible RAWS oracle was also only 0.650%, independently weakening dynamic composition.

Candidate disposition: [07_method_candidates.md](../../poc_dllm_ep_discovery/reports/07_method_candidates.md). Oracle construction: [08_candidate_oracles.md](../../poc_dllm_ep_discovery/reports/08_candidate_oracles.md).

## Prior-art attack

- [Epoch](https://arxiv.org/abs/2609.09748) already compiles diffusion blocks and sends only live/new/refresh-required positions through the EP pipeline. Our distinct residual would have to optimize the remaining sparse work; it is only 1.13--2.12% E2E.
- [TEAM](https://arxiv.org/abs/2602.08404), [REFLEX](https://arxiv.org/abs/2608.01784), and [DES](https://arxiv.org/abs/2602.00879) already cover temporal/refinement-aware expert work allocation from different angles. Confidence-conditioned expert reduction is both unsupported here and crowded.
- [DICE](https://openaccess.thecvf.com/content/ICCV2025/papers/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.pdf) covers stale-activation communication overlap in diffusion MoE.
- [MegaBlocks](https://github.com/databricks/megablocks) and existing fused/grouped MoE execution make generic “pack rows by expert” non-novel. [DeepEP](https://github.com/deepseek-ai/DeepEP) already supplies optimized EP dispatch/combine and its current V2 changes the runtime substrate further.

The only apparently orthogonal framing—post-Epoch live-row fragmentation—is blue-ocean enough to ask but too small to pursue here. Full matrix: [09_prior_art_audit.md](../../poc_dllm_ep_discovery/reports/09_prior_art_audit.md).

## Final answers

1. **Largest logical change:** decision-live work falls about 78--80%.
2. **Weak physical response:** whole measured MoE falls only 21--24%; dispatch about 23--26%.
3. **Support broadening:** absent in raw phase aggregates; modest at fixed M, without the proposed tiny-growth triple.
4. **Rows/expert collapse:** yes naturally, but primarily with physical ready-pool shrinkage.
5. **Expert plateau:** partial on GSM8K, not robust enough across HumanEval for the strong gate.
6. **Best latency feature:** expert support/row-shape, not count alone.
7. **Late fragmentation:** real descriptively; not shown to be independently caused by logical liveness.
8. **Compaction effect:** it exposes more tiny expert groups.
9. **Post-Epoch residual:** no; only 1.13--2.12% perfect E2E.
10. **Strongest novel oracle:** current perfect fragmentation removal, 2.85--3.31%.
11. **Material total-latency method:** none.
12. **EP specificity:** unproven; TP4 was not triggered after the EP4 gate failed.

## Final disposition

**`CHARACTERIZATION-SIGNAL`** — retain the cost-model and post-compaction interaction results, but do not pursue this branch as a paper-level training-free inference method under the requested thresholds.

Machine-readable evidence begins at [README.md](../../poc_dllm_ep_discovery/README.md), with final gate details in [final_decision.md](../../poc_dllm_ep_discovery/reports/final_decision.md). No live method PoC was run because its oracle gate failed: [10_best_candidate_live_poc.md](../../poc_dllm_ep_discovery/reports/10_best_candidate_live_poc.md).
