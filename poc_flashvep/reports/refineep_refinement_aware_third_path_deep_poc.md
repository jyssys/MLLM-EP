# RefineEP: Refinement-Aware Third EP Path Deep PoC

## Executive decision

**Final status: `NO-KERNEL-HEADROOM`.**

LLaDA2.0-Flash refinement does create the hypothesized continuum of compacted
fresh EP shapes. However, on 4xH100 NVLink EP4 the existing DeepEP low-latency
path already wins the full measured range. A credible fixed-contract third
path would improve the strongest existing post-compaction path by only 2.29%
on GSM8K and 2.10% on HumanEval. Even an impossible zero-control path is below
5%. The mandatory 12% CUDA gate therefore failed, so no RefineEP kernel or
runtime integration was implemented.

## Evidence boundary

| result | type |
|---|---|
| current 5.925/7.261 s request anchors | measured clean E2E |
| fresh-M/expert/rank geometry | measured routes + future-known live-row filter |
| normal/LL communication envelope | measured 4-GPU replay, 5 restarts |
| O0/O1/O2/O3 request results | analytical post-compaction oracles |
| RefineEP speedup/correctness | not measured; kernel not built |
| Epoch performance | not measured; compaction is sensitivity only |

This distinction matters: the 23.63%/24.28% total current-to-O3 reduction is
mostly hypothetical liveness compaction plus an existing path. It is not a
RefineEP result.

## Substrate

The local checkpoint revision is
`744c3f8c6c8317d2377d6d16d8a3d4be2caef563`: BF16, 32 layers, hidden 4096,
256 routed experts/top-k8, 64 routed experts per rank, one shared expert, MoE
intermediate 1024. Execution is dense TP4 + routed EP4/DP1: DeepEP dispatch,
owner-rank fused experts, reverse combine. Only physical GPUs 4--7 were used;
all pairs are NV18.

The best-static request configuration is submitted batch 32,
`mini_batch_size=32`, generation/block 32, threshold 0.9. Clean request medians
are 5.925 s/NFE66 for GSM8K-32 and 7.261 s/NFE86 for HumanEval-32.

## Finding 1: the refinement shape continuum is real

The atlas contains 4,650 layer-waves from two real trajectories. GSM8K fresh M
has p10/p50/p90/max 4/164/483/1024; HumanEval has 5/102/440/802. Medium-small
plus medium shapes account for 66.15%/65.88% of all rows in the two tasks.

Refinement strongly changes physical shape after hypothetical exact liveness
compaction:

| task | early median M | late median M | early p50 rows/expert | late p50 | late tiny-expert fraction |
|---|---:|---:|---:|---:|---:|
| GSM8K | 925.5 | 35.5 | 14.25 | 2 | 68.5% |
| HumanEval | 802 | 15.5 | 13 | 2 | 79.4% |

Thus the accurate statement is: **dLLM refinement creates a continuum of
prefill-like to decode-like physical EP shapes inside one block.** It does not
literally repeat prefill and decode.

## Finding 2: there is no third gap in the existing envelope

The replay uses 25 real route shapes and 15 matched M/fanout/skew controls.
Each policy was measured with 8 warmups, 30 repeats, randomized order, and five
independent process restarts. Timing is critical-rank wall from same-GPU CUDA
events.

| shape | normal fresh | LL | median LL gain |
|---|---:|---:|---:|
| very-small | 0.2449 ms | 0.0909 ms | 62.66% |
| small | 0.2457 | 0.0903 | 63.23% |
| medium-small | 0.2492 | 0.0928 | 62.74% |
| medium | 0.2610 | 0.1292 | 50.48% |
| large | 0.2768 | 0.2125 | 24.22% |

LL beats normal in 25/25 real cases in every restart and 15/15 controlled
cases. Varying fanout, remote fraction, and skew changes the margin but never
creates an uncovered regime. Exact cached-normal metadata beats LL only for
M802/1024 and is invalid as a general refinement policy when routes change.

The legacy LL API warned about unavailable IBGDA, then used its explicitly
enabled single-node NVLink/P2P fallback. Current upstream DeepEP has newer V2
support, but the validated local PyTorch 2.8 contract cannot run it; this
limits coverage without creating a positive claim ([DeepEP](https://github.com/deepseek-ai/DeepEP),
[legacy API](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)).

## Finding 3: large operator-local tax does not become large request headroom

The measured sustainable aggregate two-way bandwidth is 854.7 GB/s on the
same GPUs. For very-small through large cases, payload-only lower bounds range
from about 0.0005 to 0.0719 ms, while LL measures 0.0909 to 0.2125 ms. This
looks locally large, but compacted LL communication sums to only 259.1 ms
(GSM8K) and 298.3 ms (HumanEval), 4.37% and 4.11% of current request time.

The oracle deliberately favors RefineEP:

| task | post-compact normal | O0 existing LL | O1 zero-control | O2 physical LB | O3 credible third path |
|---|---:|---:|---:|---:|---:|
| GSM8K | 4882.8 ms | 4630.9 | 4427.3 | 4466.4 | 4524.9 |
| HumanEval | 5979.6 ms | 5616.0 | 5369.4 | 5420.8 | 5498.0 |

| task | O0 vs normal | impossible O1 extra vs O0 | credible O3 extra vs O0 |
|---|---:|---:|---:|
| GSM8K | 5.16% | 4.40% | **2.29%** |
| HumanEval | 6.08% | 4.39% | **2.10%** |

O3 assumes only 50 us total control per dispatch+combine invocation, already
about 2.5x below the observed tiny LL kernel floor. Making this an implausible
25 us still yields just 3.34%/3.25%. The negative therefore does not depend on
a conservative target.

## Correctness and robustness

Every replay preserves global source/received assignment counts. The maximum
LL relative L2 versus normal is 0.000897, consistent with BF16 reduction-order
drift. Normal and cached-normal semantics match exactly. This validates the
existing communication comparison; it is not RefineEP correctness.

Absolute restart medians vary with node state, but the paired LL winner is
invariant in 125/125 real case-restart comparisons. Observer-heavy Nsight
totals were excluded; CUDA-event medians are primary. Nsight only confirms a
substantial fixed LL kernel floor. Nsight Compute was unavailable, so no SM
utilization result is invented.

## Why CUDA implementation stopped

The spec requires >=12% credible request E2E on both tasks. O3 is 2.29%/2.10%,
and the impossible O1 is still below the 5% kill gate. A CUDA prototype would
be research activity without decision-changing evidence. The following were
therefore deliberately not implemented:

- fixed-contract dispatch/combine CUDA kernels;
- persistent refinement service;
- runtime selector;
- LLaDA2 integration;
- EP8 and second-model expansion.

## Prior-art attack

The workload characterization may be useful, but the mechanism space is
crowded. Epoch/FreshLane covers what fresh work remains; this PoC only studies
how such work would execute ([Epoch](https://arxiv.org/abs/2609.09748)).
StreamEP exposes streaming-tile EP mechanics on H100/NVLink
([repository](https://github.com/evolutionaryscale/StreamEP)). FlashMoE and
mKernel pursue persistent/fused dispatch-expert-combine paths
([FlashMoE](https://flash-moe.github.io/),
[paper](https://papers.nips.cc/paper_files/paper/2025/file/918d938bd209e5b56072777366f8a211-Paper-Conference.pdf),
[mKernel](https://github.com/uccl-project/mKernel)). UniEP also explores a
unified expert-parallel megakernel ([paper](https://arxiv.org/abs/2604.19241)).

Consequently, fixed buffers, fused layout, NVLink specialization, or a
persistent kernel are not clean novelty by themselves. RefineEP would need a
new exact data contract or an actually uncovered performance regime.

## Final answer to the research question

The **shape phenomenon exists**, but the **kernel opportunity does not** on
this 4xH100 EP4 contract. Existing LL already occupies the medium/small
refinement continuum, and the whole-request mass remaining above an aggressive
third-path target is too small. Do not build RefineEP here. If exact liveness
compaction is integrated, first enable and validate the existing LL path.

## Artifacts

- Detailed decision: `poc_refineep/reports/final_decision.md`
- Machine-readable atlas: `poc_refineep/REFINEMENT_EP_SHAPES.csv`
- Replay corpus/summary: `poc_refineep/EP_SHAPE_REPLAY.jsonl`,
  `poc_refineep/EP_SHAPE_SUMMARY.csv`
- Five-restart real/controlled results: `EXISTING_KERNEL_BENCH_RESTARTS.csv`,
  `CONTROLLED_KERNEL_BENCH_RESTARTS.csv`
- Oracles: `HEADROOM_ORACLE.csv`, `CONTROL_FLOOR_SENSITIVITY.csv`
- Raw result root: `poc_refineep/results/refineep_20260914_172330/`
- Figure inventory: `poc_refineep/reports/FIGURE_STATUS.md`

## Sources

- [DeepEP official repository](https://github.com/deepseek-ai/DeepEP)
- [DeepEP legacy documentation](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)
- [Epoch](https://arxiv.org/abs/2609.09748)
- [StreamEP official repository](https://github.com/evolutionaryscale/StreamEP)
- [FlashMoE project](https://flash-moe.github.io/)
- [FlashMoE paper](https://papers.nips.cc/paper_files/paper/2025/file/918d938bd209e5b56072777366f8a211-Paper-Conference.pdf)
- [mKernel official repository](https://github.com/uccl-project/mKernel)
- [UniEP](https://arxiv.org/abs/2604.19241)
