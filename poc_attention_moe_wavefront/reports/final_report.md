# Attention–MoE Wavefront Phase-0 PoC

## Executive decision

**Final status: NO-GO.** The most optimistic proportional-progress,
zero-split-cost ceiling reaches a median 15.47% Amdahl-adjusted TTFT reduction
over three multimodal workloads, so the preregistered split-tax gate was
correctly entered. Fresh empirical split measurements then reject the two
assumptions that make that ceiling actionable:

1. exact fragment cost is not token-linear—two fragments cost about 2.41x the
   unsplit Attention and 2.41x the unsplit MoE at the median; and
2. the useful mathematical cut is not a modality boundary—the modality-boundary
   O3 ceiling is only 1.59% median TTFT while a generic one-third cut recovers
   essentially all of the ideal O1 ceiling.

The exact sequential one-third split regresses clean prefill CUDA span by
40.37% median. The naive concurrent split regresses it by 479.76%, gives a
negative overlap-efficiency estimate, and fails exact-logit correctness on the
largest vision-heavy workload. A stock-or-candidate lower envelope therefore
has 0% contention-corrected benefit. Phase-1 zero-copy/custom-kernel work is not
recommended for this direction.

## 1. Environment

| Item | Value |
|---|---|
| Repository HEAD | `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f` |
| Branch | `flashvep/attention-moe-wavefront-poc` |
| Model | Qwen3-VL-30B-A3B-Instruct, BF16 |
| Model snapshot | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Topology | TP2 / DP2 / EP4 / PP1 |
| GPUs | physical 4,5,6,7; four H100 80GB; all-pair NV18 |
| Runtime | vLLM 0.20.0+cu129, PyTorch 2.11.0+cu129, CUDA 12.9 |
| Collective stack | NCCL 2.28.9, DeepEP 1.2.1+73b6ea4 |
| Attention / expert | FlashAttention 3 / Triton unquantized MoE |
| EP communication | DeepEP high throughput |
| Execution controls | DBO off for baseline/N1; eager; CUDA graphs off |

The original checkout was dirty and was not edited. Work was done in an
isolated worktree and all new code/results live below
`poc_attention_moe_wavefront/`. Total fresh live four-GPU wall time was 30.06
minutes, or 2.004 aggregate GPU-hours.

## 2. Runtime validation

Source inspection confirms the full-query barrier in the installed runtime.
`Qwen3MoeAttention.forward` computes and returns the complete attention output;
`Qwen3MoeDecoderLayer.forward` then invokes post-attention RMSNorm and the MoE
MLP. There is no token-range readiness interface at that boundary. Fresh logs
independently confirm `DeepEPHTAll2AllManager`,
`DeepEPHTPrepareAndFinalize`, Triton experts, EP4, and the allowed GPU mapping.

Timing uses same-device CUDA event differences only. Logical stage latency is
the slowest rank duration; cross-device absolute clocks are never subtracted.
The paired instrumentation overhead median was 7.05%, exceeding the 3% target,
so instrumented data is used only for stage attribution and uninstrumented
request repetitions provide the TTFT denominator.

## 3. Baseline critical-path breakdown

The four workloads contain actual processor token counts, not nominal prompt
lengths.

| Workload | Total / vision tokens | Clean TTFT (ms) | Attention (ms) | Dispatch (ms) | Expert (ms) | Combine (ms) | MoE total (ms) | Attention+MoE / TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 text control | 2363 / 0 | 127.15 | 33.01 | 15.91 | 27.17 | 7.25 | 65.13 | 77.2% |
| W2 standard image | 248 / 228 | 123.16 | 30.26 | 16.84 | 20.18 | 5.39 | 63.61 | 76.2% |
| W3 vision-heavy | 2363 / 2340 | 242.31 | 33.48 | 15.68 | 27.38 | 6.81 | 65.34 | 40.8% |
| W4 multi-image | 511 / 488 | 130.22 | 30.28 | 16.42 | 21.39 | 4.58 | 64.66 | 72.9% |

Attention+MoE is material. W3 nevertheless has a smaller request-level share
because vision encoding and other TTFT work dominate its denominator. Thus a
large decoder-layer percentage cannot be promoted directly to a TTFT claim.

## 4. Attention/MoE scaling

The initial O1–O4 analysis deliberately used the requested split-cost-zero
counterfactual: measured unsplit stage time was divided proportionally by the
logical token cut, with no extra launch, copy, packing, or materialization.
This is a mathematical ceiling, not an empirical statement about FlashAttention
progress.

The follow-up exact sequential grid measured 126 request/cut cases with two
repetitions per cut over 48 layers and all four EP ranks. Results reject linear
scaling:

| Metric | p10 | median | p90 |
|---|---:|---:|---:|
| `(A_head + A_tail) / A_all` | 2.10x | 2.41x | 3.23x |
| `(E_head + E_tail) / E_all` | 2.08x | 2.41x | 3.53x |

Even after granting perfect zero-contention overlap to these empirical fragment
costs, the per-layer best empirical cut is slower in **144/144** multimodal
request-layer groups. The fragment-aware O1 request projection is −54.18%
median TTFT (a regression, not a saving). This does not prove that a future
custom fused kernel can never expose tile readiness; it proves that the current
exact fragment contract offers no positive oracle.

## 5. O1/O2/O3/O4 zero-cost oracle

| Workload | O1 per-layer | O2 per-request | O3 modality boundary | O4 global 1/3 |
|---|---:|---:|---:|---:|
| W1 text control | 16.90% | 16.90% | N/A | 16.90% |
| W2 standard image | 16.38% | 16.31% | 1.59% | 16.31% |
| W3 vision-heavy | 8.90% | 8.89% | 0.11% | 8.89% |
| W4 multi-image | 15.47% | 15.45% | 12.70% | 15.45% |
| **Multimodal median** | **15.47%** | **15.45%** | **1.59%** | **15.45%** |

All percentages are Amdahl-adjusted against clean TTFT. O2 and O4 recover
nearly all of O1, but that is evidence for a generic balanced pipeline cut, not
for modality-aware scheduling.

## 6. Amdahl-adjusted TTFT oracle

The preregistered Stage-C gate was passed because the optimistic multimodal O1
median is 15.47%, just above the 15% threshold. Per-request results reveal the
fragility of that median: W3, the most vision-heavy request, reaches only 8.90%
because Attention+MoE is a smaller part of its TTFT. The corresponding
fragment-aware empirical O1 median is −54.18%, and the stock lower envelope is
therefore 0% achievable improvement through the measured exact split path.

## 7. Optimal split analysis

The proportional O1 optimum is effectively constant at one third across every
layer and workload. O2 fractions are 0.3301–0.3307; the global O4 fraction is
0.330089. This is the point at which the proportional Attention-tail and
MoE-head terms balance in the oracle equation. It is generated by stage ratios,
not by modality.

After empirical fragment costs are included, the request-level best cut moves
inconsistently (0.50 for W2, 0.67 for W3, and 0.10 for W4), while every cut
remains slower than stock. A changing optimum inside an all-negative curve is
not actionable scheduling evidence.

## 8. Modality-boundary relevance

For W2 and W3, the canonical end-of-vision boundary lies near the end of the
sequence (fractions 0.935 and 0.992), whereas the ideal optimum stays near
0.33. Their normalized distances are 0.605 and 0.662. W4 has multiple image
spans and a closer internal boundary, yet the best-boundary oracle still does
not establish a systematic modality rule.

Most importantly, text-only W1 exhibits the same ideal 0.33 optimum and an even
larger 16.90% ceiling. The hypothesized opportunity is therefore not
MLLM-specific. The median O3 TTFT ceiling of 1.59% fails the 8% relevance gate.

## 9. Naive split results

The bounded physical benchmark used the same three multimodal inputs and a
one-third split. Each policy has one warmup, one detailed observation, two clean
CUDA repetitions, and a correctness request.

| Request | N0 stock (ms) | N1 sequential (ms) | N2 concurrent (ms) | N1 vs N0 | N2 vs N0 |
|---|---:|---:|---:|---:|---:|
| coffee | 123.23 | 187.35 | 752.96 | +52.03% | +511.00% |
| method | 138.30 | 194.14 | 754.37 | +40.37% | +445.44% |
| coffee+rocket | 130.61 | 181.60 | 757.22 | +39.04% | +479.76% |
| **median** | — | — | — | **+40.37%** | **+479.76%** |

Greedy first tokens agree for all requests. N1's largest logit deviation is
cosine 0.99979 / relative L2 2.04%. N2's vision-heavy method request reaches
cosine 0.99522 / relative L2 10.14%, so the concurrent measurements cannot be
used as correctness-preserving speedup evidence even if they had been faster.

## 10. Overlap efficiency

N1 stage traces estimate a median physical overlap opportunity of about
33.45 ms. Instead of hiding that work, N2 adds about 560–576 ms relative to N1.
Using the preregistered formula yields median raw `eta = -16.75`.

This very negative value should not be interpreted as a pure hardware
Attention-versus-Tensor-Core contention coefficient. The N2 diagnostic runs
two model microbatches using cooperative Python threads, separate CUDA streams,
per-layer dependency events, and a shared DeepEP collective path. The nearly
token-size-invariant ~750 ms duration and correctness drift implicate runtime
coordination/collective ordering in addition to resource contention. It is
nevertheless the faithful result for the available minimal naive execution
contract: it provides no usable overlap.

## 11. Split-tax decomposition

The evidence localizes costs at two levels:

1. **Physical fragmentation (dominant before concurrency):** N1 costs +40.37%
   at the median. The 126-cut atlas independently shows both Attention and MoE
   fragment sums at ~2.41x unsplit. This combines extra kernel launches,
   duplicated per-fragment setup/QKV work, attention metadata and K/V access,
   smaller MoE routing/packing batches, and smaller expert GEMMs.
2. **Concurrent runtime/collective tax (dominant in N2):** N2 adds another
   ~289–317% relative to N1. The two-stream wrapper invokes shared EP machinery
   concurrently, creating synchronization/progress overhead and invalid
   numerical behavior on the largest input.

No evidence shows that simple tensor-copy elimination would recover most of
either cost. “Zero-copy” attacks only one component and cannot remove the
observed launch/shape and concurrent-collective contract problems.

## 12. Contention-corrected oracle

The raw measured overlap efficiency is negative for all three requests. For a
conservative policy oracle, negative efficiency is clamped to zero and the N1
physical split tax is still charged. This produces raw projected TTFT changes
of −52.06%, −23.04%, and −39.16%; the multimodal median is **−39.16%**.
Allowing a runtime to choose stock instead gives a best-policy improvement of
**0%**. Both are below the required ~5% corrected-headroom threshold.

Moving only Dispatch/Combine to another stream was not separately implemented:
the full measured exact-fragment O1 is negative in every layer, modality carries
no useful cut signal, and N2 violates correctness. A communication-only split
could diagnose the wrapper, but it cannot rescue the Phase-0 research claim or
meet the request-level gate under these observations.

## 13. Limitations

- The proportional O1–O4 values are explicitly optimistic analytical ceilings;
  linear progress is rejected rather than validated.
- Baseline TTFT has three independent engine restarts, but N0/N1/N2 are one
  bounded restart per policy with two clean repetitions per workload. The
  negative effects are 39–511%, far above restart noise, but they are not a
  five-restart positive claim.
- The physical split is implemented through current vLLM metadata scopes and
  kernel calls. It does not emulate a hypothetical future FlashAttention kernel
  exposing tile-completion events at zero cost.
- Three multimodal requests cover standard, high-resolution, and multi-image
  cases. They are sufficient to falsify the proposed modality-boundary rule but
  not to characterize every prompt layout.
- The N2 auxiliary add-on flush hung after primary data persistence. The
  analysis excludes the missing add-on trace and uses only complete primary
  forward/layer records.

These limitations make the ideal ceiling uncertain upward, but do not support
continuing this particular direction: a paper-level successor would require a
new kernel/collective execution contract and would still lack a modality-aware
signal.

## 14. Final GO/HOLD/NO-GO

| Gate | Result |
|---|---|
| G0 runtime/timing validity | PASS with attribution caution (7.05% observer tax) |
| G1 affected-region relevance | PASS for W1/W2/W4; W3 40.8% still material |
| G2 optimistic ideal TTFT | PASS narrowly: O1 median 15.47% |
| G3 modality relevance | FAIL: O3 median 1.59%; text-only behaves similarly |
| G4 overlap | FAIL: median raw eta −16.75; N2 correctness drift |
| G5 split-tax removability | FAIL: N1 +40.37%; fragment ratios ~2.41x |
| Contention-corrected headroom | FAIL: stock lower envelope 0% |

The decisive result is not merely “the naive implementation is slow.” The
mathematical headroom exists only under an empirically rejected token-linear,
zero-fragment-cost assumption; actual exact fragment kernels remain slower even
with perfect overlap, and the suggested modality boundary contains almost none
of the theoretical opportunity. Production scheduler, custom kernel, and
Phase-1 zero-copy work should not be pursued for this hypothesis.

NO-GO
