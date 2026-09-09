# MLLM MoE Transient Branch Compression PoC

**Final status:** `NO_OUTPUT_COMPRESSIBILITY`

**Date:** 2026-09-10 (KST)

**Model:** Qwen3-VL-30B-A3B-Instruct, BF16

**Runtime:** vLLM 0.20.0+cu129, DeepEP 1.2.1+73b6ea4

**Topology:** TP2 / DP2 / EP4, DeepEP high-throughput, TritonExperts, DBO off

**Physical GPUs:** H100 4,5,6,7 only

**Result root:** `poc_flashvep/deepep_revalidation/results/mllm_moe_transient_branch_compression_poc_20260910_021423/`

## Executive decision

The proposed operation—retain every visual token and original top-k route, but
reuse selected same-expert branch outputs—fails before implementation. An
impossible grouping oracle that sees the true expert outputs achieves **0%
median strict-safe visual branch reduction** for every group cap 2/3/4/6/8;
the best single sample/layer reaches only **0.0283%**. Qwen's experts generally
expand, rather than contract, differences among nearby visual states. Even a
new exact router-weighted centroid evaluation at the smallest group size has
about **32.3%** affected-update relative-L2, versus the 1% gate.

The result is independently killed by downstream propagation and economics. A
favorable output-aware 5%-row intervention lowers next-router agreement to
97.83% after one layer and 41.93% across six selected layers. The measured
high-resolution request spends only 7.71% of E2E in the whole MoE span and
3.26% in expert compute; therefore even magical removal of every expert GEMM
cannot meet the 8% implementation gate. The quality-safe O1/O2 oracle rounds
to zero, and O3 is necessarily non-positive after mapping and reconstruction.

This is a gate-driven early stop, not an environment failure. The live runtime
path, branch capture, checkpoint replay, 27 tested causal questions, all seven
specified adjacent successors, and four forced rethink points were completed.
A compact DeepEP method prototype and Kimi transfer were correctly not entered
after both prerequisite gates failed.

## 1. Evidence and runtime fidelity

Fresh live capture used the intended production-style path:

- two DP drivers, each TP2, with `enable_expert_parallel=True`;
- EP world size 4 and EP ranks 0–3;
- `DeepEPHTAll2AllManager` and `DeepEPHTPrepareAndFinalize`;
- `TritonExperts` local execution;
- physical CUDA visibility exactly `4,5,6,7`;
- DBO disabled;
- identical one-token greedy output from both DP drivers.

The capture contains nine real images across natural, fine-grained, and
chart/document content at 336/448/672-pixel edges and layers
4/12/24/36/44/47. It includes 2,211 visual tokens, 106,128 visual routed branch
assignments, 2,436 prompt tokens, and 358,291 evaluated same-expert pairs.
Observer-heavy capture was never used as clean performance timing.

Captured weighted branches reconstruct the stock combined output with minimum
cosine 0.9999943 and maximum relative-L2 0.3412%. More importantly, an exact
checkpoint replay across 27 request/layer rows has median relative-L2 0, worst
p99 `1.56e-8`, and minimum cosine 0.9999983. The geometric result is therefore
not a reconstruction artifact.

Two early launches in a generic environment lacked DeepEP and stopped before
model execution. They are recorded as port failures and contribute no method
evidence.

## 2. Output-level compressibility

### 2.1 Pair geometry

| Modality / pairing | Input rel-L2 median | Expert-output rel-L2 median | Output/input ratio | Pair screen pass |
|---|---:|---:|---:|---:|
| Vision arbitrary | 1.0186 | 1.1681 | 1.1304 | 0.029% |
| Vision hidden-nearest | 0.6749 | 0.8309 | 1.2173 | 0.150% |
| Vision output-nearest oracle | 0.6893 | 0.8162 | 1.1750 | 0.162% |
| Vision 1D contiguous | 0.8072 | 0.9672 | 1.1844 | 0.063% |
| Vision true 2D neighbor | 0.7783 | 0.9347 | 1.1889 | 0.091% |
| Text arbitrary | 1.1038 | 1.2455 | 1.1171 | 0% |
| Text hidden-nearest | 0.9462 | 1.0987 | 1.1464 | 0% |
| Text output-nearest oracle | 0.9613 | 1.0830 | 1.1177 | 0% |

The pair screen is deliberately looser than the group acceptance gate
(cosine>=0.99 and rel-L2<=10%); even under it, safety is almost absent. The
group oracle additionally requires branch-local safety and combined-update
cosine>=0.9999 / rel-L2<=1%.

At common input-distance support within the same layer, expert, and pair type,
Vision output error is 0.033–0.072 lower than Text. This small relative
modality effect is real, but it is not an operational compression regime:
Vision's impossible output-nearest median error remains 81.6%.

The main hidden assumption is false. Same-expert routing does not mean the
expert maps distinct visual inputs to a common output. The median ratio is
above one for every general pairing. Layer 47 alone contracts
output-nearest pairs (ratio 0.739), but its absolute output error is still
40.8%, pair-screen pass is below 1%, and group-safe reduction peaks at only
0.0283%.

### 2.2 Does expert-output similarity add information?

Input/output distance Spearman correlation is 0.881. An impossible
output-nearest selector improves median output relative-L2 only 0.831→0.816
and pair-screen pass only 0.150%→0.162% over hidden-nearest selection.

A leave-one-request-out control over 338,490 visual pairs produced:

| Features added after input distance + layer/expert/policy | Median held-out RMSE reduction |
|---|---:|
| Whole top-k route Jaccard | 0.034% |
| Router-weight pair features | 0.419% |
| Spatial distance | 0.035% |

Among 14,727 visual pairs with route Jaccard >=0.75, median output relative-L2
is 0.606 and no pair passes the 1% strict tolerance. Thus route, weight, and
spatial context cannot recover a deployable safety certificate.

### 2.3 Group oracle and representatives

The output-aware group oracle reports:

- median safe branch reduction **0%** at caps 2,3,4,6,8;
- maximum sample/layer reduction **0.0283%**;
- exactly 0% at layers 4–44;
- 0% for Text;
- median per-rank reduction 0%.

Representative choice is not the bottleneck. For cap-2 contiguous groups, the
affected-token combined-update relative-L2 medians are anchor 48.1%, output
medoid 46.1%, exact centroid 33.5%, and exact router-weighted centroid 32.3%.
All have median strict pass 0%. A newly evaluated centroid is stronger than
reuse of any existing token yet still misses the 1% gate by about 32×.

## 3. Spatial and RLE-like route structure

Route structure itself is common:

- 1D same-expert run rows at length>=2: median 22.22%, p90 58.82%;
- 1D rows at length>=4: median 0%, p90 17.50%;
- 2D connected-component rows at size>=2: median 42.86%, p90 79.55%;
- 2D rows at size>=4: median 0%, p90 57.55%.

This is a useful negative distinction: **route-run compressibility is not
expert-function compressibility**. The 1D and 2D pair-screen passes are only
0.063% and 0.091%, compared with 0.162% for the already unusable output
oracle. At the pair level they recover roughly 39% and 56% of that tiny oracle
pass rate, but at the actual group gate every deployable RLE/window policy
recovers 0% median safe branch reduction.

Exact packing without approximation was evaluated separately. The complete
router plus DeepEP-layout kernel envelope is 49.9 ms in a roughly 4.3 s prior
trace, at most 1.16% even if both categories vanished. A route-run packing
method can remove only a subset, so it fails the 8% child gate.

## 4. Quality propagation

The fresh single-GPU HF diagnostic preserves all token identities and original
routes/weights. It tests an impossible captured-output oracle, same-budget
lowest-contribution skipping, and spatial sharing. It is quality evidence only,
not EP performance evidence.

| Scope / policy | Logit rel-L2 | Logit cosine | Next-route agreement | Short greedy exact |
|---|---:|---:|---:|---:|
| Layer 44, output oracle, 4 requests | 6.85% | 0.99765 | 97.83% | 4/4 |
| Layer 44, skip | 7.54% | 0.99715 | 97.76% | 4/4 |
| Layer 44, spatial | 7.46% | 0.99721 | 97.46% | 4/4 |
| Six layers, output oracle, 8 requests | 23.00% | 0.97338 | 41.93% | 8/8 |
| Six layers, skip | 23.41% | 0.97242 | 41.18% | 8/8 |
| Six layers, spatial | 23.66% | 0.97195 | 40.65% | 8/8 |

The apparent 8/8 short greedy agreement is not promoted: internal routing and
logits have already diverged severely, and the tiny answer set is insensitive.
A full benchmark was not run because every policy failed the registered local
and >=99% next-route screens. No claim about benchmark accuracy is made.

## 5. Sharing versus expert skipping

At matched visual-row budgets, the true-output sharing oracle is only slightly
better than zeroing the smallest contributions:

| Row budget | Output-share affected rel-L2 | Skip affected rel-L2 | Spatial-share affected rel-L2 |
|---:|---:|---:|---:|
| 1% | 6.07% | 7.16% | 9.00% |
| 2% | 8.87% | 9.68% | 11.30% |
| 5% | 11.75% | 12.92% | 15.93% |
| 10% | 16.39% | 17.60% | 20.35% |
| 25% | 26.15% | 28.44% | 36.39% |

Strict affected-token pass is 0% for sharing and skipping at every budget.
Sharing therefore does not produce a distinct quality–efficiency Pareto point;
when only low-weight rows can be perturbed, the idea collapses toward the
expert-skipping space occupied by MoDES, AnyExperts, and ACE [3,4,7].

## 6. Economic oracle

The clean high-resolution Qwen3-VL c8 trace has per-layer T_MoE 1.218 ms,
48 MoE layers, request p50 758 ms, and expert share 42.25%. The deliberately
optimistic ceilings are:

| Counterfactual | Direct request-E2E upper bound |
|---|---:|
| Remove the complete MoE span | 7.71% |
| Remove every expert computation, retain dispatch/combine | 3.26% |
| Remove all gate/up expert work | 2.17% |
| Remove all down-projection work | 1.09% |
| Remove a hypothetical 5% of expert rows | 0.16% |
| Remove the measured strict-safe rows | approximately 0% |

- **O1 compute-only:** approximately 0%, because strict-safe row reduction is
  zero in the median.
- **O2 communication-aware:** approximately 0%; there are no safe compacted
  rows whose dispatch/combine traffic can scale.
- **O3 feasible:** <=O2 after map creation, gather/scatter, compact-list
  formation, reconstruction, metadata, padding, and possible grouped-GEMM
  degradation. It cannot be positive in the aggregate.

This is stronger than a pessimistic cost model: even a free, perfect codebook
that removes *all* expert compute is capped at 3.26%. Assignment reduction
cannot translate into the required 10–12% request benefit on this four-GPU
regime.

## 7. Adjacent successor search

All specified children were screened after the main oracle failed.

1. **Sub-expert stage:** sharing gated intermediate `z_j` and applying the down
   projection yields exactly `E_e(h_j)`, already bounded by the output oracle.
   Down-only and gate/up-only ceilings are 1.09% and 2.17% E2E.
2. **Contribution-weighted sharing:** slightly outperforms skipping locally but
   remains far outside quality gates, has <=0.16% ideal E2E at a 5% budget,
   and collides with contribution-aware skipping literature.
3. **Eligible layers:** only layer 47 is relatively favorable; no 8–16-layer
   safe subset exists.
4. **Exact spatial packing:** total relevant overhead envelope is <=1.16%.
5. **Expert codebook/low rank:** even free removal of all expert compute is
   <=3.26% E2E.
6. **Cheap repair:** scalar/norm correction cannot repair direction error;
   exact centroids are more expressive and still have ~32% error.
7. **Opposite result:** experts amplify nearby visual differences. This is a
   useful warning for pre-MoE merging, but it exposes no >=10% direct oracle.

Twenty-seven causal questions are recorded in `RESEARCH_TREE.md`, including
forced rethinks after H05/H10/H15/H20. No adjacent candidate has >=10% direct
headroom.

## 8. Prior-art and novelty assessment

Before measurement, same-forward cross-token branch sharing was distinct from
FastMMoE's token/width reduction [1,2], MoDES/AnyExperts contribution removal
[3,4], XShare's shared expert subset [5], SERE's expert substitution [6], and
FastV/SparseVLM token pruning [9,10]. MoECa is the closest conceptual neighbor
because it reuses expert-branch features across diffusion timesteps [8]; it
already makes generic “branch reuse” an unsafe novelty claim. SpecMoE instead
speculates tokens and verifies them [11]. Standard vLLM Omni EP retains exact
dispatch/execute/combine semantics [12].

Thus an exact semantic gap existed, but novelty is immaterial after the real
Qwen outputs reject its premise. No production contribution should be framed
from route runs or low-weight rows alone: those axes are already adjacent to
FastMMoE, MoDES, XShare, and ACE.

## 9. Direct answers to the required questions

1. **Are Vision branch outputs more compressible than Text?** Slightly in a
   matched relative sense, yes; operationally, no. The best Vision oracle still
   has 81.6% median output relative-L2 and 0% median group-safe reduction.
2. **Does output similarity add information beyond hidden cosine?** Only a tiny
   amount: 0.831→0.816 median error and +0.012 percentage points pair-screen
   pass. It does not create a usable safety regime.
3. **How much output oracle does spatial/RLE grouping recover?** Pair-screen
   rates are ~39% (1D) and ~56% (2D) of an oracle that is itself 0.162%; at the
   real group gate both recover 0% median safe reduction.
4. **How many rows can be removed without whole-token pruning?** Strict median
   0%; maximum 0.0283% in one sample/layer. Forced reductions are large but do
   not preserve the local quality contract.
5. **Are next-layer routing and final quality retained?** No under the registered
   screen. Route agreement is 97.83% after one modified layer and 41.93% after
   six. Short greedy outputs happened to match, but benchmark promotion was
   disallowed and benchmark preservation is not claimed.
6. **What request-E2E gain follows from assignment reduction?** Quality-safe
   O1/O2 are approximately 0%, and O3 cannot be positive. A hypothetical 5%
   expert-row reduction is capped at 0.16%; eliminating every expert compute is
   capped at 3.26%.
7. **Exact difference from FastMMoE/MoDES/MoECa?** This PoC preserves tokens,
   original routes, and router weights while sharing `E_e(h)` across same-forward
   token branches. FastMMoE changes tokens/width, MoDES removes expert
   contributions, and MoECa reuses branches across diffusion timesteps. The
   distinction is real, but the measured phenomenon is not.
8. **Does the sub-expert child survive?** No. It is algebraically dominated by
   full-output substitution and independently capped below 2.17% E2E.
9. **Was another >=10% oracle found?** No. Contribution weighting, eligible
   layers, exact packing, codebooks, and repair all fail quality or direct
   headroom gates.
10. **Should this research continue?** No for transient branch sharing or its
    listed successors on this regime. The correct next action is to stop method
    engineering, not to relax the evidence threshold.

## 10. Evidence limitations

- The fresh set has nine images, three content categories, three resolutions,
  and six layers rather than a full benchmark.
- Quality propagation is single-GPU HF execution, not a clean EP speed test.
- No Kimi validation or compact DeepEP prototype was run.
- Economic mapping uses a previously measured clean request trace rather than
  timing the rejected method.

These limitations restrict broad quality claims, but do not create a plausible
false negative for the decision gates: the impossible oracle misses row safety
by over 500×, the best centroid misses local error by ~32×, and the entire
expert-compute ceiling is independently below the implementation gate.

## 11. Final classification

| Gate dimension | Outcome |
|---|---|
| Geometry | **FAIL** — expert-induced expansion; oracle-safe reduction 0%. |
| Quality | **FAIL** — one- and six-layer route/logit drift. |
| Economic mapping | **FAIL** — all expert compute <=3.26% request E2E. |
| Implementation | Not entered by spec after upstream gates failed. |
| Novelty | Semantic gap existed; no supporting phenomenon or headroom. |

**Final status: `NO_OUTPUT_COMPRESSIBILITY`.**

## Sources

1. [FastMMoE paper](https://arxiv.org/abs/2511.17885)
2. [FastMMoE official repository](https://github.com/MindVLA-Team/FastMMoE)
3. [MoDES, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MoDES_Accelerating_Mixture-of-Experts_Multimodal_Large_Language_Models_via_Dynamic_Expert_CVPR_2026_paper.html)
4. [AnyExperts](https://arxiv.org/abs/2511.18314)
5. [XShare](https://arxiv.org/abs/2602.07265)
6. [SERE](https://arxiv.org/abs/2602.07616)
7. [ACE](https://arxiv.org/abs/2609.05228)
8. [MoECa](https://arxiv.org/abs/2606.15615)
9. [FastV](https://arxiv.org/abs/2403.06764)
10. [SparseVLM](https://arxiv.org/abs/2410.04417)
11. [SpecMoE](https://arxiv.org/abs/2604.10152)
12. [vLLM Omni expert-parallel design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/expert_parallel/)
