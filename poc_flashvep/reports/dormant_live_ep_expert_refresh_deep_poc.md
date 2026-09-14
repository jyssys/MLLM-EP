# Dormant-Live EP Expert Refresh: deep PoC report

## Executive decision

**Final label: `CHARACTERIZATION-SIGNAL`.**

LLaDA2.0-Flash contains many still-MASK tokens that are more than one
refinement step from their baseline acceptance time: 29.65% of logical
token-row opportunities on GSM8K and 28.46% on HumanEval at H=1.  That is a
real dLLM lifetime phenomenon.  It does **not** become a paper-level routed-EP
refresh opportunity:

- those rows account for 8.09%/8.61% optimistic request E2E in the removable
  dispatch+expert+combine path;
- preserving calibrated DeepEP fixed cost lowers even perfect, future-aware
  removal to 6.95%/7.27%;
- DORMANT routes and expert outputs are less stable than ACTIVE ones;
- periodic stale reuse changes NFE, acceptance order, and most final sequences;
- route/destination invalidation improves safety but leaves only 0.63%/0.54%
  projected E2E;
- a held-out, high-precision future-free joint proxy projects 2.37%/1.30%, with
  expert features adding essentially no value beyond semantic features.

There is no measured speedup: causal runs computed all exact work and replaced
outputs afterward.  The study stops before a variable-row DeepEP runtime in
accordance with the gate.

## Scope and evidence contract

The frozen substrate is LLaDA2.0-Flash 100B, BF16, batch/mini32, generation and
block length 32, threshold 0.9, dense TP4 + routed EP4 on physical GPUs 4--7.
The actual path is DeepEP normal dispatch, 64 owner-local routed experts per
rank, Triton fused expert execution, and reverse combine.  All GPU launches
used `CUDA_VISIBLE_DEVICES=4,5,6,7`; log-visible logical IDs 0--3 are the
expected remapping.

Evidence is split into:

1. **Measured clean request:** three independent process launches per task.
2. **Measured structural/causal:** observer-heavy route/value traces and output
   interventions; their wall time is excluded.
3. **Analytical oracle:** state-attributed CUDA-event mass normalized to an
   identical-substrate clean trace and fixed-cost-corrected with measured
   DeepEP payload curves.

The clean medians are 5.925s/NFE66 for GSM8K and 7.261s/NFE86 for HumanEval.
GSM8K's first cold launch was 8.839s and is retained in the log.

## Stage 0: Window-Diffusion positive control

Official code at `d8bb349` reproduces the active/buffer/far-field state
transition in a deterministic code-level smoke.  Its full supported
LLaDA-8B-Base checkpoint was unavailable, so model-level quality/speed is
`ENVIRONMENT-LIMITED` and not counted.  The [paper](https://arxiv.org/abs/2601.20332)
and [official repository](https://github.com/vhicrgit/Window-Diffusion) support
only the conceptual premise that still-MASK tokens need not all occupy the
same compute window.

## Dormant census and cost mapping

| task | H | ACTIVE | DORMANT | DEAD | perfect feasible post-Epoch E2E |
|---|---:|---:|---:|---:|---:|
| GSM8K | 0 | 7.40% | 35.84% | 56.76% | 8.55% |
| GSM8K | 1 | 13.59% | 29.65% | 56.76% | **6.95%** |
| GSM8K | 2 | 18.89% | 24.35% | 56.76% | 5.61% |
| GSM8K | 4 | 27.52% | 15.72% | 56.76% | 3.52% |
| GSM8K | 8 | 37.81% | 5.43% | 56.76% | 1.15% |
| HumanEval | 0 | 10.33% | 34.75% | 54.92% | 8.88% |
| HumanEval | 1 | 16.63% | 28.46% | 54.92% | **7.27%** |
| HumanEval | 2 | 21.51% | 23.58% | 54.92% | 6.01% |
| HumanEval | 4 | 29.11% | 15.97% | 54.92% | 4.02% |
| HumanEval | 8 | 38.42% | 6.66% | 54.92% | 1.64% |

H=0 uses impossible future information to call every not-currently-accepted
MASK dormant; H=1 is the primary near-frontier guard.  H=1 DORMANT rows are
18.80%/15.13% of timing-aligned physical MoE rows and 18.75%/15.14% of remote
assignments.  They map to 8.09%/8.61% optimistic removable request time, but
only 6.95%/7.27% when communication floors are kept.  The logical census and
physical-row share intentionally use different denominators.

## The decisive opposite result

| task/state (H=1) | exact top-k | exact destination | hidden cosine | routed output cosine | routed output rel-L2 |
|---|---:|---:|---:|---:|---:|
| GSM ACTIVE | 18.31% | 60.91% | 0.945 | 0.882 | 0.515 |
| GSM DORMANT | 10.96% | 53.51% | 0.895 | 0.781 | 0.689 |
| Human ACTIVE | 19.13% | 61.37% | 0.951 | 0.895 | 0.486 |
| Human DORMANT | 9.70% | 52.96% | 0.886 | 0.774 | 0.695 |

Future-dormant tokens are less stable on every value needed for routed-output
reuse.  They are dormant with respect to *acceptance*, not representation
evolution.  This invalidates the central computational analogy with a dormant
cache tier.

## Oracle and causal tournament

| H=1 policy | GSM feasible E2E | Human feasible E2E | causal result |
|---|---:|---:|---|
| perfect future removal | 6.95% | 7.27% | impossible upper bound; not executed |
| periodic K2 | 3.18% | 2.96% | exact sequences 12/32, 16/32 |
| periodic K4 | 4.42% | 4.37% | exact sequences 10/32, 12/32 |
| periodic K8 | 4.98% | 5.15% | exact sequences 5/32, 10/32 |
| route/destination K8 | 0.63% | 0.54% | exact sequences 17/32, 20/32 |

The bounded task scores sometimes remain equal—baseline scores are only 5/32
and 6/32—but that is not evidence of equivalence.  All non-baseline policies
change NFE, acceptance timing, and final token sequences.  The route-gated
policy is directionally safer yet still far from exact and economically dead.

## Deployability

At a training-selected >=99% dormant precision threshold, held-out semantic
features recover 33.80%/18.91% of the future oracle, projecting 2.35%/1.38%
E2E.  Expert-only features recover 0.054%/0%; joint features project
2.37%/1.30%.  Expert state therefore provides no robust extra signal over the
semantic proxy.  No controller or live physical skipping path was built.

## Prior-art and novelty attack

[Window-Diffusion](https://arxiv.org/abs/2601.20332) partitions active, buffer,
and far-field masked tokens; [Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache)
uses sliding-window, attention-aware, layer-aware KV refresh;
[SureLock](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e82cfb0ee6ce329759d0d3c90fbbccc4-Abstract-Conference.html)
locks converged/unmasked tokens; [Epoch](https://arxiv.org/abs/2609.09748)
compacts live/new/refresh-required expert rows; [REFLEX](https://arxiv.org/abs/2608.01784)
changes refinement-aware expert budget; and [TEAM](https://arxiv.org/abs/2602.08404)
changes decoded caching and expert activation/decoding behavior.

The surviving novelty boundary would have been normal top-8 routed semantics
on refresh, but reduced refresh frequency for still-MASK future-dormant rows,
with additional post-Epoch EP savings.  The measurements fail before novelty
becomes the limiting issue.

## Answers to the 20 required questions

1. **Window structure reproduced?** Code-level active/buffer/far behavior yes;
   full dense-model reproduction environment-limited.
2. **Future-dormant MASK fraction?** H=1: 29.65% GSM8K, 28.46% HumanEval.
3. **Routed-EP cost?** 8.09%/8.61% optimistic request E2E for fresh-router
   dispatch+expert+combine.
4. **After Epoch DEAD removal?** The same 8.09%/8.61% optimistic residual;
   fixed-floor perfect ceiling 6.95%/7.27%.
5. **More route-stable than ACTIVE?** No, exact top-k is about 10% versus 19%.
6. **Destination more stable?** No, exact set is about 53% versus 61%.
7. **K-step refresh quality-safe?** No full-trajectory policy is safe.
8. **Fresh router improves safety?** Yes directionally, but not to equivalence.
9. **Route-change trigger improves Pareto?** Safety improves; economics collapse
   to 0.63%/0.54% E2E.
10. **Shared-fresh/routed-stale valid?** It isolates the target causally but
    still changes most trajectories.
11. **Perfect removal oracle?** H=1 optimistic 8.09%/8.61%, feasible
    6.95%/7.27%.
12. **Quality-safe oracle?** No validated nonzero full-trajectory point; tested
    periodic points are 2.96--5.15% projections and unsafe.
13. **Independent post-Epoch oracle?** Perfect feasible mean 7.11%; route-safe
    mean 0.59%.
14. **Future-free proxy?** Technically possible at low recall; economically
    only 1.30--2.37% projected E2E.
15. **EP features add information?** Negligible on GSM8K and negative on
    HumanEval at the safety threshold.
16. **Remote bytes/expert rows reduced?** Oracle H1 removal is about
    18.7%/15.1%; causal periodic K8 would defer 13.3%/10.4% remote payload.
17. **Rows become latency?** Only partially; DeepEP floors reduce H1 perfect
    headroom by 1.14/1.33 percentage points before controller overhead.
18. **Complementary to Window/Elastic?** Only conceptually for remaining buffer
    rows; no material safe residual was demonstrated.
19. **Distinct from REFLEX/TEAM?** Mechanistically yes—refresh frequency rather
    than top-k/decoding—but no viable operating point survives.
20. **Paper-level headroom?** No on this substrate.

## Reproduction artifacts

- Machine-readable derived results: `poc_dormant_live_ep_refresh/*.csv`.
- Reports: `poc_dormant_live_ep_refresh/reports/`.
- Acceptance plans and analysis scripts: `poc_dormant_live_ep_refresh/plans/`
  and `scripts/`.
- Raw campaign: `poc_dormant_live_ep_refresh/results/dormant_live_ep_refresh_20260914_152829/`.
- The runtime instrumentation is preserved as a patch under
  `poc_dormant_live_ep_refresh/patches/`.

No measured latency improvement is claimed.  The correct conclusion is that
the Dormant-Live population is a characterization signal, while sparse routed
expert refresh is a quality/economics NO-GO for LLaDA2.0-Flash EP4 at this
operating point.
