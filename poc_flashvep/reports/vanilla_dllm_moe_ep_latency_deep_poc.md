# Vanilla dLLM MoE EP latency deep PoC

## Decision

**CHARACTERIZATION-ONLY.** Vanilla SDAR has a large reference-EP penalty and
strong temporal routing/load persistence, but every independent candidate fails
the pre-registered 5% feasible request-E2E implementation gate. The primary
rank-local-refinement idea yields 0% feasible gain; the strongest exact oracle
(block-hot replicas) is only 0.153%. No production method or TEAM adaptation is
justified.

This study starts from vanilla SDAR-30B-A3B and asks whether a training-free
system can remove the frequency or scope of full global expert-parallel
refinement. TEAM is only a motivation and optional composition control.

## Evidence boundary

- **Measured clean:** fixed-output-budget request latency from randomized
  Single/EP2/EP4 restarts.
- **Measured instrumented:** same-device CUDA-event stage spans, temporal route
  and hidden/contribution similarity, local-draft proposal behavior, P2P copy,
  and same-shape EP-degree replays.
- **Oracle:** any future-aware selection, proportional load-to-time mapping, or
  verification-coalescing projection. These are never presented as live gains.
- **Runtime limitation:** sparse ownership and remote execution are real, but
  transport is an exact NCCL `all_to_all_single` reference path rather than
  DeepEP. All communication-derived conclusions therefore include a
  1.0/0.75/0.5/0.25 cost sensitivity.

## Fair vanilla scaling

On 4 GSM8K + 4 HumanEval prompts, three balanced-order engine restarts executed
an identical fixed output budget: 1,166 forwards and 14,487,552 token-expert
assignments per run. Median bounded quality was 62.5% for every topology.

| Topology | 8-request clean wall | Mean request | vs Single | Remote assignment fraction | Remote hidden traffic |
|---|---:|---:|---:|---:|---:|
| Single | 161.511 s | 20.091 s | 1.000x | 0% | 0 |
| EP2 | 184.931 s | 23.230 s | 1.145x | 49.95% | 59.283 GB |
| EP4 | 192.096 s | 24.437 s | 1.189x | 74.94% | 88.936 GB |

The exact reference EP4 path is 18.9% slower than Single and 3.9% slower than
EP2. This is not a claim about DeepEP: the verified substrate deliberately uses
separate NCCL A2As for hidden rows and metadata, reverse A2A for branches, and a
dense-state broadcast. A separate stage trace localizes the EP-degree growth to
route/split preparation and communication; its observer tax (12.6%/13.1%/67.3%
for EP1/2/4) prevents treating those detailed spans as clean additive wall.

## Vanilla temporal structure

At EP4 and lag 1, 82.18% of routed branches persist, rank-destination sets match
for 80.02% of tokens, hot-expert Jaccard is 76.85%, and rank-load cosine is
0.9971. Yet the hidden state changes by 27.6% relative L2, stable branch outputs
by 32.4%, and the combined MoE update by 49.1% (cosine 0.829). Routing/load is
predictable; numerical work is not reusable as if unchanged.

Exact layout stability is weaker still: the full rank-count vector repeats in
only 0.144% of adjacent layer/iteration pairs, ordered owner vectors in 16.12%
of tokens, and owner multisets in 43.73%. This kills wholesale route-plan reuse
despite aggregate-load predictability.

## Candidate tournament

| Candidate | Best cap/oracle | Feasible request gain | Decision |
|---|---:|---:|---|
| Rank-local refinement + global verification | 0% | 0% | KILL |
| Multi-step local drafting | 0.000052% | 0% | KILL |
| Exact block-hot replica cache | 0.153% | 0.153% | KILL |
| Elastic EP degree | 0% | 0% | KILL |
| Confidence-gated EP scope | no quality-safe oracle | 0% | KILL |
| Exact temporal delta | 0% (FP8 free-codec cap 3.11%) | 0% | KILL |
| Dependency-safe overlap | free-communication cap 6.23% | 0% | KILL |
| Rank-coherent near-tie routing | 0.59% proportional cap | 0% | KILL |
| Exact route-layout reuse | <0.1% loose cap | 0% | KILL |

The primary live diagnostic is decisive. A rank-local draft costs 66–70% of a
global exact forward. At the best local top-k=4, four-view union hit is 54.21%;
4/4 consensus is 88.86% precise but covers only 5.95%, while 3/4 consensus has
only 56.03% precision. Even granting independent validity and a free perfect
trajectory verifier, K=4 reduces model-forward work by only 0.000052%.

Elastic replay is equally decisive: EP1 is fastest at every M from 1 through
256 and at every captured request step, so the per-iteration dynamic oracle is
identical to best static. Exact expert copies are cheap (0.0619 ms for one
9.437 MB expert), but replica-rank compute inflation erases communication
savings. Its best future-aware point is 0.153% request E2E.

## Prior-art and durability

The surviving framings are also crowded. Fewer-expert MoE drafting is covered by
Self-Speculative MoE, DraftExpert, and SpecMoE; valid bidirectional dLLM
verification is covered by SimSD and trajectory-level speculation. Exact expert
replication has predictive-replication/TIDE neighbors; layer refresh and stale
communication collide with Epoch/DICE; marginal EP-cost action selection is
adjacent to EcoSpec. The only initially plausible gap—using existing EP rank
partitions as parallel dLLM refinement views—fails the measured cost and
consensus gates.

Durability is negative rather than fragile-positive. At 0.75× communication
cost, the exact replica oracle falls from 0.153% to 0.033%; at 0.5× it becomes
zero. The impossible free-communication cap falls from 6.23% to 3.11% at 0.5×.
A more production-capable transport makes these candidates less attractive.

## Final answers

1. **Fair vanilla scaling:** Single 161.511 s, EP2 184.931 s, EP4 192.096 s for
   the same eight-request work; EP4 is 18.9% slower than Single.
2. **EP4 bottleneck:** reference route preparation and dispatch/combine grow;
   local expert compute does not compensate at M≤256.
3. **dLLM temporal structure:** route, rank destination, and load persist, but
   hidden/expert outputs change materially and exact layouts rarely repeat.
4. **Rank-local proposal quality:** insufficient; union and consensus coverage
   are too small relative to 66–70% draft cost.
5. **Can global EP invocation count fall?** Not with this training-free physical
   rank-view mechanism; the optimistic multi-step oracle is effectively zero.
6. **Hot replicas:** measured copy amortization exists, but critical-rank work
   moves rather than disappears; net oracle 0.153%.
7. **Elastic EP:** no crossover, hence 0% dynamic headroom over best static.
8. **Conditional scope/overlap:** no quality-safe scope and no independent exact
   batch-one slack.
9. **Strongest independent candidate:** exact block-hot replica cache, still only
   0.153% and prior-art-adjacent.
10. **Faster runtime:** all communication-based headroom shrinks; none survives.
11. **TEAM adaptation:** not run because the vanilla gate failed.
12. **Next research:** do not pursue this candidate family on this substrate;
   move to batch/concurrency-level dLLM work where independent requests create
   real slack, or to a production EP backend characterization before proposing
   a communication method.

Detailed reports and plots are under `poc_vanilla_dllm_ep/`. Raw observer-heavy
rank JSON is intentionally excluded from Git; derived CSV/JSON and all figures
are retained.

The adversarial source-by-source comparison and direct links are in
`poc_vanilla_dllm_ep/reports/11_prior_art_audit.md`.
