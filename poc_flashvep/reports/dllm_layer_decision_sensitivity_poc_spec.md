# dLLM MoE EP Layer/Decision Sensitivity PoC — Working Contract

## Objective

Discover, rather than assume, whether LLaDA2.0-Flash 100B true-EP4 spends
material latency on transformer-layer or routed-expert computation that has
little causal influence on diffusion token acceptance/unmask decisions.  A
method is considered only after causal intervention, request-level cost
mapping, and final-generation quality validation establish a strong signal.

## Fixed substrate and safety

- Use only physical GPUs `0,1,2,3`; never touch GPUs 4–7.
- Primary model: local verified revision of `inclusionAI/LLaDA2.0-flash`.
- Runtime: dense TP4 plus routed EP4, 256 routed experts/top-8, 64 experts per
  rank, DeepEP dispatch, owner-rank fused expert execution, reverse combine.
- Reproduce the strongest prior best-static setting, not a weak default:
  submitted batch 32, `mini_batch_size=32`, generation 32, block length 32,
  config 42, threshold 0.9, BF16.
- Use the same bounded GSM8K and HumanEval samples for baseline and every
  causal intervention.
- Task-owned burn may run only while the experiment is idle.  Confirm PID,
  owner, command, UUIDs and free memory before stopping it.  Restart burn on
  GPUs 0–3 after the final handoff.

## Evidence order

1. Reconfirm clean best-static quality/latency baseline.
2. Capture observer-separated layer × refinement-state traces.
3. Build layer stability maps; do not equate similarity with necessity.
4. Apply causal layer/routed/shared/stale interventions.
5. Measure current decision, future trajectory, final answer and physical
   latency cost independently.
6. Compute request-level necessity/removal oracles.
7. Generate method candidates only if the oracle gate is passed.
8. Audit prior art adversarially only for promoted candidates.

## Required traces

For each request/block/refinement iteration/layer, capture phase, live/masked/
accepted rows, confidence and margin; input/output/update norms; cosine and
relative L2; attention and MoE update norms; routed/shared expert output norms;
router entropy/top-k mass; expert geometry; and attention/router/dispatch/
expert/combine/layer latency.  Clean latency and observer-heavy trace results
must remain separate.

## Causal interventions

- A: full layer identity bypass (`h_out = h_in`) as a strong diagnostic.
- B: routed-MoE contribution bypass while preserving attention/shared path.
- C: previous-iteration routed-MoE output for the same logical row/layer.
- D: shared-only and routed-only decomposition.
- E: contiguous shallow, middle-low, middle-high and deep layer groups after
  individual-layer screening.

Interventions must first be applied one layer at a time, with matched pre-state
at the first affected phase wave.  Record top-1 flips, accepted-set changes,
confidence-margin changes, sampled logit KL/JS, future NFE/trajectory changes,
final sequence divergence, GSM8K exact-answer accuracy and HumanEval pass@1.

## Core maps and scores

Produce layer × normalized-refinement-phase maps for hidden relative-L2 update,
attention update, MoE update, output cosine, router entropy, component latency,
and causal decision sensitivity.  Keep `DecisionSensitivity` and
`PhysicalCost` separate, then report a cost-per-decision-impact or equivalent
efficiency score.  The target region is high latency cost and low causal
decision sensitivity.

## Oracles

- O1: optimistic independent single-layer oracle.
- O2: greedily validated multi-layer oracle.
- O3: validated contiguous layer-group / routed-MoE-only group oracle.
- O4: final-trajectory and benchmark-quality-safe oracle.
- Decision-removal oracles at layer, routed-MoE layer, layer-group and (only
  where evidence permits) token×layer / token×expert granularity.

Report perfect and realistically implementable oracles separately.  Convert
cost to request E2E using measured clean runtime shares; never call an
observer-heavy or stage-only result an E2E gain.

## Gates

- `<5%` request E2E: kill.
- `5–8%`: weak/characterization.
- `8–12%`: promising / HOLD.
- `12–20%`: strong.
- `>20%`: very strong.

A method prototype is forbidden below 8%.  Strong discovery additionally
requires a phase-shifting sensitivity map, at least 10% E2E cost in a
low-sensitivity region, a phase-layer oracle of at least 12%, or another clear
decision-contribution/physical-latency mismatch.

## Candidate generation if warranted

Candidate families may include refinement-phase layer refresh, routed-MoE-only
selective refresh, a decision-critical layer set, decision-critical routed
work, or periodic exact refresh.  The candidate must be training-free and be
distinguished from generic late-layer skipping, DICE-style image-diffusion
selective synchronization, Epoch live-row compaction, TEAM/REFLEX, generic
early exit and MoE expert skipping.

## Required outputs

- layer-phase stability, decision-sensitivity and latency maps
- decision sensitivity vs latency scatter
- MoE-only and, if feasible, shared-vs-routed contribution maps
- single-layer, greedy multi-layer, group and final-trajectory oracles
- candidate method table and prior-art audit
- complete experiment/GPU time logs and final decision

Final label must be exactly one of `NO-NOVEL-SIGNAL`,
`CHARACTERIZATION-SIGNAL`, `HOLD-CANDIDATE`, `STRONG-CANDIDATE`, or
`VERY-STRONG-CANDIDATE`.

The final report must explicitly answer all twelve questions in the user
prompt, including layer-update evolution, similarity-versus-necessity,
insensitive E2E share, routed/shared contributions, strongest oracle,
final-quality survival, DICE/Epoch distinction, and whether a paper-level
training-free method is justified.
