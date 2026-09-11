# Experiment log

## 2026-09-11 setup and audit

- Created isolated branch/worktree from repository HEAD `f0dc8372`.
- Read the complete 721-line working specification.
- Pinned TEAM official repository at `e9c502e5753ce79f660371e2fb4a8666f66cae75`.
- Pinned SDAR repository at `4c2749ba103448f45520e8411533710a1e66574d`.
- Pinned checkpoint revision `c351bbc37d240aa6871f167e8f92d694281b0c22`.
- Checkpoint audit: 128 routed experts, top-8, 48 layers, 61,064,245,248 bytes.
- Began official checkpoint download and isolated environment setup while the
  task-owned GPU 6/7 utilization process remained active.

## Stage A — official positive control

- Staged separate baseline and TEAM model views from the same checkpoint.
- Loaded the released baseline and TEAM generation functions directly from the
  official OpenCompass wrappers; this avoided unrelated optional OpenCompass
  imports without rewriting the decoding algorithms.
- Clean restart speedups were 2.335x, 1.714x, and 1.832x (median 1.832x).
- The 128-token bounded subset scored GSM8K 3/4 for both.  HumanEval was 2/4
  versus 1/4, but the differing case ended mid-function.  At 256 tokens both
  baseline and TEAM passed that HumanEval case and the truncated GSM8K case.
- Structural tracing reproduced the paper's direction: NFE -41.7% and active
  expert events -50.8%.  Total token-expert assignments rose 11.5% because
  TEAM's four-way speculative exploration creates larger forwards.
- Observer tax was 36.7% for baseline and 12.2% for TEAM at the 32-token trace
  point, so trace time is not used as clean latency evidence.

## Stage B — semantics-preserving true EP2

- Added a reference TP1/DP1/EP2 executor only after Stage A passed.
- Rank 0 owns experts 0–63; rank 1 owns 64–127 in every layer.  Router and TEAM
  policy remain on rank 0; branch rows use NCCL `all_to_all_single`, local
  experts execute only on their owner, results return by a second A2A, and the
  combined tensor is broadcast for the replicated non-MoE path.
- Non-warm/cold-controlled restart speedups were 1.289x, 1.267x, and 1.551x;
  the warm paired point was 12.022 s baseline versus 7.751 s TEAM (1.551x).
- The EP2 executor is a correctness/reference backend, not DeepEP and not a
  claim of production performance.

## Stage C — residual profiling and adversarial controls

- Same-device CUDA events localized router, preparation, dispatch, local
  expert execution, combine, and whole MoE.  Module hooks measured attention,
  decoder layers, and LM head in observer-heavy runs.
- The Python expert loop made expert execution look dominant.  Replacing it
  with the already-available vLLM fused expert primitive reduced clean TEAM
  latency from 7.751 s to 3.338 s (-56.9%); this is a substrate/trivial-fix
  control, not a new candidate.
- On the fused control, MoE occupied 40.84% of clean request time and attention
  30.44%.  Individual MoE components were router 9.69%, prepare 9.43%, dispatch
  3.41%, expert 10.27%, and combine 6.49%.
- TEAM had 51.00% repeated logical position/expert routes across speculative
  branches, but exact expert-input duplicates were only 0.0295% (<=1% relative
  distance: 1.134%; <=5%: 3.672%).  Routing equality is therefore not a valid
  proxy for exact expert-output reuse.
- Candidate oracles were calculated only after the fused trivial-fix attack.
  The best independent perfect oracle, committing speculative branches before
  their full execution, was 6.28% request E2E.  All plausible feasible effects
  are smaller and no >=10% independent direction survived.

## Validation

- Seven artifact invariants pass: complete/disjoint ownership, true EP2 topology,
  nonzero remote work, dispatch/combine conservation, EP2 positive trend, and
  oracle arithmetic.
- Direct layer-0 equivalence against the original full-expert block passed:
  Python EP2 cosine 0.99999994 / rel-L2 0.0505%, fused EP2 cosine 1.0 /
  rel-L2 0.0553%, with exactly matching router logits.
- Final decision: `POSITIVE-CONTROL-ONLY`.
