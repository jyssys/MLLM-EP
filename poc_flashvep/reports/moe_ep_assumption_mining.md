# MoE-EP structural assumption mining

Date: 2026-09-07
Branch: `flashvep/moe-ep-assumption-mining`
Model: Qwen3-VL-30B-A3B-Instruct
Runtime: vLLM 0.20.0+cu129, DeepEP 1.2.1+73b6ea4, BF16, V1 eager
Topology: TP2 / DP2 / EP4 / PP1, DeepEP high-throughput, DBO off, prefix
cache off
GPU policy: `CUDA_VISIBLE_DEVICES=1,2,3,4`

## Final decision

**FINAL STATUS: `ASSUMPTION_SPACE_NO_GO`**

This is a conservative closure of the searched structural space, not a claim
that MoE serving has no future problems.  The search generated 50 distinct
execution assumptions.  Source reading and existing fresh GPU evidence were
used before authorizing a new run.  No candidate simultaneously met the
pre-registered analytical gate (>=20% direct request-level headroom), a
non-trivial counterfactual, and a credible novelty gap.  Therefore no fresh
GPU sweep was justified; the utilization process on physical GPUs 1--4 was
left running as requested, but it is not counted as research time.

## What was inspected

* Existing reports: autonomous discovery, fixed-shape tail root cause,
  wait-aware tail, online runtime discovery, speculative overlap, modality,
  coalescing and prior TP/EP studies.
* Existing live evidence: 21,456 successful requests and 59,789 logical steps
  in the final online campaign; 18,241.93 s valid live serving in the prior
  discovery; 89,664 valid logical invocations in the fixed-shape economic
  analysis.
* Local source: `deepep_ht.py`, `ubatching.py`, `dp_utils.py`,
  `modular_kernel.py`, `gpu_model_runner.py`, `all2all.py` from the installed
  vLLM 0.20 environment.  Three source passes are recorded in the companion
  files.
* Paper assumptions: DeepEP, vLLM EP, DA-MoE, TEMPO, Moebius/HAP, PROBE,
  Gimbal, ExpertPlex, ELDR/Semantic Parallelism, ScMoE, FarSkip, SpecMoE,
  Capacity-Aware-MoE/MACS/SERE and EPLB.

## Structural assumptions generated

`ASSUMPTION_CATALOG.csv` contains 50 distinct assumptions across dependency,
ownership, materialization, workload, scheduling, communication, hardware,
model and request clusters.  Each row records why the current system makes the
choice, the counterfactual work that would disappear, new overhead/dependency,
analytical headroom, prior-art risk and status.

The forced rethink checkpoints at 10/20/30/40 assumptions are in
`RETHINK_10.md`, `RETHINK_20.md`, `RETHINK_30.md` and `RETHINK_40.md`.  The
central failed assumptions were:

1. a large rank-local event is a large user-visible opportunity;
2. frequent materialization is automatically high-mass;
3. static routing geometry explains unexplained online latency; and
4. modality semantics can be relaxed without verification/critical-path cost.

## Analytical headroom gate

The full calculation is in `ANALYTICAL_HEADROOM.md` and
`HEADROOM_GATE.json`.  Key bounds are:

| Structural counterfactual | Best available bound | Decision |
|---|---:|---|
| scoped removal of outstanding DeepEP wait | 1.09% direct request tail excess | DROP |
| communication-SM/config envelope | 0.11% median, 2.10% max | DROP |
| device-side expert metadata lifecycle | 0.1137 ms typical/call; nominal <1% request | DROP |
| DP dummy/rendezvous/partition | -0.26%, -0.03%, -1.83% controls | DROP |
| phase/composition/turnover | <=2.5% request-level | DROP |
| fanout/incidence beyond load | +0.001% held-out error | DROP |
| vision/image contract after controls | 0.3--2% | DROP |
| partial/speculative execution | old oracle ~11.4%, verification unresolved | WEAK / DROP |

The previously reported 23.83% p25 fixed-tail projection is intentionally not
used as primary evidence: it was not a direct request join. The direct
request-level economic analysis caps that branch at 1.09%.

## Top counterfactuals

### Token-scoped combine release

DeepEP currently exposes one invocation-level `EventOverlap`; a counterfactual
would release completed token groups. It is structurally interesting, but the
known direct tail mass is 1.09%, and partial downstream verification is both
expensive and adjacent to SpecMoE/ScMoE/FarSkip. Status: `CLOSED BEFORE GPU`.

### Engine-owned cross-request slack

The worker owns stream switching and DeepEP handles while the engine provides a
step descriptor. Moving state-aware slack decisions to the engine could create
a new ownership contract. Existing common-state/turnover experiments move
both arms together and leave <=2.5% request-level effect. Status:
`UNKNOWN/DEFERRED`, not a positive finding.

### State-aware workspace/communication contract

DeepEP buffers, KV state and the CUDA allocator have separate lifecycles.
Common runtime regimes are observable, but no intervention-specific request
mass is isolated; typical metadata is ~0.11 ms and the config envelope is <=
2.10%. Status: `UNKNOWN/DEFERRED`.

Details and trivial-fix attacks are in `TOP_COUNTERFACTUALS.md` and the three
candidate notes.

## Why the search stopped before GPU

The contract says GPU is used only after the analytical and prior-art filters.
All scored candidates had headroom scores 1--2/5; no candidate reached the
>=20% promotion gate.  A fresh GPU run would either repeat already-closed
directions or measure an unobservable contract without a request-level join.
This is a deliberate resource decision, not an execution failure.

## What current systems already handle

DeepEP already overlaps communication with compute and uses explicit
previous-event/receiver dependencies.  vLLM already coordinates DP token
counts and execution mode per step, supports async scheduling, and provides
eager/DBO paths.  Existing literature handles load histograms, placement,
request locality, chunked prefill, skipping and speculative expert work.
The remaining choices are mostly ownership/lifecycle questions, but their
measured mass is either small or not yet observable.

## What could be a future new problem

The only defensible follow-up is a separate observability-first study: expose
nonblocking token/rank readiness and allocator/KV/clock state without adding a
per-layer synchronization tax, then measure a direct request-level oracle.
It would be promoted only if it finds repeated >=20% analytical mass and
survives simple synchronization, backend, buffering and batch controls.  This
branch does not claim that result.

## Required research judgement

* **Dense model:** not tested in this branch; no dense generalization claimed.
* **MoE without EP:** not tested; no claim that all effects require EP.
* **EP interaction:** present in the source dependency contract, but no
  high-mass causal intervention survived.
* **MLLM-specificity:** no surviving modality-causal candidate; image effects
  collapse to front-end/token-volume effects after controls.

## Artifacts

* Working contract: [moe_ep_assumption_mining_spec.md](./moe_ep_assumption_mining_spec.md)
* Negative map and catalog: [ASSUMPTION_MINING_NEGATIVE_MAP.md](../assumption_mining/ASSUMPTION_MINING_NEGATIVE_MAP.md), [ASSUMPTION_CATALOG.csv](../assumption_mining/ASSUMPTION_CATALOG.csv)
* Tree/frontier: [RESEARCH_TREE.md](../assumption_mining/RESEARCH_TREE.md), [ACTIVE_FRONTIER.md](../assumption_mining/ACTIVE_FRONTIER.md)
* Prior art: [PAPER_ASSUMPTION_MATRIX.md](../assumption_mining/PAPER_ASSUMPTION_MATRIX.md)
* Adversarial collision matrix: [PRIOR_ART_MATRIX.md](../assumption_mining/PRIOR_ART_MATRIX.md)
* Gate and decision: [ANALYTICAL_HEADROOM.md](../assumption_mining/ANALYTICAL_HEADROOM.md), [FINAL_DECISION.md](../assumption_mining/FINAL_DECISION.md)
