# Top counterfactuals and adversarial attacks

## 1. Token-scoped readiness instead of global combine

**CURRENT:** `DeepEPHTPrepareAndFinalize._receiver` waits on one
`EventOverlap` before expert inputs and `_finalize` waits before copying the
combined output.  **COUNTERFACTUAL:** release token groups whose destination
rank is ready while late ranks continue.  **WORK DISAPPEARS:** idle waiting on
the critical path, not expert computation.  **WORK MOVES:** readiness and
partial-output bookkeeping move into the next transformer block.  **NEW COST:**
attention must tolerate ragged/partial hidden state or recompute on rejection.
**HEADROOM:** exact route replay shows no tail; direct observed tail mass 1.09%.
**ATTACK:** ScMoE/FarSkip/partial-completion literature; no measured >=20%.
**DECISION:** UNKNOWN as a future semantic study, not a paper candidate now.

## 2. Engine-owned dependency scheduling

**CURRENT:** worker `execute_model` owns stream switching and DeepEP handle
lifetime; scheduler supplies a step descriptor.  **COUNTERFACTUAL:** engine
tracks communication debt and borrows independent request slack.  **WORK
DISAPPEARS:** unnecessary cross-request idle gap.  **WORK MOVES:** state
telemetry/queue management to engine.  **NEW COST:** fairness, IPC and KV
ownership. **HEADROOM:** common-state experiments moved both arms; request
effects <=2.5%. **ATTACK:** vLLM scheduler, PROBE/Gimbal, generic async serving.
**DECISION:** UNKNOWN/low evidence.

## 3. Device-side expert metadata lifecycle

**CURRENT:** Python list -> CPU tensor -> nonblocking GPU copy per receiver
(`modular_kernel.py:106-115`). **COUNTERFACTUAL:** persistent device counts or
GPU recomputation. **WORK DISAPPEARS:** repeated allocation/copy. **WORK
MOVES:** one kernel or persistent buffer. **NEW COST:** shape/lifetime hazards.
**HEADROOM:** 0.1137 ms median/call, nominal <5.5 ms over 48 layers; <1% of
request time. **ATTACK:** ordinary allocation reuse. **DECISION:** DROP.

## 4. Scoped DeepEP notify/barrier

**CURRENT:** full intranode notify/barrier completion gates dispatch/combine.
**COUNTERFACTUAL:** notify only dependent ranks/tokens. **WORK DISAPPEARS:**
outstanding wait. **WORK MOVES:** sparse dependency graph. **NEW COST:**
protocol correctness. **HEADROOM:** old giant tail is real, but direct request
removable mass 1.09%; wait-aware policy had no stable E2E gain. **ATTACK:**
DeepEP/ASAP generic overlap. **DECISION:** DROP for paper-sized direction.

## 5. Safe mid-step admission

**CURRENT:** DP token counts and execution mode are synchronized per scheduler
step. **COUNTERFACTUAL:** admit compatible work when a subset of ranks is
ready. **WORK DISAPPEARS:** scheduler bubble. **WORK MOVES:** admission and
KV bookkeeping. **NEW COST:** fairness and collective agreement. **HEADROOM:**
prior mixed/turnover request effects <=2.5%, wave-only effects invalid.
**ATTACK:** Layered/Chunked Prefill and generic continuous batching.
**DECISION:** DROP/LOW_HEADROOM.
