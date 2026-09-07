# Prior-art attack matrix

| Candidate | Closest work a reviewer would cite | Exact collision? | Why retained or killed |
|---|---|---|---|
| token-scoped combine release | ScMoE, FarSkip-Collective, SpecMoE, speculative decoding | adjacent dependency relaxation | killed by 1.09% direct mass and verification cost |
| engine-owned cross-request slack | PROBE, Gimbal, ExpertPlex, ELDR, vLLM scheduler | adjacent serving ownership/admission | retained only as UNKNOWN until a direct state oracle exists |
| device-side metadata lifecycle | DA-MoE/RaMP and fused-MoE kernel engineering | not exact, but trivial | killed because 0.1137 ms typical and ordinary reuse solves it |
| scoped DeepEP notify/barrier | DeepEP, ASAP, asynchronous MoE overlap | close runtime mechanism | killed by small direct mass and unstable drain policy |
| phase-specific contract | Moebius, HAP, Layered/Chunked Prefill, ZeRO-Prefill | close workload/topology switching | killed by <=2.10% config envelope and known framing |
| workspace/clock/state observability | generic GPU runtime/allocator telemetry | no exact collision found | UNKNOWN; no request-level headroom measured |
| top-k/partial contribution | Capacity-Aware-MoE, MACS, SERE, SpecMoE | close routing/skipping | killed by quality and oracle gates |

The matrix is intentionally adversarial.  “No exact paper found” is not a
novelty claim: an UNKNOWN candidate must still demonstrate direct headroom and
causality before a literature search can become meaningful.
