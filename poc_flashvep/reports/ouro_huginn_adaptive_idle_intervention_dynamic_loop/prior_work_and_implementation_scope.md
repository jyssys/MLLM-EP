# Prior work and implementation scope

Reviewed on 2026-11-08. These are source findings, not campaign performance results.

## FluidPD and FluidToken

[FluidPD, sections IV-B1 through IV-B3](https://arxiv.org/html/2610.06917v1) transfers a computed token prefix and executes the suffix at D.
Its PPI compares accumulated predicted prefix service with remaining TTFT deadlines.
Its DPI separates active Decode pressure from committed KV occupancy.
Its suffix budget uses predicted TPOT slack. A multiplier can distribute slack across several Decode iterations.
The policy allocates the advertised token budget to earlier queued requests.

This campaign uses the qualified RLT token connector and a shared empirical safety policy.
It does not execute the official SGLang system or reproduce FluidRole.
It does not claim to reproduce every FluidToken optimization step.
Use the name **FluidToken-style reproduction** for this baseline.
An online cutoff must use current state and past costs. It cannot use later arrivals.

## Load-aware Prefill Deflection

[Load-Aware Prefill Deflection, section 4.3](https://arxiv.org/html/2607.02043v1) evaluates Decode-side chunk schedules under per-step latency constraints.
Its predicted chunk cost depends on Decode batch size and KV context.
It places a new request at D and avoids that request's intermediate P-to-D transfer.
This campaign instead transfers an already computed P frontier.
It must retain the resulting transfer and source-buffer costs.
Deflection is not a substitute for the requested boundary negotiation.

## Continuous Depth Batching

[CDB](https://arxiv.org/html/2608.09444v2) forms batches between recurrent steps and manages depth-indexed state.
The reference LoopServe runtime uses a different engine and KV interface.
Its early-exit and aliasing behavior cannot be imported into fixed-R exactness tests without qualification.
The campaign reuses the current RLT frontier and recurrent execution interfaces.
It does not identify cross-depth batching itself as a new contribution.

## ExactLoop

[ExactLoop](https://arxiv.org/html/2611.05150v1) separates optimizations by their exactness requirements.
It also studies graph coverage, KV layout, and transfer-progress costs.
The campaign retains the qualified numerical path and native ACK protocol.
It does not apply the complete ExactLoop stack.
Any remaining gain after that stack is therefore NOT_MEASURED.

## Attribution limits

A generic logical-layer scheduler can reproduce a cut at the same recurrent boundary.
Such a result is EQUIVALENT_MECHANISM.
The experiment must distinguish useful load redistribution from a generic graph, polling, or submission improvement.
Kernel efficiency, causal correctness, and online SLO-goodput are separate findings.
No source review establishes a measured Loop advantage.
