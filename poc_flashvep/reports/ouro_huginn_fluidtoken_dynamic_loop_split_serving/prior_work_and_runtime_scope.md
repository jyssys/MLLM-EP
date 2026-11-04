# Prior work and runtime scope

## FluidToken source

FluidToken moves a prompt suffix after Prefix KV transfer.
PPI uses queued service predictions and remaining TTFT deadlines.
DPI separates active Decode load from occupied-or-reserved KV capacity.
The policy evaluates each Prefill scheduling iteration.
It allocates a finite suffix budget to earlier queued requests.
The stolen suffix remains on its own TTFT path.
TPOT slack limits the suffix budget.
Its slack multiplier can permit occasional delayed token iterations.
The paper uses SGLang, not this qualified vLLM-RLT runtime.
These details come from [FluidPD, Section IV-B and Algorithm 1](https://arxiv.org/html/2610.06917v1).

The current source search found no verified author repository for FluidPD.
This is a search limitation, not proof that no repository exists.
No FluidPD implementation was copied.

## Reproduction limits

The existing local adapter chooses discrete calibrated cuts at admission.
It rechecks Decode credit and latency slack at the actual migration frontier.
This differs from the paper's complete queue-window budget algorithm.
The local implementation must therefore use the label `FluidToken-style reproduction`.
It must not use the label `official FluidPD`.

The Token and Loop policies must use the same observed-state safety contract.
An offline cost table is a measured predictor, not a future-information oracle.
An isolated maximum is not a guaranteed concurrent-serving latency bound.
The final report must state any remaining algorithm differences.

## Other related mechanisms

DuetServe uses adaptive GPU multiplexing to address phase imbalance.
Its engine and scheduling contract differ from the local separate-phase connectors.
Its results are motivation, not measurements of the current runtime.
See the [DuetServe paper](https://arxiv.org/html/2511.04791v2).

Load-aware Prefill deflection routes Prefill work to Decode nodes.
It selects chunk schedules using latency predictions and token-latency constraints.
This provides a related generic alternative to recurrent-boundary migration.
See [Towards Load-Aware Prefill Deflection](https://arxiv.org/html/2607.02043).

## Loop and logical-layer equivalence

A logical-layer partition can select the same recurrent boundary.
With identical state, placement, and execution, the two mechanisms are equivalent.
Renaming the boundary does not establish Loop-specific novelty.
This campaign does not treat equivalent boundaries as independent mechanisms.

## Cache and scheduling contract

The inherited atomic runtime sets `enable_prefix_caching=False`.
The native Huginn path also rejects enabled prefix caching.
The new startup audit records the actual native flag and final reused-token counter.
All main runs must retain zero reused Prefix tokens.
Repeating 128 texts in N512 therefore does not imply Prefill reuse.
It also does not create 512 independent texts.

Main split runs retain separate-phase `refill` scheduling.
They do not combine Prefill and Decode rows in one shared-core GEMM.
The native Atomic PD control and full-snapshot control remain distinct transport controls.
Any transport-only improvement must remain separate from a Loop-Split improvement.
