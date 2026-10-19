# What an aliased unroll can and cannot establish

This note limits the interpretation of the campaign's exact-DAG controls.
It is not a proof that recurrence-specific distributed inference is impossible.

## Representation identity

An exact fixed-R4 Ouro execution can use four logical visits to the same physical layer stack.
A logical unroll can retain the same parameter storage, depth-indexed KV, loop norms, and operation order.
Changing only that representation need not expose a new mathematical dependency.

Therefore, an execution diagram that labels four GPUs with R1–R4 does not prove useful concurrency.
The actual input, cache version, and resource must exist at each operation.
The [exact frontier contract](exact_frontier_contract.md) states those requirements.

## Why this is not a universal novelty rejection

An aliased generic executor could reproduce many valid proposed schedules after someone implements the required state and ownership protocol.
That observation alone does not show that existing runtimes already supply the protocol efficiently.
An architecture can still have value if repeated physical operators change a measured placement or memory optimum.

Such a claim needs more than graph equivalence.
It needs the actual changed storage, communication, capacity, or critical-path cost.
It also needs a strong same-resource executor with the relevant alias-aware capabilities.
A comparison against a needlessly materialized logical model is insufficient.

## Current campaign evidence

The C14 depth/chunk construction maps directly to an ordinary causal pipeline with retained weight aliases.
No additional state or communication mechanism changes that particular mapping.
The mixed-depth ready-union construction has direct WaveFront/CDB and generic batching controls.
The current remote-MLP layout loses in the measured isolated-source microbenchmark.
Those conclusions have distinct scopes.
They do not combine into a theorem about every exact architecture.

The full-core physical-copy counterfactual tests another question.
It asks whether physical sharing changes the tested execution under matched allocated-copy capacity.
Its v1/v2 cache provenance is invalid or unvalidated.
The v4 path has distinct saved operands, but numerical and runtime-binding evidence remain separate.
No causal HBM-sharing conclusion follows yet.

## Conditions for a qualifying hybrid

1. State the ownership or execution feature that actually depends on repeated physical operators.
2. Identify a concrete cost that the feature removes or reduces.
3. Charge its extra state, movement, control, and synchronization.
4. Compare with an alias-aware conventional executor under the same capacity and workload.
5. Demonstrate incremental physical benefit or identify the still-unmeasured test.

The campaign can reject a tested construction without rejecting every future recurrence-specific hybrid.
It can also record useful generic engineering without renaming it as a new execution axis.
