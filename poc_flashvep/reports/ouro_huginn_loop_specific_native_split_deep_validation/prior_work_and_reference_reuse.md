# Prior work and reference reuse audit

## FluidToken comparison scope

FluidToken transfers a completed prompt prefix and executes the suffix on the paired Decode worker.
Its PPI estimates queue deadline pressure. Its DPI includes active Decode load and KV admission pressure.
Its suffix budget uses predicted Decode iteration slack.
The paper implements the system on SGLang.
Our exact static connectors and bounded offline action search use vLLM-RLT.
They are a FluidToken-style reproduction, not the official system or its complete online policy.
The inspected paper and code search did not identify an author code repository.
This search result does not prove that no implementation exists.
Source: [FluidPD, Section IV-B and Algorithm 1](https://arxiv.org/html/2610.06917v1).

This study permits large Token prefixes and suffixes.
It does not force the Token baseline into C256.
It applies the same native graph policy, cache budget, state validity, and SLO contract to all axes.

## Reference LoopServing system

The reference README describes a GPU-resident elevator with device KV funding and release queues.
Its pooled allocation code uses positional completion credits, deep stock, generation ownership, and window-boundary preemption.
Its fixed-depth mode still has model-specific prompt and positional KV obligations.
Its cost model prices complete waves and includes controller, group, row, and context costs.
It warns that insufficient variation prevents separate cost identification.

These are useful design checks, not compatible binaries or drop-in data structures.
The reference uses vLLM, pooled KV, device controllers, and different execution interfaces.
The current qualified runtime uses vLLM-RLT, native plane-indexed KV, NIXL, and native ACK ownership.
This campaign does not copy early-exit rules, shared-depth approximations, or weights.
It reuses the qualified runtime and connectors.
The oracle must include context cost, complete boundary work, and admission credit.
It must not multiply recurrent rows by one constant and call that measured execution time.

Inspected reference files:

- `poc_flashvep/ref_loopserve/README.md`.
- `poc_flashvep/ref_loopserve/source/vllm/v1/core/ouro_pooled_accounting.py`.
- `poc_flashvep/ref_loopserve/source/vllm/v1/worker/ouro_runner/loop_cost.py`.

## Logical-layer equivalence

An unrolled logical graph can retain aliases to the original recurrent weights.
It does not need separate physical copies for every logical layer.
Ouro has 24 physical recurrent layers per loop. A Loop3 cut corresponds to 72 completed recurrent logical layers.
Huginn has two Prelude layers, four recurrent layers per loop, and two Coda layers.
A Loop31 cut corresponds to 2 + 31 × 4 = 126 completed logical layers before remaining core/Coda work.
The generic representation must also retain the same injection and boundary state.

The same logical boundary, aliased weights, valid KV, hidden state, and execution order form EQUIVALENT_MECHANISM.
The same argument applies to a valid token-by-logical-layer staircase.
This campaign does not count a renamed logical boundary as a new contribution.
A measured advantage over Token-only can justify a useful redistribution action.
It does not, by itself, prove an exclusive recurrent-only mechanism.
