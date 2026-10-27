# Prior work and implementation boundary

Status: `SOURCE_AUDIT_COMPLETE_IMPLEMENTATIONS_PENDING`.
Model and runtime identities appear in `00_runtime_audit.md`.
Source verification date: 2026-10-27.

## REFLEX

[REFLEX](https://arxiv.org/html/2608.01784v1) allocates expert counts using block roles and lagged confidence progress.
Its reported edge-count savings do not establish physical EP latency savings.
Its paper discusses this distinction explicitly.

The paper's default role budget is 8 with a role offset of 4.
Its active-block reallocation can add another 2 experts.
Those settings can require 12 or 14 experts for a token.
This campaign restricts allocation to subsets of the original top-8 selection.
A clipped top-8 subset implementation would therefore be a proxy, not the paper algorithm.

The inspected paper page did not identify a runnable author repository.
The targeted code search did not establish one.
This limited search does not prove that no implementation exists.
Faithful REFLEX execution remains `NOT_RUN`.
Any local P3 variant must use the label `REFLEX_INSPIRED_PROXY`.

## Epoch

[Epoch](https://arxiv.org/html/2609.09748v1) changes block-level expert support and fresh-token EP dispatch.
It also uses a versioned decoded-state cache with refresh rules.
This campaign does not inherit those correctness claims.
It cannot report an Epoch result from a conceptual proxy.

The inspected article did not establish a runnable author implementation.
Epoch complementarity remains `NOT_RUN_UNAVAILABLE_IMPLEMENTATION`.
This campaign must not treat original top-8 subset pruning as equivalent to Epoch.

## TEAM

[TEAM](https://arxiv.org/abs/2602.08404) uses temporal and spatial routing consistency.
Its [author repository](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM) provides an SDAR-based implementation.
The documented target is SDAR-30B-A3B, not this Mini checkpoint.
This campaign does not replace its required model with SDAR.
It also forbids unverified stale-route or stale-output reuse.
TEAM benchmark execution therefore remains `NOT_RUN`.

## Required new evidence

Reduced expert count alone is not an EP-specific contribution.
The joint method must beat refinement-only allocation at the same executed-edge budget.
The comparison must include real dispatch, expert work, overhead, and generation quality.
The main outcome must use isolated request E2E latency.

The official [LLaDA2.X repository](https://github.com/inclusionAI/LLaDA2.X) identifies the model family.
The [dInfer repository](https://github.com/inclusionAI/dInfer) provides the native execution basis.
The local patched source hashes remain necessary for exact reproduction.
