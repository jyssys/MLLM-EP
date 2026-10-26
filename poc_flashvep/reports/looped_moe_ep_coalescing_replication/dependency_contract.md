# Causal execution and ownership

The native checkpoint determines every request's logical execution order.
Attention produces the current hidden state.
The unchanged router selects the current top-2 experts and scores.
Dispatch cannot precede that selection or the corresponding hidden state.
Expert execution requires the exact resident BF16 weights.
Combine must finish before the request's residual and downstream layers.

For one request, the next recurrent pass depends on the preceding pass.
The shared expert identity does not make those passes independent.
For different requests, current ready inputs can be independent.
Only that independent frontier can support exact cross-request coalescing or overlap.

`figures/dependency.dot` contains an editable dependency graph.
The graph does not release any future-pass hidden state early.
It does not infer readiness from loop indices alone.

## Ownership

Before replication, logical expert e belongs to rank e modulo P.
Every routed physical site uses this same stable rule.
IFM has 25 logical experts per site.
EP4 owner counts are 7/6/6/6.
EP8 owner counts are 4/3/3/3/3/3/3/3.
Those unequal counts do not create dummy logical experts.

Dense attention, router, normalization, and residual weights remain replicated.
The source row belongs to its row index modulo P.
Dispatch delivers the exact top-2 rows to expert owners.
Combine returns them to their original source ranks.
The original scores restore the token result before source restoration.
All-to-All order remains identical on every rank.

## Ready observer

The diagnostic runs one native thread and CUDA stream per independent request.
Each thread stops at an actual routed-expert boundary.
Its CUDA event defines local input readiness.
The coordinator broadcasts a fixed order of candidate work.
Every rank checks the relevant event.
An All-Reduce minimum creates an all-rank readiness quorum.
Only quorum-ready work enters the recorded eligible queue.

The coordinator does not merge work or predict future routes.
It uses one EP communicator in a globally consistent order.
The output event protects downstream use.
Explicit stream recording protects input and output allocation lifetimes.
The observer checks generated tokens against sequential B1 references.
Its control synchronization makes it a heavy diagnostic, not a clean serving engine.

## Native state limitation

All three official paths used here recompute the full token prefix.
They do not expose a cached-decode KV policy.
This experiment therefore preserves native full-prefix attention semantics.
It cannot claim paged-KV or cached-decode validation.
Huginn's batch-indexed initial state remains unchanged.
Independent B1 requests are not relabeled as a native B4 request batch.
