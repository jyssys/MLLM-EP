# Execution ledger

All final evidence comes from successful, marker-audited runs in the validated
DeepEP environment. Raw logs and tensors remain local and are excluded from
Git; compact metrics and manifests are versioned.

| Phase | Successful run | Physical model executions | Evidence |
|---|---|---:|---|
| discovery rich trace | `discovery_trace_v1` | 96 conditions | 3,456 layer/step sites |
| discovery exact probes | `discovery_h1_probes_v5` | 352 | 32 requests, five policies, K6/K4 |
| confirmation rich trace | `confirmation_trace_v1` | 72 conditions | 2,592 layer/step sites |
| confirmation exact probes | `confirmation_primary_probes_v1` | 216 | 24 requests, frozen primary + controls |

The exact-probe smoke produced 20 logits and passed all 4 intervention marker
checks before the full run.

Setup failures retained locally:

1. `v1`: wrong system Python lacked `transformers`.
2. `v2`: the general environment lacked installed DeepEP kernels.
3. `v4`: the initial hook attempted rank-local renormalization. DeepEP presents
   only locally owned paths on a rank, so the code could see zero retained local
   mass. It was corrected to freeze the **global original retained weight mass**
   in the control record; the smoke marker audit then passed on the successful
   DeepEP environment.

No failed attempt contributes a metric to the final result.
