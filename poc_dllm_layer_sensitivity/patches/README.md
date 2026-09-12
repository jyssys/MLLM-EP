# dInfer instrumentation patch

`0001-research-add-layer-decision-sensitivity-diagnostics.patch` is generated
from external dInfer commit `8c9561f5badf185b0ddcf38fc4753e3b2a49af88`, based on
`9132ce9b2580ac2e64bcaaf975b863a5005c7739`.

It adds environment-controlled layer/routed/shared/stale interventions,
layer-component traces, decision/logit captures, sequential policy execution
without reloading the 100B model, and ephemeral rendezvous-port selection.
The default `mode=none` execution preserves the validated baseline semantics.

Apply it to the recorded base with:

```bash
git am 0001-research-add-layer-decision-sensitivity-diagnostics.patch
```

The external worktree was intentionally not pushed to the upstream dInfer
repository; the self-contained patch is versioned with this PoC instead.
