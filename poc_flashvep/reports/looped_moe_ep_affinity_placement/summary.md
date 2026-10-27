# Summary

IN_PROGRESS: this is a preview, not a completed PoC.

Verdict: **PENDING**.

The mandatory TRUE EP matrix contains 90/90 completed cells.
The study uses all three original checkpoints at EP4 and EP8.
The available native cloze and multiple-choice anchors reproduce the published level within the declared compatibility rule.
This compatibility rule is not an equivalence test.

| Model | EP | Prefill ms | TPOT ms | 32-token E2E ms | Output tokens/s | Static/generic E2E |
| --- | --- | --- | --- | --- | --- | --- |
| ifm_ouro | 4 | 311.692 | 317.686 | 10162.157 | 3.149 | 0.978× |
| ifm_ouro | 8 | 317.768 | 328.200 | 10497.931 | 3.048 | 1.007× |
| ifm_huginn | 4 | 367.431 | 322.281 | 10311.558 | 3.103 | 0.973× |
| ifm_huginn | 8 | 325.614 | 324.210 | 10368.953 | 3.086 | 0.982× |
| foil1 | 4 | 46.865 | 48.118 | 1540.144 | 20.777 | 0.979× |
| foil1 | 8 | 49.046 | 50.702 | 1625.627 | 19.683 | 0.970× |

| Model | Lag-1 cells | Conditional Recall@2 | Pass/domain popularity Recall@2 | Increment |
| --- | --- | --- | --- | --- |
| ifm_ouro | 5040 | 0.555 | 0.457 | 0.098 |
| ifm_huginn | 5040 | 0.550 | 0.424 | 0.126 |
| foil1 | 900 | 0.428 | 0.204 | 0.224 |

Held-out recurrent affinity does not establish an executable shortcut.
The bridge restores replicated dense states after every MoE call.
Its original payload is per expert edge, not per unique destination.
The generic source control reduces some traffic without consistent E2E improvement.
No validated generic-relative loop-aware latency oracle meets the prototype gate.

Whole-E2E percentage measurements remain pending. Read `e2e_breakdown.md` for the current status.
`accuracy_reproduction.md` contains quality counts, intervals, and unavailable tasks.
`report.md` describes scope, evidence, and the narrow conclusion.
