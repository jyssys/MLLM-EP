# Active frontier

Priority is information gain per GPU-hour, not expected positivity.

| Priority | IDs | Why now | Planned live block |
|---:|---|---|---|
| 1 | H01,H02,H30 | Establish a trustworthy observer before interpreting any trace | hook/no-hook and buffered/deferred controls |
| 2 | H03,H04,H05,H07,H20 | Source shows explicit cross-DP coordination; tests a distributed interaction rather than routing statistics | balanced/skewed/idle/phase-mismatch DP campaign |
| 3 | H06,H08,H10,H11,H28 | Continuous batching changes the collective sequence; direct request E2E is measurable | fixed request-multiset scheduling campaign |
| 4 | H09,H14,H16,H26,H27 | Detect shape/attention-to-MoE regime transitions with matched M/layer controls | chunk/context/layer campaign |
| 5 | H12,H13,H15,H23,H24 | Falsify temporal/MLLM and generic co-tail explanations | randomized history and modality campaign |
| 6 | H17,H18,H19,H21,H22,H25,H29 | Source-derived or deeper controls, promoted only when earlier blocks leave material mass | bounded diagnostics |
| 7 | H31,H32,H33,H34,H35 | Second source pass exposed scheduler-default, CPU-rendezvous, and per-layer metadata assumptions | restart-based or narrow monkey-patch controls |
| 8 | H37,H38,H39,H40,H41 | Rethink after five tests found wave/request metric divergence and a recurrent source-level metadata path | throughput control, metadata instrumentation, co-tail control |
| 9 | H42,H43,H44,H45,H46,H47,H48,H49 | Post-20-test frontier: common state transition, pinning, arrival/completion semantics and async-scheduler controls | fresh-worker replications and bounded runtime controls |

## Resumed status (complete)

- H36 completed on a fresh worker with direct dummy-RPC markers: measured
  negative at request level.
- H31 completed with async scheduling disabled: measured negative for the
  prior wave divergence.
- H32 completed with NCCL DP synchronization forced: measured negative for the
  matched request-shape contrast.
- H33/H38/H43/H45/H46/H48 completed in the final fresh-worker v7 queue. The
  frontier is now frozen: no candidate has a stable direct request-level
  multi-percent causal effect after the source-derived controls.
