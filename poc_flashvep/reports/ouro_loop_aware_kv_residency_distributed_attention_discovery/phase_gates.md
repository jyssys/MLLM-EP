| Phase | Status | Evidence and limit |
| --- | --- | --- |
| P0 | PASS | Pinned native runtime, model, topology, 1/2/4/8 contention, and ownership checks. |
| P1 | CONDITIONAL | Long-context attention and KV capacity measured. Clean exclusive attribution and hardware counters remain qualified. |
| P2 | CONDITIONAL | Exact generic layout, allocator, tiering, and local sequence-parallel controls. No loop-specific advantage established. |
| P3 | PASS | Real two-rank exact BF16 sharding and NCCL merge. Operator only. |
| P4 | PASS | Real four/eight-rank operator scaling. Full-cache serving and TP comparison remain unimplemented. |
| P5 | NO_GO | No measured increment beyond the identical generic operator or allocator. |
| P6 | NOT_RUN | No qualifying loop-aware full-model candidate. No benchmark accuracy or sustainable SLO result. |

Gate G5 is `NO_GO_INCREMENTAL` for the current loop-specific claim.
Generic operator results remain valid and separate.
