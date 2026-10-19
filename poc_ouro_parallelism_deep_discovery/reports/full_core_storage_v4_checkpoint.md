# Full-core storage controls after complete cache isolation

Status: completed controlled-work measurements, with bounded provenance checks and unresolved numerical attribution.
This is not a candidate architecture result.
The v1/v2 attribution remains invalid or unvalidated.
Their raw records and later warnings remain unchanged.

## Completed v4 scope

The [queue](../artifacts/storage_fully_isolated_queue_v4.json) completed all eight jobs.
Both Base models ran on physical P2 and P4.
P2 uses physical GPUs4/6. P4 uses GPUs4/5/6/7.
Every arm uses BF16, fixed R4, native asynchronous scheduling, FlashAttention, and CUDA Graphs.

Both arms allocate four physically distinct, value-identical parameter cores.
Alias mode executes one core at all depths.
Copies mode executes a distinct core at each depth.
The setup retains the same declared attention objects, depth-KV slots, RoPE modules, and shared boundary weights.
The original tensor identity checks cover every copied block parameter.

Each job sets distinct vLLM, whole-model AOT/Inductor, Triton, and ordinary compilation-cache roots before import.
The CPU exports retain every rank's generated host graph and hash.
All eight jobs pass their saved-graph provenance checks.
Alias weight-argument counts are 194 and 386 for Base1.4B and Base2.6B.
Copies counts are 770 and 1538.

These counts establish distinct saved interfaces, not downstream kernel-pointer identity.
The [one-call probe](lifted_operand_binding_probe_contract.md) supplies a separate observed-binding check for Base1.4B P2.
Both observed arms use graph mode `NONE` for the 1024-token prefill call.
It does not inspect cached decode-kernel parameters.

## Controlled E2E observations

Each entry gives alias / copies median wall time in ms.
Each cell contains two frozen request shapes and five repeats per shape.
The request generates 128 teacher-forced tokens.
This is controlled-work live inference, not natural-task quality.
Arm order is sequential.
Two shapes do not support a broad request-level confidence claim.

| Model | TP | N1024, ms | N4096, ms | N8192, ms | Copies / alias |
|---|---:|---|---|---|---|
| Base1.4B | 2 | 924.323 / 924.817 | 1003.928 / 1006.402 | 1089.611 / 1153.157 | 1.0005× / 1.0025× / 1.0583× |
| Base1.4B | 4 | 785.122 / 741.289 | 828.887 / 783.087 | 929.368 / 876.148 | 0.9442× / 0.9447× / 0.9427× |
| Base2.6B | 2 | 1819.865 / 1815.633 | 1980.373 / 1982.383 | 2274.779 / 2270.749 | 0.9977× / 1.0010× / 0.9982× |
| Base2.6B | 4 | 1573.848 / 1609.328 | 1656.279 / 1713.031 | 1840.558 / 1845.124 | 1.0225× / 1.0343× / 1.0025× |

A ratio above 1.000× favors the alias job for that measured metric.
No ratio represents a new architecture speedup.
The raw trend is not consistent across sizes and TP degrees.
That observation alone does not isolate the physical-sharing effect.

## Numerical observations

All four model/TP comparisons differ numerically.
The separate captures cover four teacher positions, selected hidden states, and sampled KV.
Every comparison retains all four top-1 choices on each rank.
That agreement does not constitute a numerical or natural-quality PASS.

| Model | TP | Captured-logit relative L2 |
|---|---:|---:|
| Base1.4B | 2 | 0.007919 |
| Base1.4B | 4 | 0.006441 |
| Base2.6B | 2 | 0.013160 |
| Base2.6B | 4 | 0.009885 |

The [complete record](../artifacts/storage_fully_isolated_checkpoint3.json) retains unrounded per-rank values and every context.
Independent compilation and process state remain possible confounds.
This report does not identify an arithmetic, kernel-selection, or synchronization cause.
No post-hoc tolerance converts the differences into parity.

## Same-arm fresh-process control

A fresh Base1.4B P2 alias repeat uses the same settings and workload.
It also isolates all compilation caches.
The [repeat record](../artifacts/storage_same_alias_repeat_v1.json) retains the comparison.
Its saved graph passes alias provenance on both ranks.

| Context | First alias, ms | Fresh alias repeat, ms | First / repeat |
|---:|---:|---:|---:|
| 1024 | 924.323 | 928.330 | 0.9957× |
| 4096 | 1003.928 | 1006.802 | 0.9971× |
| 8192 | 1089.611 | 1092.513 | 0.9973× |

This same-arm comparison also has nonzero captured-logit relative L2: 0.00791896400461355 on both ranks.
All four top-1 choices match.
Numerical differences therefore occur without changing the requested alias/copies intervention.
This observation does not establish their unique cause.
It prevents attributing every numerical difference to physical sharing.

## Capacity and interpretation

Alias and copies have equal unique parameter-storage bytes and recorded peak allocated bytes at each v4 model/TP point.
The raw record retains allocator and buffer checks per rank.
These are capacity measurements.
No permitted HBM traffic counter exists in this campaign.
No result establishes an L2 hit rate or HBM-read saving.

The controls establish a more careful counterfactual test than v1/v2.
They do not establish a consistent large sharing-dependent latency benefit.
They also do not prove that sharing is harmful or useless.
Selective retention, another ready frontier, and capacity-constrained ownership remain separate questions.
Adaptive-mode storage timing and natural alias/copies quality remain `NOT_RUN`.
No new engine follows from this checkpoint.
