# Analysis and final verdict

## Verdict: BLOCKED

The full TP4 serving comparison remains blocked by numerical qualification gate C2.
The experiment does not establish that Loop PP4 is faster or slower than optimized TP4.
It also does not establish a general architectural NO-GO.

We implemented and executed actual four-rank TP4.
It shards QKV heads and SwiGLU tensors and performs actual NCCL All-Reduce.
All 192 bounded component checks pass. Short full-model eager and graph probes also pass.
Longer full-depth hidden/logit values exceed the frozen limits, although observed token IDs match.
Native vLLM reproduces this distinction between token agreement and internal numerical equivalence.
We therefore preserve the failure and do not publish unqualified TP latency.

## What the valid PP control shows

At C4/P128/O32, the common four-process path narrows the Layer/Loop median ratio from 1.688× to 1.166×.
This supports a substantial runtime-implementation contribution to the old PP advantage.
It does not show that all remaining cost is architectural.
The three Layer-PP repeats vary widely. The residual advantage needs more samples and a qualified TP baseline.
Warm-up protocols and position/buffer preparation also differ between legacy and MP controls.
The ratio change does not isolate the effect of multiprocessing alone.

Both MP placements preserve bitwise owner KV and tested natural greedy outputs.
Both show different-request GPU kernel overlap on four physical GPUs.
Their transport uses direct peer copies, not hidden-state host staging.
Their process-local FIFO queues remove intermediate central-controller/GIL scheduling.
Local publication waits and per-stage graph/layout work remain.

| C | Runtime | Layer / Loop makespan | 95% paired bootstrap CI |
| --- | --- | --- | --- |
| 1 | legacy | 1.480× | [1.470,1.565]× |
| 4 | legacy | 1.688× | [1.575,1.860]× |
| 1 | mp | 1.252× | [1.066,1.276]× |
| 4 | mp | 1.166× | [1.159,1.956]× |

## Physical communication evidence

| Arm | Heavy window ms | Different-request kernel overlap ms | P2P copies | P2P MiB | P2P service sum ms |
| --- | --- | --- | --- | --- | --- |
| layer_pp4 | 590.028 | 143.852 | 508 | 31.640839 | 2.138 |
| loop_pp4 | 586.493 | 127.547 | 124 | 6.328339 | 1.507 |

The heavy PP traces have nearly equal per-rank kernel-service totals.
Their P2P service difference is much smaller than the historical wall-time difference.
Transfer count alone cannot explain serving latency.
The current heavy windows also include correctness observers and cannot replace clean clocks.

The bounded TP operator trace contains the following actual collectives:

| GPU | Actual operator All-Reduce kernels | NCCL kernel-service sum ms |
| --- | --- | --- |
| 4 | 32 | 12.146522 |
| 5 | 32 | 35.076385 |
| 6 | 32 | 15.187722 |
| 7 | 32 | 62.118359 |

These 32 reductions per rank come from eight component inputs, not one complete model forward.
The four ranks jointly produce 128 recorded All-Reduce kernels.
The kernel name identifies FP32 SUM, RING, and LL.
Python call logs retain each operation's actual tensor shape and logical operand bytes.
The nominal 192 reductions per full R4 core are visible in short graph capture logs.
They are not a measured serving collective profile.
The operator observer has substantial rank-readiness skew. Its NCCL service sums are not production link costs.

## Interpretation and decision scores

| Question | Result |
| --- | --- |
| True TP4 implementation and four-GPU participation | Demonstrated on operators and bounded model probes |
| Full-model numerical qualification | FAIL against frozen limits at P512 |
| Common multiprocess PP control | Implemented, bitwise bounded correctness, physical overlap |
| Complete 21-row three-way serving comparison | NOT_RUN |
| Optimized external TP4 serving baseline | NOT_RUN, native eager numerical diagnosis only |
| Loop PP advantage over TP4 | UNRESOLVED |
| Novel loop-specific execution mechanism | NOT_ESTABLISHED |
| Dynamic-depth follow-up | NOT_RUN, not justified by this result |

The smallest next action is a numerical-qualification study, not a scheduler implementation.
It should compare per-layer FP32 reference errors across the existing BF16 single-GPU and TP backends.
It must define any revised acceptance criterion prospectively on independent calibration inputs.
It must not retroactively convert this failed frozen gate into PASS.
If that study qualifies TP4, the preserved 21-row workload can support a fresh matched serving campaign.

## Limitations

The final verdict is a contract gate, not proof that correct TP inference is impossible.
Bounded token agreement is not benchmark accuracy or arbitrary-length generation parity.
The external engine differs in kernels, batching, cache layout, and software versions.
Only two short PP hardware shapes received three paired repeats.
No exclusive E2E attribution or quantitative observer correction is available.
We did not test a maximum-concurrency or HBM capacity boundary.
Excluded runs and failed attempts remain preserved and explicitly identified.

See [benchmark_summary.md](benchmark_summary.md), [correctness.md](correctness.md), and [transport_control.md](transport_control.md).
