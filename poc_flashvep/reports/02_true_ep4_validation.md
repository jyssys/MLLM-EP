# Physical TRUE EP4 validation

Status: `PASS_GROUP_A_SMOKE`. Group B EP4 validation remains `NOT_RUN`.
Model revision: `dad945cac317da394b390f82c7b40691d8a881ed`.
Evidence: `../results/refinement_aware_ep_expert_allocation/ep4/A_sanity/`.

## Ownership proof

| EP rank | Physical GPU | Global experts per MoE layer | Local expert count | MoE layer count |
|---:|---:|---|---:|---:|
| 0 | 0 | 0–63 | 64 | 19 |
| 1 | 1 | 64–127 | 64 | 19 |
| 2 | 2 | 128–191 | 64 | 19 |
| 3 | 3 | 192–255 | 64 | 19 |

Every rank records its actual CUDA UUID.
Every rank inspects the loaded expert map and parameter tensors.
The audit asserts the exact contiguous ownership ranges.
The loaded expert tensors use BF16.
The process group has four ranks.
DeepEP normal mode performs dispatch and combine between those ranks.

The dense path uses TP4.
The MoE bridge partitions source token rows before EP dispatch.
It restores full logical output rows after combine.
It leaves shared expert computation intact.

## Fresh no-op checks

| Request | Native NFE | No-op token parity | No-op NFE parity | Trace token parity | Evidence class |
|---|---:|---|---|---|---|
| sanity:0 | 7 | PASS | PASS | PASS | MEASURED_GPU |
| sanity:1 | 8 | PASS | PASS | PASS | MEASURED_GPU |

These checks compare all retained expert edges with an identity policy hook.
They do not establish approximate-policy quality.
They do not establish full-benchmark HF/EP4 equivalence.

## Traffic evidence

The trace stores current global expert IDs before dispatch.
It stores received owner-local expert IDs and actual received buffer row counts.
Remote token assignments are nonzero.
This proves inter-rank expert work, not four independent replicas.

The logical activation payload is 4,096 bytes per BF16 hidden row.
DeepEP can deduplicate multiple experts on the same destination rank.
The audit counts unique token–destination pairs before estimating payload bytes.
Those bytes are `ESTIMATED_BYTES_FROM_MEASURED_ROUTING`.
They are not physical NVLink wire counters.

`ep_trace_audit.json` contains counts and timing observations per rank.
The same file contains structural pruning counterfactuals.
Those counterfactuals are not physically executed sparse policies.

Primary isolated request latency remains `NOT_MEASURED`.
Group B physical EP4 proof remains `NOT_RUN`.
The current HF jobs on Group B are independent Single-GPU jobs, not EP4.
