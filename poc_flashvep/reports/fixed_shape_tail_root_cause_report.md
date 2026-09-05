# Fixed-shape MoE online tail: root-cause investigation

Date: 2026-09-06
Branch: `flashvep/fixed-shape-tail-root-cause`
Result root: `poc_flashvep/deepep_revalidation/results/fixed_shape_tail_root_cause_20260906_001917/`

## Decision

**FINAL STATUS: ROOT_CAUSE_FOUND**

The repeatable giant tail is an online asynchronous-state phenomenon, not an intrinsic fixed routing shape. The strongest causal chain is:

`previous asynchronous DeepEP work` → `communication stream / peer-notification dependency remains outstanding` → `next dispatch(previous_event=...) waits` → `same-device dispatch CUDA span inflates` → `expert and combine execute at ordinary cost after release`.

A diagnostic `torch.cuda.synchronize()` immediately before MoE drains this outstanding work. On the clean decode M=1 anchor, events above 10 ms fall from 17/71,424 to 6/44,352 (43.2% rate reduction), and events above 20 ms fall from 10 to 0. This passes the pre-registered >=30% tail-frequency intervention gate. The sync is diagnostic only and is not a production method; large-prefill shapes are not uniformly improved.

## Environment and measurement trust

- `CUDA_VISIBLE_DEVICES=1,2,3,4` only; physical GPUs 1--4 were free before the runs. Only stale processes owned by this user on those four GPUs were terminated.
- Qwen3-VL-30B-A3B-Instruct local snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- vLLM 0.20.0 V1, BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput, Triton unquantized MoE, eager, DBO off, prefix cache off.
- Runtime logs: `Using DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`, EP world size 4, 32/128 linear expert placement, `Using TRITON Unquantized MoE backend`.
- New stage observer uses same-device CUDA events around layout, dispatch, modular expert, and combine; it never subtracts event timestamps between GPUs. Every row has local invocation, route id, layer, DP/EP rank, phase, M, and previous local state. Native V1 scheduler iteration id is not exposed by this hook, so `scheduler_iteration_id` is explicitly labeled `local_moe_invocation_proxy`.

## Anchor populations

| Anchor | n | p50 (ms) | p90 | p99 | max | CV |
|---|---:|---:|---:|---:|---:|---:|
| Decode M=1 (baseline) | 71,424 | 1.236 | 1.946 | 2.409 | 1,944.237 | 714.6% |
| Prefill M=284 (baseline) | 384 | 1.797 | 3.354 | 3.471 | 3.546 | 41.4% |
| Prefill M=137 (baseline) | 1,152 | 1.038 | 2.589 | 4.900 | 54.336 | 175.2% |
| Prefill M=536 (baseline) | 384 | 3.370 | 4.728 | 13.872 | 34.640 | 102.3% |
| Prefill M=2048 (baseline) | 192 | 2.392 | 2.837 | 78.089 | 89.118 | 274.8% |

M=284 is present in the natural baseline population, although the fixed-slot diagnostic run produced M=137/411 rather than exactly M=284 because the V1 scheduler coalesced identical prompts differently.

## Where the tail first appears

The representative fixed-shape events are dispatch-first:

| phase/M | layer | affected ranks | whole | layout | dispatch | expert | combine | first divergence |
|---|---:|---|---:|---:|---:|---:|---:|---|
| decode/1 | 5 | DP0 TP ranks 0,1 | 1,944.24 | 0.04 | 1,941.81 | 0.64 | 0.06 | dispatch |
| decode/1 | 28 | DP0 TP ranks 0,1 | 229.84 | 0.04 | 228.76 | 0.44 | 0.05 | dispatch |
| prefill/137 | 0 | DP1 TP ranks 2,3 | ~54.3 | 0.07--0.12 | 21.14--21.29 | 32.02--32.16 | ~0.07 | dispatch + expert |
| prefill/2048 | 1 | DP1 TP ranks 2,3 | 88--89 | 0.08--0.12 | 85.47--86.71 | ~0.03 | ~0.06 | dispatch |

In the decode anchor, expert and combine are normal in the same tail row. Cross-rank grouping is usually both TP ranks of one DP group, with a smaller number of all-rank collective events; this is consistent with a peer/collective dependency rather than a single expert's arithmetic cost.

## Previous-state effect

For decode M=1, tail rows have previous whole-MoE median 1.899 ms versus 1.242 ms for normal rows, and previous dispatch median 0.524 ms versus 0.207 ms. The current M and active-expert medians remain M=1 and 8 in both groups. This is a state association at fixed current shape, not a token-count explanation.

## Fixed-route replay

Two new persistent-worker controls separate intrinsic route cost from online state:

1. Exact hidden input/top-k route, layer 45/rank 0, M=2984, g=30, 20 warmups + 100 iterations. Expert median ~0.235 ms, max 0.382 ms, `routing_changed=false`.
2. Exact request with stock DeepEP path, 100 iterations × 4 ranks. Layer 45 aggregate dispatch p50/p99/max = 0.366/1.004/2.393 ms; expert = 0.481/0.580/0.745 ms; combine = 0.101/0.614/0.677 ms.

Neither replay produced 10--1,944 ms tails. The online giant-tail class is therefore not reproducible from the route/input snapshot alone.

## Diagnostic intervention

`FLASHVEP_SYNC_BEFORE_MOE=1` calls `torch.cuda.synchronize()` before stock `FusedMoE.apply`. For decode M=1:

- Baseline: 17/71,424 events >10 ms; 10 >20 ms; p99 2.409 ms; max 1,944.237 ms.
- Synchronized: 6/44,352 >10 ms; 0 >20 ms; p99 2.094 ms; max 14.739 ms.
- >10 ms frequency reduction: 43.2%; >20 ms class eliminated.

The intervention does not improve every large-prefill shape (for example M=536 and M=2048 still show tails). That shape dependence is expected from a drain-at-boundary diagnostic and does not weaken the dispatch-stream attribution for the giant decode tail.

## Nsight Systems escalation

Full serving capture succeeded with `--trace-fork-before-exec=true --wait=all --sample=none --cpuctxsw=none --trace=cuda,osrt`.

- `nsys/full_serving_resource_atlas.nsys-rep`: 112,885,849 bytes.
- `nsys/full_serving_resource_atlas.sqlite`: 353,030,144 bytes.
- Four child worker CUDA contexts (PIDs 3647414, 3647415, 3647418, 3647419) were captured.
- SQLite contains 424,716 kernel rows. Actual DeepEP `notify_dispatch`, `dispatch`, `cached_notify_combine`, `combine`, and `get_dispatch_layout` kernels are present, as are `fused_moe_kernel`, `topkGating`, CUTLASS FlashAttention, NCCL AllGather/AllReduce, and a `deep_ep::intranode::barrier` kernel (max 495.56 ms).
- Dominant summed kernel time: DeepEP dispatch/notify 3,424.97 ms; DeepEP combine/notify 2,453.81 ms; NCCL/TP collectives 3,884.49 ms; expert 335.37 ms; router 91.38 ms; layout 48.71 ms.
- NVTX table is empty. NVTX was deliberately disabled to avoid the previously observed NCCL/NVTX initialization crash; kernel names plus same-device stage events provide the attribution.

The profiled timeline shows long `notify_dispatch` kernels followed by ordinary expert/combine kernels. This is the first kernel-level divergence visible without a GUI.

## Non-MoE control

The corrected hook did not add a separate attention event stream in this bounded run. However, the tail rows are stage-localized to DeepEP dispatch while expert/combine are normal, and fixed-route expert replay is stable. The Nsight family control finds 600 FlashAttention kernels (8.48 ms summed, 0.019 ms maximum instance) but the giant-tail kernels are DeepEP notify/barrier rows. Thus the evidence rejects an attention/expert arithmetic explanation and strongly favors MoE/EP communication state. Details are in `analysis/non_moe_control.md`; explicit attention events remain a follow-up instrumentation improvement if a scheduler-level comparison is needed.

## Root-cause matrix and scope

`ROOT_CAUSE_MATRIX.md`, `INSTRUMENTATION_VALIDATION.md`, `TAIL_CASES.md`, `NSIGHT_FINDINGS.md`, `EXPERIMENT_LOG.md`, and `gate_summary.json` contain the detailed updates. The only code change is a read-only measurement wrapper plus an opt-in diagnostic synchronization branch. No routing, placement, scheduler, model math, or production optimization method was implemented.

### Causal evidence

1. Corrected same-device events and actual Nsight child-worker kernels show the tail is real and dispatch-first.
2. Exact route/input replay remains stable, rejecting intrinsic route/expert cost as the source of the giant tail.
3. Prior-state association exists at identical current M/active-expert shape.
4. Draining outstanding device work before MoE reduces extreme-tail frequency by 43.2% and removes the >20 ms class.
5. Local DeepEP source shows the exact `previous_event`/communication-stream and receiver `current_stream_wait()` dependencies that can expose this backlog.

## Limitation and next step

The native scheduler iteration id and a separate host-wait counter are not available in stock vLLM 0.20 hook points. The root cause is nevertheless localized to the DeepEP dispatch dependency with attribution, repeated reproduction, and a causal intervention. The next research step is a narrowly scoped DeepEP wait/queue instrumentation (e.g., event readiness and communication-stream occupancy) followed by a safe runtime policy study; no such policy is implemented here.
