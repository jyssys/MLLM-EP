# EP2 runtime audit

Status: **TRUE EP2 VERIFIED** on physical GPUs 6 and 7.

## Pinned substrate

| Component | Value |
|---|---|
| Parent commit | `6fdc42ecfe400b21b12463dbc663f70c11ef5341` |
| dInfer | `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` |
| vLLM | `0.10.2` |
| PyTorch / CUDA | `2.8.0+cu128` / `12.8` |
| Transformers | `4.55.2` |
| DeepEP | `1.2.1+73b6ea4`, source `73b6ea4a439ba03a695563f9fd242c8e4b02b37c` |
| Model | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, converted fused checkpoint |
| Dtype | BF16 |
| GPUs | physical 6/7, H100 80 GB; UUIDs `GPU-e3f3998e-...` and `GPU-4cc26b88-...` |
| Link | `NV18` between GPU6 and GPU7 |

The stock dInfer fused-weight loader indexed expert shards using the TP group.
The minimal substrate patch in `patches/dinfer_true_ep.patch` makes ownership
follow the EP group and removes the global TP-all-reduce monkeypatch. The
runner refuses any `CUDA_VISIBLE_DEVICES` value other than `6,7`.

## Ownership and topology

Both audited backends reported:

- TP1 / DP2 / EP2;
- 64 global experts and 32 local experts per rank;
- rank 0 owns experts 0--31; rank 1 owns experts 32--63;
- 402,653,184 local expert-weight bytes/rank;
- 12,582,912 bytes per BF16 expert.

The two ownership sets are disjoint and their union is exactly 0--63. The
scaling records conserve `global_M * top_k` routed assignments.

## Communication-path proof

### vLLM naive

Runtime manager: `NaiveAll2AllManager`; fused path:
`UnquantizedFusedMoEMethod`.

The rank-0 CUDA profile of one layer contained four
`ncclDevKernel_Broadcast_RING_LL` operations (hidden and router payloads from
both DP ranks), one BF16 NCCL all-reduce for combine, and the fused expert
kernel. This is real remote dispatch/combine, not TP-only sharded compute.

### DeepEP high throughput

Runtime manager: `DeepEPHTAll2AllManager`; modular path:

`FusedMoEModularKernel -> DeepEPHTPrepareAndFinalize -> TritonExperts`.

The profile directly contains:

- `deep_ep::intranode::notify_dispatch`;
- `deep_ep::intranode::dispatch`;
- `deep_ep::intranode::cached_notify_combine`;
- `deep_ep::intranode::combine`.

The pinned vLLM manager uses 20 communication SMs and constructs a 1 GiB
intra-node NVLink buffer. dInfer bypasses the vLLM GPU worker, so the runner
explicitly calls `prepare_communication_buffer_for_model` after model load.

Profiler-call timings contain first-use/observer effects and are not used as
benchmark numbers.

## Correctness boundary

- Naive and DeepEP HT produced bit-identical single-layer output for the audit
  input: max absolute difference 0 and relative L2 0.
- Every fixed-work shape had zero output-sum difference across the two backends.
- All six clean request runs produced the same generated token sequence.
- Rank-local results agreed exactly after EP combine.

A fresh one-GPU reference was not run after the user returned the GPUs to burn.
The preceding EP4 substrate audit found topology-dependent BF16 accumulation
drift, so this PoC makes no cross-topology quality claim. The present policy
does not change routes or arithmetic; its correctness claim is backend-to-
backend equivalence on the same EP2 topology.

Raw evidence: `results/ep2_20260911_170000/runtime_audit*`.
