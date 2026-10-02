# Dispatch: what is and is not separated

Installed source: vLLM prepare_finalize/deepep_ht.py + modular_kernel.py; DeepEP buffer.py + csrc/deep_ep.cpp + csrc/kernels/intranode.cu at73b6ea4.

| Requested boundary | Evidence | Limitation |
|---|---|---|
| A control/count/layout | get_dispatch_layout API DIRECTLY_TIMED (perturbed diagnostic); get_dispatch_layout and notify_dispatch KERNEL_CLASSIFIED | notify includes multi-rank barriers, count exchange and channel prefixes |
| B packing/permutation | KERNEL_CLASSIFIED intranode::dispatch | fused with C/D; no independent ms |
| C actual cross-rank movement | same fused dispatch kernel | pure movement UNRESOLVED; no true bandwidth |
| D receive/tail/readiness | notify_dispatch barriers + dispatch polling, KERNEL_CLASSIFIED; native first-layer excess INFERRED_RESIDUAL attribution supported by ready-barrier replay | network rendezvous vs upstream producer not directly split |
| E expert-ready epilogue | native receiver waits event, remaps IDs, constructs CPU-count metadata | composite prepare residual INFERRED_RESIDUAL, no fabricated exclusive duration |

Fresh count path has num_worst_tokens=0: notify_dispatch populates mapped host counters; C++ spins for received token/expert counts, then allocates receive buffers and launches payload kernel. This count/allocation handshake is absent from reverse combine's cached handle. Layout CPU/GPU launch gaps are part of native prepare, not HBM expert reads.

On the SAME clean cumulative-critical rank, high visual first-layer Dispatch 29.597ms versus text 1.199ms. It comprises 55.7% of visual's48-layer Dispatch sum. Remaining47 visual layers average 0.5031ms, text 0.5039ms.
Thus the visual increment is chiefly an upstream producer/rank-readiness encounter at the first MoE, not48 uniformly slower data transfers. All-producer-ready route replay removes that excess. Generic count/barrier/host-launch composite remains asymmetric. Nsight shows count/barrier kernels much longer than layout kernels and fused payload kernels on waiting ranks; classifier emits own-rank distributions and first/max times. Nsight starts/stops perturb readiness, so its exact first-layer duration is NOT asserted to reproduce the clean29ms.

Remaining47-layer mean request D/C ratios: visual 4.123, same-model text 3.805, standalone LLM 3.547. This is the scope of the generic-asymmetry secondary verdict: the complete48-layer visual ratio is NOT asserted identical to both controls.

Expert weight reads belong to expert execution, not Dispatch. No exclusive A+B+C+D+E numeric partition is claimed where kernels fuse these operations.
