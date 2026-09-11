# Backend regime comparison

## Availability and path validation

| Backend | Status | Evidence / blocker |
|---|---|---|
| vLLM naive | AVAILABLE | `NaiveAll2AllManager`; NCCL broadcast dispatch + BF16 all-reduce combine |
| allgather_reducescatter | UNSUPPORTED | Not an accepted backend in pinned vLLM 0.10.2 |
| PPLX | BLOCKED | `pplx_kernels` absent; pinned manager requires it and legacy direct methods are not implemented |
| DeepEP HT | AVAILABLE | `DeepEPHTAll2AllManager`; real intra-node notify/dispatch/combine kernels |
| DeepEP LL | BLOCKED | NVSHMEM/manager initialized, then pinned modular `forward_impl_chunked` failed its hidden-state dtype assertion; no unsafe bypass |

DeepEP HT was built in an isolated environment from commit
`73b6ea4a439ba03a695563f9fd242c8e4b02b37c` against PyTorch 2.8/CUDA 12.8.

## Fixed-work winner map

| Local M | Naive MoE (ms) | DeepEP HT MoE (ms) | Winner | Winner gap |
|---:|---:|---:|---|---:|
| 2 | 0.965 | 0.993 | Naive | 2.96% |
| 4 | 0.964 | 0.996 | Naive | 3.34% |
| 8 | 0.968 | 1.002 | Naive | 3.48% |
| 16 | 0.972 | 0.990 | Naive | 1.80% |
| 32 | 0.980 | 1.002 | Naive | 2.28% |
| 48 | 0.989 | 1.000 | Naive | 1.11% |
| 64 | 0.982 | 1.011 | Naive | 2.97% |
| 96 | 0.996 | 1.023 | Naive | 2.74% |
| 128 | 0.989 | 1.022 | Naive | 3.35% |
| 192 | 1.009 | 1.044 | Naive | 3.49% |
| 256 | 1.054 | 1.085 | Naive | 3.01% |

DeepEP HT executes the expert portion faster, but its modular prepare/dispatch
overhead cancels that advantage in this isolated first-layer replay. There is
no winner crossover and the gaps are far below the preferred 10% local gate.

## Clean full-request result

Three independent, randomized-order, warmup-5 engine restarts per backend:

| Backend | Restart request times (ms) | Median (ms) |
|---|---|---:|
| Naive | 526.393, 534.088, 776.625 | 534.088 |
| DeepEP HT | 501.984, 587.253, 501.086 | 501.984 |

DeepEP HT is 6.01% faster at the median. All six runs have NFE=19 and identical
generated tokens. This is useful static-backend engineering, not adaptive-
policy headroom.

The apparent first-layer replay/request inversion is not treated as a new
regime claim: the request spans all 16 layers and full dInfer state, while the
replay fixes one synthetic layer/route. Crucially, both diagnostics agree that
one backend dominates their entire tested size/state range.

## Switching feasibility

The pinned vLLM reads `VLLM_ALL2ALL_BACKEND` while constructing the CUDA
communicator. It also chooses a different fused-MoE implementation at model
construction: Naive uses the legacy fused method, while DeepEP uses a modular
prepare/finalize object, Triton expert implementation, persistent DeepEP
buffers, and different layouts. dInfer initializes the communication buffer
once after model loading.

Consequently, iteration-level switching is not a cheap existing call: it would
require co-resident paths or model/manager reconstruction, layout/workspace
management, synchronization, and CUDA-graph treatment. No switch was
implemented because the zero-cost oracle is only 0.174% MoE-local.
