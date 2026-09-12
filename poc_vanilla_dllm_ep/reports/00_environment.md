# Environment and execution contract

## Scope

- Objective: independent, training-free latency reduction for **vanilla**
  MoE diffusion-language-model inference under expert parallelism.
- TEAM is not the primary method; it is an optional composition control only
  after a vanilla candidate passes the oracle/live gates.
- Single/EP2/EP4 use physical GPUs `0`, `0,1`, and `0,1,2,3` respectively.
- No process or command in this PoC uses physical GPUs 4–7.

## Hardware

- 8× NVIDIA H100 80GB HBM3 host; this task is restricted to GPU UUIDs:
  - GPU0: `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`
  - GPU1: `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`
  - GPU2: `GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e`
  - GPU3: `GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8`
- GPU0–3 are fully connected by NV18 links (`nvidia-smi topo -m`).

## Software and source pins

- TEAM official repository: `e9c502e5753ce79f660371e2fb4a8666f66cae75`
- SDAR repository: `4c2749ba103448f45520e8411533710a1e66574d`
- Model: local `SDAR-30B-A3B-Chat-b32-c351bbc`
- PyTorch: `2.8.0+cu128`; CUDA runtime: `12.8`
- Transformers: `4.52.4`
- vLLM: `0.10.2`
- FlashAttention: `2.8.3`
- Dtype: FP16 (the released checkpoint/runtime contract)
- Attention implementation: PyTorch SDPA, matching the validated positive-control
  harness.
- Local expert execution: vLLM `fused_experts`; distributed transport uses an exact, semantics-preserving NCCL `all_to_all_single` reference path.

## True EP contract

- 128 experts in every one of 48 sparse decoder layers.
- EP1: 128 resident experts on rank 0.
- EP2: 64 contiguous experts per rank.
- EP4: 32 contiguous experts per rank.
- Rank 0 computes the released router; token/expert branch rows are sent to the
  owning rank, evaluated only on resident expert weights, returned by reverse
  A2A, reduced in original token-major/top-k order, and broadcast so replicated
  dense state remains aligned.
- Layer validation against the released reference passed for EP2 and EP4:
  router-logit max error 0, output cosine at least 0.99999988, relative L2 at
  most 0.0545%.

## Stage-0 comparability corrections

Two apparent controls were rejected before the final run:

1. At threshold 0.95, tiny topology-dependent FP16 differences changed the
   adaptive acceptance trajectory and produced 415 versus 409 global forwards.
2. Threshold 1.0 made per-iteration transfer fixed, but the released harness
   still stopped at EOS. A later Single restart therefore executed fewer model
   forwards than EP2/EP4. Those runs are retained under
   `results/stage0_early_stop_unmatched_threshold1/` and are not fair scaling
   evidence.

The final Stage-0 control uses threshold 1.0 **and disables EOS early stop** so
every topology executes the same fixed output budget. Reverse-A2A contributions
are restored to released token-major/top-k order before reduction. EP1 also
bypasses vacuous world-size-one collectives; it uses the same fused expert
primitive without pretending that a self-A2A is real communication.
