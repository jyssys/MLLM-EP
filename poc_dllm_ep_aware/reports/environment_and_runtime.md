# Environment and runtime audit

## Scope and evidence classes

All **new** GPU executions in this study used only physical GPUs 0–3.  GPU
UUID checks were enforced in both distributed runners.  GPU 0 was used for
single-GPU runs, 0–1 for EP2, and 0–3 for EP4.  The node reports `NV18`
between every pair of the eight H100s.  Previously collected single-GPU/EP2
positive-control artifacts were reused as explicitly permitted by the spec;
some of those legacy artifacts were originally collected on GPUs 6–7.  No new
command in this sprint used GPUs 4–7.

Three evidence classes are kept separate throughout:

1. **faithful clean request evidence**: official TEAM/SDAR single-GPU and the
   exact TEAM EP reference path;
2. **structural execution evidence**: observer-heavy stage traces and the
   paper-faithful REFLEX/DES ports;
3. **analytical bounds**: zero-cost or future-aware oracles, never reported as
   measured speedups.

## Pinned software

| Component | Version / revision |
|---|---|
| TEAM repository | `e9c502e5753ce79f660371e2fb4a8666f66cae75` |
| SDAR repository | `4c2749ba103448f45520e8411533710a1e66574d` |
| dInfer | `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` plus the prior true-EP patch |
| TEAM env | PyTorch 2.8.0+cu128, Transformers 4.52.4, vLLM 0.10.2, FlashAttention 2.8.3 |
| LLaDA env | PyTorch 2.8.0+cu128, Transformers 4.55.2, vLLM 0.10.2 |
| TEAM model | SDAR-30B-A3B-Chat-b32, 128 experts, top-8, 48 layers, FP16 |
| LLaDA model | LLaDA-MoE-7B-A1B-Instruct, 64 experts, top-8, 16 layers, BF16 load |

The staged SDAR baseline and TEAM modeling sources hash to
`2fc40b63...fe6333` and `cdc6a839...51e634`, respectively.  The dInfer fused
OLMoE source plus true-EP patch hashes to `d9b8ce70...698b1c`.

## Physical execution paths

### TEAM

The EP2/EP4 runner preserves the released decoder and all TEAM decisions.  Rank
0 evaluates the released router and TEAM masks; branch rows are sorted by the
owner of each global expert; NCCL `all_to_all_single` sends hidden, expert,
token, and weight rows; each rank holds a disjoint expert shard and uses vLLM
`fused_experts`; the weighted result returns through reverse A2A.  EP4 has 32
of 128 experts per rank.  Layer-0 validation against the unsharded reference
gave cosine 0.99999994 and relative L2 0.0560%.

This is true physical EP, but the dispatch is an exact reference transport, not
DeepEP.  Therefore clean TEAM results establish scaling on this substrate, not
production DeepEP throughput.

### LLaDA / dInfer

The local dInfer model was previously validated at 32/16 experts per rank for
EP2/EP4.  Stock vLLM 0.10.2's `NaiveAll2AllManager` is an important limitation:
`dispatch()` multicasts full hidden states and full router logits, and
`combine()` all-reduces the full hidden tensor.  It does **not** make REFLEX's
lower selected-pair count reduce transport.  For action-level diagnosis we
therefore used a separate reference assignment-A2A path with 32/16 resident
experts per EP2/EP4 rank and fused local experts.

That path physically realizes ragged top-k, but is not a production DeepEP
backend.  Dynamic-k DeepEP integration was not attempted after the quality and
oracle gates failed.

## Instrumentation and observer tax

CUDA stages use same-device CUDA events; no cross-GPU absolute timestamp is
subtracted.  Logical request latency uses wall time after device synchronization.
The prior TEAM single-GPU trace showed +36.7% baseline and +12.2% TEAM observer
tax.  TEAM EP4 detailed traces were much more intrusive, especially in Python
route preparation, and are used only for work counts and mechanism direction.
Headline EP4 latency is from three clean restarts.

## EP4 calibration surface

The bounded surface uses exact LLaDA expert dimensions H=2048/I=1024, BF16,
16 resident experts per rank, NCCL A2A, and the vLLM fused expert kernel.  It
covers 84 route shapes: 256/1024/4096 assignments, destination fanout 1–4,
remote fraction 0.25/0.5/0.75 where applicable, and 2/8/16 active experts per
active rank, with 5 warmups and 30 measured repetitions.  This is a calibration
of the reference substrate, not a claim about DeepEP.
