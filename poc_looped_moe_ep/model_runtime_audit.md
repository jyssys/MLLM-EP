# Pre-integration model/runtime gate

The complete authoritative contract was read before implementation. Source
availability was checked on 2026-10-14, not inferred from a paper announcement.
See `results/model_candidates.json`, `prior_source_audit.json` and the exact
HF/checkpoint audit commands. No random-weight architecture is a candidate.

## Selected bounded mechanism probe

`ShourenWSR/how-to-loop-moe`, `Foil-1`, revision
`c1c857bab4665f00302f191fde5feb274ea97936`, Apache-2.0.
The released safetensors contain 553,730,048 BF16 elements and 9,437,184 FP32
RoPE-buffer elements (1,145,221,096 file bytes including metadata).
The learned model is approximately 0.554B parameters; model-card ~0.563B
also reflects the tensor/buffer accounting. This is much too small to justify
four H100s in a deployment. **No performance GO is possible from this model.**

One shared core pool (64 experts, top-2) is invoked in 16 native passes.
Each expert has three 1024×1536 matrices: 9 MiB/BF16 expert, 576 MiB/core pool.
Router and parameter-free RMS norms are shared; 16 attention modules are
untied. Independent head and tail pools each have 8 experts/top-2 and execute
once. They are included in whole-model work, excluded from core divergence.
Attention scale is the native muP 1/head_dim, not 1/sqrt(head_dim).
Weights will be checked by actual ownership and storage identity.

The actual official HF code is cache-free: every AR step recomputes the
whole prefix. Native H=16 is fixed. No KV engine will be invented, and no
full-prefix timing will be advertised as cached MoE decode performance.
Single-device reference and a four-rank TP1/replicated-dense, source-SP4,
EP4, DP1, PP1 mechanism test are the bounded feasibility attempt.

The validated Torch2.8 environment lacks public `F.grouped_mm`; the older
`torch._grouped_mm` exists with the same relevant argument signature. A local
API compatibility shim will be tested, without upgrading shared packages.
If it cannot preserve correctness, the physical probe stops. Expert sharding
will reuse the validated normal DeepEP dispatcher/owner-local-ID fix. Only
the expert subexpression may be bridged; native attention/router/loop/auxiliary
computation and BF16/FP32 buffer policy remain unchanged.

## Larger/alternative availability and bounded exclusion

- Loopie 6B/20B preview links cited by the paper return HTTP401 for anonymous
  model metadata; the linked `loopie/megatron` and `loopie/vllm` URLs return
  HTTP404. This is an access finding, not a universal absence claim.
- LOOM's pinned public repository provides training/evaluation code and
  describes local checkpoint paths; no downloadable published weight link
  was identified in that README. Do not treat its random initialization as
  pretrained evidence.
- Tying-the-Loop repository supplies code; a training script is not a released
  large checkpoint or validated EP serving runtime.
- **Larger authentic weights DO exist:** IFM MoE Middle/Huginn 500B-token
  checkpoints have ~8.78B resident parameters, BF16, native xLLM artifacts.
  Their official native inference is single-GPU, full-prefix recomputation.
  Inspected xLLM MoE does include A2A/model-parallel code, but the released
  configuration has **25 experts**, and its native ownership constructor
  requires exact divisibility by MP world size. Therefore native EP4 is
  rejected (25 % 4 != 0). Supporting unequal/padded ownership, strict native
  artifact loading and another runtime is not a one-line config change.
  No claim that these weights are unavailable is made; no second engine will
  be engineered to bypass this bounded feasibility constraint.
- The 0.2B ml-ryanlee fallback is real, but smaller than Foil and cannot repair
  the representativeness limitation.

This gate permits only a small-model mechanism characterization. Stage B/C
timing promotion requires additional evidence and must not be inferred from
structural balance or small-model fixed collective overhead.

## Observed gate outcome

The bounded bridge succeeded. Native reference repeat is exact; the tiny
four-rank probe has bitwise-identical logits on every rank. The first failed
probe is retained: NCCL rejected a noncontiguous publication buffer inherited
from `zeros_like`. A contiguous buffer fixes that actual seam, with regression
coverage. No attention or loop/cache execution engine was rewritten.

Subsequent 8-calibration/24-held-out runs preserve native router IDs/weights,
representative block outputs, last-token logits and short greedy continuation
under a genuine capacity-preserving noncontiguous expert-owner permutation.
The live expert weights are sharded (16 core +2 head +2 tail experts/rank),
not four independent replicated requests. Dense attention is replicated TP1;
all ranks jointly serve ONE B1 request through source-SP4 and physical EP4.
See `results/model_manifest.json`, `owner_weight_identity.json` and `validity.json`.
The result is a valid small-model mechanism test, not a representative EP4 GO.
