# PoC Specification: Dispatch–Combine Asymmetry Discovery on GPUs 0,1,4,5

**Authoritative server path**  
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/dispatch_combine_asymmetry_discovery_spec.md`

**Project:** MLLM MoE EP Communication Discovery  
**PoC type:** Physical TRUE-EP4 discovery + architecture-matched LLM control  
**Allowed physical GPUs:** `0,1,4,5` only  
**Primary MLLM:** `Qwen/Qwen3-VL-30B-A3B-Instruct`  
**Primary LLM control:** `Qwen/Qwen3-30B-A3B`

---

## 0. Goal

Answer two questions:

1. Is the large Dispatch >> Combine asymmetry previously seen in Qwen3-VL MoE EP a generic MoE-EP property, or is there an additional visual-heavy MLLM effect?
2. What physical sub-component actually causes Dispatch to grow while Combine remains comparatively small?

Prior reference from a different GPU set:
- high visual Dispatch ~3.43 ms/layer
- Combine ~0.19 ms/layer
- ratio ~18x
- ~2,029 total tokens / ~1,980 visual tokens

**These are sanity anchors only. They were not measured on GPUs 0,1,4,5. Re-measure all primary baselines on 0,1,4,5.**

Do not implement an optimization method in this PoC.

---

## 1. GPU/topology contract

Use only:

`CUDA_VISIBLE_DEVICES=0,1,4,5`

Do not touch other GPUs.

Record:
- physical GPU IDs
- logical CUDA rank mapping
- GPU UUID/model/HBM
- `nvidia-smi topo -m`
- NVLink/NVSwitch/PCIe topology
- foreign GPU load
- git SHA/branch
- CUDA/PyTorch/vLLM/DeepEP versions
- model revisions

Primary topology should remain, if valid:

`TP2-DP2-EP4-PP1`

If CUDA remaps visible devices, document both:
- physical: `0,1,4,5`
- logical: `0,1,2,3`

Do not compare absolute timing against prior GPUs 2,3,6,7 without re-measurement.

---

## 2. Runtime freeze

For primary comparisons keep fixed:
- BF16
- 128 experts
- top-k 8
- EP4
- linear expert placement
- EPLB off
- redundant experts 0
- DBO off
- prefix caching off
- eager execution
- DeepEP high-throughput path for prefill
- identical timing boundaries
- identical route-accounting logic

Do not silently switch A2A backend, precision, placement, top-k, TP/DP/EP topology, or instrumentation between arms.

---

## 3. LLM control

Use standalone:

`Qwen/Qwen3-30B-A3B`

as the architecture-matched text-only control.

Verify loaded configs before running. Expected matched MoE geometry:
- 48 layers
- hidden size 2048
- 128 experts
- top-k 8
- MoE intermediate size 768
- BF16

The comparison is for communication behavior, not task quality.

---

## 4. Primary workloads

### Arm A — MLLM visual-heavy

Model:
`Qwen/Qwen3-VL-30B-A3B-Instruct`

Reuse frozen requests where possible:
- Low ~175 tokens
- Medium ~721
- High ~2,029 total / ~1,980 visual

High is the primary condition.

### Arm B — same MLLM, text-only token matched

Same Qwen3-VL model.

Use text-only prompts at approximately:
- 175
- 721
- 2,029 tokens

Reuse the prior ~2,033-token control if available.

### Arm C — standalone Qwen3 LLM, text matched

Model:
`Qwen/Qwen3-30B-A3B`

Use text prompts at approximately:
- 175
- 721
- 2,029 tokens

Keep identical EP placement/backend/dtype/timing boundaries.

---

## 5. Optional decode control

If cheap:
- B8 decode-only Qwen3-VL
- B8 decode-only Qwen3 LLM

Secondary only. Do not delay primary prefill discovery.

---

## 6. Baseline reproduction on the new GPU set

Before internal decomposition, cleanly re-measure the high visual MLLM arm on GPUs 0,1,4,5.

Measure:
- Dispatch ms/layer
- Expert ms/layer
- Combine ms/layer
- MoE total ms/layer
- Dispatch/Combine ratio

Do not require exact reproduction of old 3.43/0.19 values.

The new matched baseline on GPUs 0,1,4,5 is authoritative.

If the large asymmetry is not reproduced, stop after enough audit to explain why.

---

## 7. High-level timing

At every MoE layer record:
- Router completion
- Dispatch start/end
- Expert start/end
- Combine start/end

Primary metrics:
- Dispatch ms/layer
- Expert ms/layer
- Combine ms/layer
- MoE total ms/layer
- Dispatch/Combine ratio
- critical-rank stage latency

Use one defensible cumulative-critical-rank rule.

Do not average unrelated rank maxima or sum component maxima from different ranks.

Document the critical-rank rule.

---

## 8. Dispatch decomposition

Use this conceptual decomposition:

```text
Router complete
  -> A. control / count / routing coordination
  -> B. local packing / permutation / layout preparation
  -> C. cross-rank payload data movement
  -> D. receive completion / critical-rank tail / synchronization
  -> E. epilogue / expert-ready finalization
```

Audit the exact DeepEP path first.

Every component must be labeled:
- `DIRECTLY_TIMED`
- `KERNEL_CLASSIFIED`
- `INFERRED_RESIDUAL`

If not isolatable, report unresolved. Do not fabricate phase timing.

---

## 9. Combine decomposition

Conceptually:

```text
Expert output ready
  -> A. combine layout/input prep
  -> B. reverse cross-rank data movement
  -> C. reduction/scatter to source rows
  -> D. final synchronization / epilogue
```

Test, rather than assume, whether Combine is cheaper because Dispatch metadata/layout is reused.

---

## 10. Instrumentation lanes

### Lane 1 — clean timing
- minimal CUDA events
- no Nsight
- no per-layer host global synchronization
- >=10 paired repeats for final key comparisons

Use this lane for authoritative stage timing.

### Lane 2 — internal decomposition
Allowed:
- NVTX
- minimal DeepEP hooks
- CUDA events
- Nsight Systems for a few repeats
- kernel classification
- exposed device counters

Profiler-heavy runs are diagnostic only.

Measure instrumentation overhead when possible. Mark heavily perturbed strata `UNVALIDATED`.

---

## 11. DeepEP code audit targets

Identify where the following happen.

### Dispatch
- notification
- per-rank receive counts
- per-expert counts
- handle/layout construction
- local packing/permutation
- cross-rank data movement
- receive placement/unpack
- barriers/synchronization
- epilogue/finalization

### Combine
- Dispatch-handle reuse
- reverse movement
- reduction/scatter
- barriers
- finalization

Explain why each timer maps to the claimed phase.

---

## 12. Communication accounting

For every request/layer record:
- token rows
- total top-k routes
- unique destination ranks/token
- logical Dispatch hidden-row bytes
- logical Combine bytes
- per-source-rank send bytes
- per-destination-rank receive bytes
- max-rank bytes
- mean-rank bytes
- max/mean rank skew
- active experts
- expert histogram

Distinguish:
- more bytes
from
- more protocol/readiness/synchronization cost per byte

Do not infer wire metadata/padding unless directly measured.

---

## 13. Effective bandwidth

Only when a trustworthy data-movement-only interval exists:

`EffectiveBW = useful_logical_bytes / measured_data_movement_time`

Report separately for:
- Dispatch
- Combine
- Low/Medium/High
- MLLM visual
- MLLM text
- standalone LLM text

If only total stage time exists, use:

`ApparentThroughput = logical_bytes / total_stage_time`

Do not call apparent throughput true network bandwidth.

---

## 14. Token-count scaling

For each arm at approximately 175 / 721 / 2,029 tokens produce:

1. token count vs Dispatch ms/layer
2. token count vs Combine ms/layer
3. token count vs Dispatch/Combine ratio
4. token count vs Dispatch sub-components
5. logical bytes vs component latency

Use the plots descriptively; do not overfit three points.

---

## 15. Matched comparisons

### C1 — MLLM visual vs MLLM text
~2,029 visual-heavy vs ~2,029 text-only.

Question:
Does visual modality add Dispatch cost beyond token count under the same model/runtime?

### C2 — MLLM text vs standalone LLM text
Qwen3-VL text ~2,029 vs Qwen3 LLM text ~2,029.

Question:
Is Dispatch>>Combine already generic in an architecture-matched text-only MoE LLM?

### C3 — visual MLLM vs standalone LLM
Practical overall comparison only; less causally clean than C1.

---

## 16. Route-controlled TRUE-EP4 replay

If captured hidden tensors/routes exist, add:

`ROUTE_CONTROLLED_TRUE_EP_REPLAY`

At matched token count:
1. visual routes
2. MLLM-text routes
3. optionally balanced synthetic routes

Measure:
- Dispatch
- Combine
- per-rank bytes
- rank skew
- decomposition where possible

Interpretation:
- native visual/text gap persists in replay -> bare EP transport/routing candidate
- native gap disappears -> frontend/model/runtime readiness likely dominates

Do not confuse replay with native prefill.

---

## 17. Statistical protocol

Per primary arm/size:
- warmup >=3
- discovery repeats >=3
- final key comparisons >=10 paired repeats

If multiple requests are available:
- bootstrap at request/repeat level
- >=2,000 bootstrap replicates

Do not treat 48 layers as independent serving samples.

Report mean and median when tails are large.

---

## 18. Verdicts

### `GENERIC_EP_DISPATCH_COMBINE_ASYMMETRY`
MLLM visual, MLLM text, and standalone LLM text all show similar large Dispatch>Combine ratios.

### `MLLM_VISUAL_DISPATCH_EXCESS`
At matched tokens, visual MLLM Dispatch is materially larger than both text controls while Combine remains similar.

### `TOKEN_VOLUME_EXPLAINS_DISPATCH`
Visual/text/LLM approximately collapse onto the same Dispatch-vs-bytes curve.

### `DISPATCH_CRITICAL_RANK_TAIL_DOMINANT`
Receive readiness / rank tail / synchronization dominates.

Separate true network tail from upstream producer readiness if possible.

### `DISPATCH_DATA_MOVEMENT_DOMINANT`
Actual cross-rank payload movement dominates.

### `DISPATCH_LAYOUT_CONTROL_DOMINANT`
Count/layout/packing dominates.

### `ASYMMETRY_NOT_REPRODUCED`
Matched clean timing on GPUs 0,1,4,5 does not show meaningful Dispatch>>Combine asymmetry.

---

## 19. MLLM-specific promotion heuristic

Use `MLLM_VISUAL_DISPATCH_EXCESS_STRONG` only if most hold:
- visual Dispatch > matched MLLM text by >=20%
- visual Dispatch > standalone LLM text by >=20%
- Combine difference remains small
- paired CI excludes near-zero effect
- a specific Dispatch component explains a substantial part of the gap
- native and replay evidence form a coherent causal story

If visual-vs-matched-text <10% and LLM reproduces the same asymmetry:

`NO_MLLM_SPECIFIC_ASYMMETRY`

Do not force a method.

---

## 20. Required artifacts

```text
poc_dispatch_combine_asymmetry/
  artifacts/
    manifests/
    gpu_audit/
    native_mllm_visual/
    native_mllm_text/
    native_llm_text/
    route_controlled_replay/
    nsight/
    final_metrics.json
    final_verdict.json
  reports/
    environment.md
    instrumentation_audit.md
    native_stage_breakdown.md
    dispatch_internal_decomposition.md
    combine_internal_decomposition.md
    mllm_vs_llm_control.md
    route_controlled_replay.md
    final_summary.md
    reproduction.md
  figures/
  scripts/
  hooks/
  tests/
```

---

## 21. Required final summary block

```text
GPU_SET: 0,1,4,5
PHYSICAL_TO_LOGICAL_GPU_MAPPING:
GPU_MODEL:
GPU_TOPOLOGY:

MODEL_MLLM:
MODEL_LLM:
TOPOLOGY:
DEEPEP_BACKEND:
DEEPEP_VERSION:
VLLM_VERSION:

MLLM_VISUAL_HIGH_TOKENS:
MLLM_VISUAL_HIGH_DISPATCH_MS:
MLLM_VISUAL_HIGH_COMBINE_MS:
MLLM_VISUAL_HIGH_RATIO:

MLLM_TEXT_HIGH_TOKENS:
MLLM_TEXT_HIGH_DISPATCH_MS:
MLLM_TEXT_HIGH_COMBINE_MS:
MLLM_TEXT_HIGH_RATIO:

LLM_TEXT_HIGH_TOKENS:
LLM_TEXT_HIGH_DISPATCH_MS:
LLM_TEXT_HIGH_COMBINE_MS:
LLM_TEXT_HIGH_RATIO:

VISUAL_MINUS_MLLM_TEXT_DISPATCH_PCT:
VISUAL_MINUS_LLM_TEXT_DISPATCH_PCT:
MLLM_TEXT_MINUS_LLM_TEXT_DISPATCH_PCT:

DISPATCH_CONTROL_COUNT_MS:
DISPATCH_PACK_LAYOUT_MS:
DISPATCH_DATA_MOVEMENT_MS:
DISPATCH_RECEIVE_TAIL_SYNC_MS:
DISPATCH_EPILOGUE_MS:
DISPATCH_RESIDUAL_UNATTRIBUTED_MS:

COMBINE_PREP_MS:
COMBINE_DATA_MOVEMENT_MS:
COMBINE_REDUCTION_SCATTER_MS:
COMBINE_SYNC_EPILOGUE_MS:
COMBINE_RESIDUAL_UNATTRIBUTED_MS:

DISPATCH_APPARENT_THROUGHPUT_GBS:
DISPATCH_EFFECTIVE_BW_GBS:
COMBINE_EFFECTIVE_BW_GBS:

ROUTE_CONTROLLED_REPLAY_STATUS:
ROUTE_CONTROLLED_VISUAL_MINUS_TEXT_DISPATCH_MS:

INSTRUMENTATION_OVERHEAD_STATUS:
PRIMARY_DOMINANT_COMPONENT:
GENERIC_VS_MLLM_SPECIFIC:
OVERALL_VERDICT:
NEXT_ACTION:
BRANCH:
COMMIT:
```

---

## 22. Claim boundary

A positive MLLM-specific result may support:

> Visual-heavy MLLM prefill exhibits an additional Dispatch-specific EP cost beyond token-matched text and an architecture-matched MoE LLM, attributable to a measured physical component.

A generic result may support:

> Dispatch/Combine asymmetry is primarily a generic EP protocol/runtime phenomenon rather than an MLLM-specific effect.

Do not claim a new method, end-to-end speedup, production impact, or MLLM-specificity without corresponding evidence.

---

## 23. Priority order

1. Audit GPUs 0,1,4,5 and runtime.
2. Re-measure the high-visual Dispatch/Combine asymmetry on the new GPU set.
3. Run Qwen3-VL token-matched text controls.
4. Run standalone Qwen3 LLM controls.
5. Only then instrument internal Dispatch/Combine phases.
6. Run route-controlled replay if cheap.
7. Produce a verdict before designing any optimization method.

The PoC must answer only:

> Is Dispatch >> Combine generic or MLLM-specific?

and

> What physically causes the current Dispatch cost?
