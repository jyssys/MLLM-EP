# Modality-Asymmetric Attention–MoE Wavefront PoC Specification

**Project:** MLLM MoE EP  
**Primary model:** Qwen3-VL-30B-A3B-Instruct  
**Primary phase:** Prefill  
**Status:** Oracle-first PoC  
**Constraint:** GPU-only. Do not implement a production wavefront scheduler or custom fused kernel in this PoC.

## 1. Executive Summary

This PoC evaluates whether multimodal prefill contains meaningful token-granular producer–consumer parallelism between Attention and MoE execution.

Current decoder execution is effectively:

```text
Attention(all tokens)
        ↓
Router
        ↓
Dispatch
        ↓
Expert execution
        ↓
Combine
```

This creates a full-query barrier: no token enters the MoE stage until attention for all query positions finishes.

However, once the layer input has produced the K/V tensors required by the query positions, attention outputs can conceptually become ready by query group. This suggests a hidden wavefront:

```text
Attention(head)
████
    Attention(tail)
    █████████████

    MoE(head)
    ███████

                 MoE(tail)
                 ███████
```

The prior obstacle was the **split tax** introduced by naive physical partitioning: tensor copies, extra launches, duplicated work, K/V rereads, synchronization, and fragmented kernels.

The goal of this PoC is therefore not to implement the final method. It must answer, in order:

1. Is the ideal zero-split-cost wavefront TTFT headroom large enough?
2. Does the best split correlate with MLLM-specific modality structure, especially the vision/text boundary?
3. How large is the naive split tax and what causes it?
4. After resource contention is modeled, does meaningful TTFT headroom remain?

Only if those gates pass should a later phase implement a zero-copy logical split or tile-ready Attention→MoE pipeline.

---

## 2. Core Research Question

> **Must all multimodal tokens finish attention before any of them enter the MoE?**

Long-term hypothesis:

> **MLLM prefill contains latent token-level producer–consumer parallelism between attention and MoE execution, but naive physical splitting destroys much of the benefit through fragmentation and data-movement overhead.**

A successful result should support the stronger MLLM-specific statement:

> **Vision-dominant multimodal prefill creates an asymmetric execution regime that can expose useful Attention→MoE wavefront parallelism.**

---

## 3. Hypotheses

### H1 — Attention→MoE serialization leaves meaningful ideal headroom

For a layer:

\[
T_{base}=A_{all}+E_{all}
\]

For split point `h`:

\[
T_{wave}^{ideal}(h)
=
A_H(h)+\max(A_T(h),E_H(h))+E_T(h)
\]

where:

- `A_H`: attention time for head query group,
- `A_T`: attention time for tail query group,
- `E_H`: MoE time for head group,
- `E_T`: MoE time for tail group.

The perfect split oracle is:

\[
T_{oracle}=\min_h T_{wave}^{ideal}(h)
\]

This assumes zero split overhead and perfect overlap.

### H2 — Multimodal structure predicts useful splits

The optimal split should be related to at least one of:

- vision/text boundary,
- vision-token fraction,
- modality-specific token population,
- modality-specific MoE load.

If the optimum is always a generic 50/50 split, the direction becomes generic chunked pipelining rather than MLLM-specific execution.

### H3 — Naive split tax is separable and potentially removable

If the ideal oracle is strong but the measured naive implementation is weak, the gap should be attributable to costs such as:

- tensor slicing/materialization,
- duplicate QKV work,
- extra attention launches,
- K/V reread,
- attention fragmentation,
- MoE fragmentation,
- extra routing/packing,
- stream/event synchronization.

### H4 — Useful overlap may be substage-specific

Attention and expert GEMM may compete for SM/Tensor-Core resources. Therefore the PoC must distinguish:

```text
Attention ↔ full MoE
Attention ↔ dispatch
Attention ↔ expert compute
Attention ↔ combine
```

rather than assuming full Attention↔Expert overlap is always favorable.

---

## 4. Scope

### In scope

- Qwen3-VL-30B-A3B-Instruct.
- Prefill.
- Real image+text prompts.
- Text-only control.
- GPU-only inference.
- Attention timing.
- Dispatch / expert / combine timing.
- Modality-layout extraction.
- Split-scaling benchmarks.
- Zero-cost oracle.
- Modality-boundary oracle.
- Contention measurement.
- Naive split benchmark.
- Split-tax decomposition.
- Amdahl-adjusted TTFT projection.

### Out of scope

Do not implement:

- production dynamic scheduler,
- custom FlashAttention kernel,
- custom fused Attention→MoE kernel,
- CPU offload,
- expert replication,
- expert weight movement,
- LLEP,
- EPLB,
- expert placement optimization,
- AGRS↔A2A switching,
- token pruning/merging,
- KV-cache optimization,
- speculative decoding,
- model retraining,
- routing/top-k modification.

---

## 5. Existing Project Context

Reuse existing MLLM-EP infrastructure where valid:

- Qwen3-VL inference,
- EP execution,
- routed-expert capture,
- prior MoE timing hooks,
- dispatch/expert/combine profiling,
- real multimodal workloads.

However, revalidate every reused timing boundary against the current runtime. Older instrumentation must not be assumed correct automatically.

---

## 6. Logical Split vs Physical Split

### Naive physical split

```python
vision_x = x[vision_mask]
text_x = x[text_mask]
y_v = attention(vision_x, ...)
y_t = attention(text_x, ...)
```

Potential costs:

- gather/indexing,
- new buffers,
- contiguous copies,
- multiple attention launches,
- duplicated QKV setup,
- repeated K/V reads,
- concatenation,
- lower occupancy.

### Desired future abstraction

```text
Full X
  ↓
Single QKV production
  ↓
Shared K/V
  ↓
Query-range descriptors only
```

A future implementation may use:

```text
(pointer, start, length)
```

or equivalent query-range metadata without physically moving tokens.

This future implementation is **not part of Phase-0**.

---

## 7. Definitions

### Head / Tail

For `M` query tokens and split `h`:

```text
Head = [0:h)
Tail = [h:M)
```

### Modality boundary

Extract the actual token layout from the model processor. Do not assume a simple `[vision][text]` sequence unless verified.

Record:

```text
vision spans
text spans
template/system spans
B_mod
```

For multi-span layouts, define both:

- the canonical vision→text transition,
- all modality-span boundaries.

### Split tax

\[
Tax(h)=T_{measured-naive}(h)-T_{ideal-wave}(h)
\]

---

## 8. Recommended Directory Structure

```text
poc_attention_moe_wavefront/
├── README.md
├── SPEC.md
├── wavefront/
│   ├── __init__.py
│   ├── schema.py
│   ├── oracle.py
│   ├── split_model.py
│   ├── modality.py
│   └── timing.py
├── scripts/
│   ├── inspect_runtime.py
│   ├── profile_baseline.py
│   ├── capture_modality_layout.py
│   ├── estimate_split_scaling.py
│   ├── run_wavefront_oracle.py
│   ├── bench_naive_split.py
│   ├── analyze_split_tax.py
│   └── summarize_results.py
├── tests/
├── configs/
├── results/
└── reports/
```

Do not overwrite prior `poc_flashvep/` results.

---

## 9. Stage 0 — Runtime and Timing Validation

Record:

```text
git branch / commit
vLLM version / commit
PyTorch
CUDA
NCCL
GPU model
GPU topology
CUDA_VISIBLE_DEVICES
model path
dtype
TP/DP/EP/PP
attention backend
MoE communication backend
max_model_len
max_num_batched_tokens
CUDA graph / eager mode
```

Save:

```bash
nvidia-smi
nvidia-smi topo -m
```

Validate the actual layer dependency graph.

Required timing boundaries:

```text
attention_total_ms
moe_dispatch_ms
moe_expert_ms
moe_combine_ms
moe_total_ms
layer_total_ms
```

If feasible also record:

```text
qkv_projection_ms
attention_core_ms
attention_output_projection_ms
```

Run clean vs instrumented baseline and report instrumentation overhead.

Target:

```text
median instrumentation overhead < 3%
```

If higher, use instrumented runs only for attribution and clean runs for final TTFT.

---

## 10. Stage A — Baseline Critical-Path Characterization

Use at least:

### W1 — Text-only control
### W2 — Standard single-image MLLM
### W3 — Vision-heavy/high-resolution single-image
### W4 — Long multimodal prefill

Optionally include multi-image workloads.

Record per request:

```text
num_total_tokens
num_vision_tokens
num_text_tokens
vision_fraction
TTFT
scheduled_to_first_token
total_prefill_ms
```

Record per layer:

```text
layer_id
num_tokens
attention_total_ms
dispatch_ms
expert_ms
combine_ms
moe_total_ms
layer_total_ms
```

Compute the affected TTFT fraction:

\[
P_r=
rac{\sum_l(A_{r,l}+E_{r,l})}{TTFT_r}
\]

Report:

```text
Attention fraction of TTFT
MoE fraction of TTFT
Attention+MoE fraction of TTFT
```

Suggested pre-gate:

```text
Attention+MoE should preferably account for >=30% of TTFT.
```

If far below this, substantial request-level gain is unlikely.

---

## 11. Stage B — Split Candidate Generation

Use normalized candidates:

```text
0.05
0.10
0.15
0.20
0.25
0.33
0.50
0.67
0.75
0.80
0.90
```

Additionally include:

```text
B_mod - 256
B_mod - 128
B_mod
B_mod + 128
B_mod + 256
```

Clamp to valid indices and deduplicate.

If the actual multimodal layout contains multiple meaningful modality transitions, include those boundaries too.

---

## 12. Empirical Split-Latency Models

Do not assume latency scales linearly with token count.

### Attention

Build empirical:

\[
A(q)
\]

for query-group size `q`, while keeping the intended full K/V context semantics where possible.

Attention latency may vary nonlinearly because of:

- kernel launch overhead,
- query tile count,
- FlashAttention shape,
- KV length,
- causal mask,
- occupancy,
- memory traffic.

### MoE

Build empirical:

\[
E(q)
\]

or preferably:

```text
Dispatch(q)
Expert(q)
Combine(q)
```

using representative routing distributions.

Preferred oracle inputs are measured isolated latencies. Interpolation from measured points is acceptable. Pure token-proportional scaling is only allowed if validated.

---

## 13. Stage C — Zero-Split-Cost Oracle

For every request/layer/split:

\[
T_{base}=A_{all}+E_{all}
\]

\[
T_{wave}^{ideal}(h)=A_H+\max(A_T,E_H)+E_T
\]

Compute:

### O1 — Per-layer perfect split

\[
h^*_{r,l}=rg\min_hT_{wave}^{ideal}(r,l,h)
\]

Maximum headroom.

### O2 — Per-request fixed split

One split fraction for all layers in a request.

### O3 — Modality-boundary split

Force:

```text
h = B_mod
```

### O4 — Global static split

One split fraction for all requests/layers.

This hierarchy is important. If O2/O3 recover nearly all of O1, a future practical method can be much simpler.

---

## 14. Amdahl-Adjusted TTFT Oracle

For request `r`:

\[
TTFT_{oracle}
=
TTFT_{base}
-
T_{affected,base}
+
T_{affected,oracle}
\]

with:

\[
T_{affected,base}
=
\sum_l(A_{all}+E_{all})
\]

Report:

```text
affected-region oracle improvement
layer-local oracle improvement
Amdahl-adjusted TTFT improvement
```

The **Amdahl-adjusted TTFT improvement is the primary GO/HOLD/NO-GO metric**.

Never make the decision from maximum-only values. Use median and distribution across target multimodal requests.

---

## 15. Stage D — Modality-Boundary Relevance

For each layer, record:

```text
M
B_mod
h*
h*/M
B_mod/M
abs(h* - B_mod)
normalized_distance = abs(h* - B_mod)/M
```

Report:

```text
median normalized distance
p90 normalized distance
fraction within ±128 tokens
fraction within ±256 tokens
```

Also test correlation between:

```text
vision_fraction
optimal_split_fraction
ideal_oracle_gain
```

Interpretation:

### Strong MLLM-specific result

- `h*` often near modality boundary, or
- optimal split strongly changes with vision fraction, or
- multimodal workloads show different optimal behavior than text-only control.

### Weak MLLM-specific result

- optimal split stays near a generic fraction regardless of modality.

Do not force a modality-aware claim if unsupported.

---

## 16. Gate Before Any Naive Split Implementation

Use median request-level Amdahl-adjusted ideal TTFT oracle:

```text
< 8%    → NO-GO
8–15%   → HOLD
15–20%  → GO for split-tax analysis
> 20%   → Strong GO
```

If `<8%`, stop immediately. Do not implement naive overlap.

This gate is intentionally strict because later contention and split overhead can only reduce the realized gain.

---

## 17. Stage E — Minimal Naive Split Benchmark

Only execute if Stage C/D passes the gate.

Variants:

### N0 — Unsplit baseline

```text
Attention(all)
MoE(all)
```

### N1 — Physical sequential split

```text
Attention(head)
Attention(tail)
MoE(head)
MoE(tail)
```

No overlap.

Purpose: quantify fragmentation/materialization cost.

### N2 — Naive concurrent split

```text
Attention(head)
then:
    Attention(tail) || MoE(head)
then:
    MoE(tail)
```

Use CUDA streams/events only if this can be added minimally and safely.

Do not modify custom kernels in this stage.

---

## 18. Split-Tax Decomposition

Try to attribute the gap to:

```text
tensor slicing/materialization
extra attention launch
duplicate QKV work
K/V reread
attention fragmentation
MoE fragmentation
routing/packing
stream/event synchronization
resource contention
```

Use incremental variants when feasible.

The goal is to identify the dominant 1–3 costs, not to perfectly account for every microsecond.

Promising outcome:

```text
large ideal oracle
+
poor naive result
+
tax dominated by clearly avoidable copies / duplicated work / launch overhead
```

Unpromising outcome:

```text
tax dominated by fundamental compute-resource contention
```

---

## 19. Resource Contention Measurement

For representative layer/token sizes measure:

```text
A_tail standalone
E_head standalone
A_tail + E_head sequential
A_tail || E_head concurrent
```

Define overlap efficiency:

\[
\eta=
rac{(A_T+E_H)-T_{concurrent}}{\min(A_T,E_H)}
\]

Interpretation:

```text
η = 1.0  perfect hiding
η >= 0.7 strong
η >= 0.5 useful
η < 0.3 weak
η < 0   destructive contention
```

If Attention↔Expert contention is weak, separately test:

```text
Attention ↔ Dispatch
Attention ↔ Combine
```

before declaring final NO-GO.

---

## 20. Contention-Corrected Oracle

The ideal oracle is optimistic.

Use measured overlap efficiency:

\[
Saving_{overlap}
=
\eta\min(A_T,E_H)
\]

Then:

\[
T_{wave}^{corr}
=
A_H+A_T+E_H+E_T
-
\eta\min(A_T,E_H)
\]

Use size-bucket-specific `η` if enough samples exist.

Report both:

```text
zero-contention ideal oracle
contention-corrected oracle
```

The corrected TTFT oracle should drive the final decision.

---

## 21. Logging Schemas

### Baseline layer trace

```text
run_id
request_id
workload_id
layer_id
phase
num_total_tokens
num_vision_tokens
num_text_tokens
vision_fraction
attention_total_ms
qkv_ms
attention_core_ms
attn_out_proj_ms
dispatch_ms
expert_ms
combine_ms
moe_total_ms
layer_total_ms
ttft_ms
scheduled_to_first_token_ms
```

Unavailable optional fields may be null.

### Split oracle trace

```text
request_id
layer_id
split_token
split_fraction
is_modality_boundary
distance_to_modality_boundary
A_head_ms
A_tail_ms
E_head_ms
E_tail_ms
ideal_wave_ms
corrected_wave_ms
base_affected_ms
ideal_improvement_pct
corrected_improvement_pct
```

### Naive split trace

```text
request_id
layer_id
split_token
variant
attention_head_ms
attention_tail_ms
moe_head_ms
moe_tail_ms
concurrent_region_ms
total_ms
split_tax_ms
overlap_efficiency
```

---

## 22. Required Figures

Produce at least:

1. Baseline TTFT breakdown: Attention / Dispatch / Expert / Combine / Other.
2. Baseline vs O1/O2/O3/O4 TTFT oracle.
3. Optimal split fraction vs modality-boundary fraction.
4. Layer-wise optimal split.
5. Oracle gain by layer.
6. Split-tax comparison: ideal vs N1 vs N2.
7. Overlap efficiency vs token/split bucket.
8. Ideal vs contention-corrected TTFT oracle.

---

## 23. Unit Tests

Implement tests for:

### Oracle formula

Example:

```text
A_H=2
A_T=8
E_H=5
E_T=3
```

Expected:

```text
T_wave=2+max(8,5)+3=13
```

### Split candidates

Verify:

- valid range,
- no duplicates,
- modality boundary included,
- offsets clamped.

### Modality boundary extraction

Cover:

- text-only,
- single vision span,
- template tokens surrounding image,
- multiple spans if supported.

### Trace alignment

Detect:

- missing layers,
- mismatched token counts,
- duplicate rows,
- invalid request IDs.

---

## 24. Decision Gates

### G0 — Runtime validity

PASS only if timing boundaries and modality layout are verified.

### G1 — Affected-region relevance

Prefer:

```text
Attention+MoE >= 30% of TTFT
```

### G2 — Ideal Amdahl-adjusted TTFT oracle

```text
< 8%    NO-GO
8–15%   HOLD
15–20%  GO
> 20%   Strong GO
```

### G3 — Modality relevance

Positive if any of:

- optimum clusters near modality boundary,
- optimum correlates with vision fraction,
- multimodal behavior differs materially from text-only.

### G4 — Contention

Desired:

```text
median η >= 0.5
```

Strong:

```text
η >= 0.7
```

Weak:

```text
η < 0.3
```

### G5 — Split-tax removability

Proceed only if the remaining tax is dominated by plausibly removable mechanisms rather than fundamental contention.

---

## 25. Final Decision

### GO

Proceed to Phase-1 logical-split / zero-copy implementation only if:

1. median ideal Amdahl-adjusted TTFT oracle is at least ~15%,
2. contention-corrected oracle remains meaningful,
3. naive failure is explained mainly by removable split tax,
4. multimodal structure provides useful scheduling information.

### HOLD

Use HOLD if:

- oracle is 8–15%, or
- ideal is strong but contention is not characterized, or
- split tax remains unexplained.

Only low-cost follow-up experiments are allowed.

### NO-GO

Stop if:

- ideal TTFT oracle <8%,
- corrected headroom collapses below ~5%,
- Attention+MoE is too small a TTFT fraction,
- resource contention dominates,
- modality structure adds no useful distinction and generic chunking is also weak.

Do not implement custom kernels after NO-GO.

---

## 26. Strong Positive Result Example

```text
Attention+MoE = 70% of TTFT

Per-layer ideal oracle:
TTFT -24%

Per-request fixed split:
TTFT -20%

Modality-boundary split:
TTFT -17%

Overlap efficiency:
η = 0.65

Contention-corrected oracle:
TTFT -14~17%

Naive implementation:
TTFT -2%

Tax:
mostly duplicate attention launch + K/V reread + tensor materialization
```

Interpretation:

> useful parallelism exists, but current tensor/kernel boundaries prevent exposing it efficiently.

This justifies a zero-copy method.

---

## 27. Negative Result Examples

### Ideal oracle tiny

```text
ideal TTFT oracle = 4%
```

→ NO-GO immediately.

### Ideal large, corrected tiny

```text
ideal = 22%
corrected = 3%
```

→ fundamental contention; test communication overlap once, otherwise NO-GO.

### Boundary irrelevant

```text
optimal split ≈ 50% for all workloads
```

→ possibly generic chunking, weak MLLM-specific story.

### Tax remains after no-copy execution

→ physical split was not the root cause; likely fundamental kernel/resource problem.

---

## 28. Required Execution Order

1. Inspect current HEAD and existing timing infrastructure.
2. Create `poc_attention_moe_wavefront/`.
3. Validate runtime and timing boundaries.
4. Profile baseline workloads.
5. Measure Attention+MoE TTFT fraction.
6. Measure isolated Attention/MoE scaling versus group size.
7. Compute O1/O2/O3/O4 zero-cost oracle.
8. Apply the `<8% / 8–15% / >=15%` gate.
9. Analyze optimal split vs modality boundary.
10. Only if promising, implement minimal N1/N2 naive split.
11. Measure overlap efficiency.
12. Test Attention↔Dispatch/Combine if full Attention↔Expert contention is poor.
13. Compute contention-corrected oracle.
14. Decompose split tax.
15. Issue final GO/HOLD/NO-GO.
16. Only after GO, design Phase-1 zero-copy execution.

---

## 29. Future Phase if GO

### Zero-Copy One-Cut Wavefront

Desired structure:

```text
Full X
  ↓
Single QKV projection
  ↓
Shared K/V
  ↓
Q_head view → Attention → MoE
Q_tail view → Attention ─────→ MoE
```

Design goals:

- no tensor gather,
- no physical copy,
- no QKV duplication,
- no concatenate,
- shared K/V,
- preallocated outputs,
- event-based producer/consumer synchronization.

A later phase may investigate tile-ready streaming:

```text
Attention tile 0 → MoE tile 0
Attention tile 1 → MoE tile 1
Attention tile 2 → MoE tile 2
```

Do not attempt tile-level kernel work before the coarse one-cut oracle and zero-copy phase succeed.

---

## 30. Research Narrative if Successful

### Observation

Vision-heavy MLLM prefill is structurally asymmetric, but decoder layers execute all tokens monolithically.

### Problem

The full-query attention barrier hides token-level producer–consumer parallelism.

### Why naive splitting fails

Physical token splitting creates:

- data movement,
- repeated launches,
- repeated K/V access,
- fragmented kernels,
- synchronization.

### Insight

> **We do not split multimodal tokens to create parallelism; we expose parallelism that already exists between completed attention outputs and expert execution.**

### Potential method

**Modality-Asymmetric Attention–MoE Wavefront**

using:

- logical query ranges,
- shared K/V,
- zero-copy execution,
- modality-aware split selection,
- resource-aware overlap.

---

## 31. Final Report

Create:

```text
poc_attention_moe_wavefront/reports/final_report.md
```

It must contain:

1. Environment
2. Runtime validation
3. Baseline critical-path breakdown
4. Attention/MoE scaling
5. O1/O2/O3/O4 zero-cost oracle
6. Amdahl-adjusted TTFT oracle
7. Optimal split analysis
8. Modality-boundary relevance
9. Naive split results
10. Overlap efficiency
11. Split-tax decomposition
12. Contention-corrected oracle
13. Limitations
14. Final GO/HOLD/NO-GO

End with exactly one of:

```text
GO
HOLD
NO-GO
```

and justify it using the gates above.
